#trainer.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel # 追加: アダプタ読み込み用
from .latent_space import generate_self_play_data, construct_latent_space
from .policy_optimization import run_policy_optimization
from .expansion import prepare_dpo_dataset, fine_tune_with_dpo 
from game.environment import WerewolfGame
from agents.lspo_agent import LSPOAgent
import config
import pickle
import os
import gc # 追加: メモリ解放用
import shutil
import datetime

class LSPOTrainer:
    def __init__(self, config):#最初に呼び出される初期化項目
        self.config = config          #configから情報を得る
        self.device = config.DEVICE   #使用するデバイス情報を得る
        print("Initializing LSPOTrainer...") #初期化開始を通達
        
        # 再ロード時にも使用するため、self に保存する
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4") #4bit制度でロード/計算時は16bitを用いる/精度の劣化を抑える量子化方式
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_LLM_MODEL)#対応するトークナイザをhuggingfaceからダウンロード
        self.llm = AutoModelForCausalLM.from_pretrained(config.BASE_LLM_MODEL, quantization_config=self.quantization_config, device_map="auto", dtype=torch.bfloat16) #torch_dtype→dtypeに修正
        #4bit量子化を適用/モデルをGPUやCPUに自動で配置（1枚のGPUに乗らない大きなモデルも扱える）/モデルの重みを16bit浮動小数店で扱うように指示, メモリ使用量を削減/
        
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token #トークナイザのパディングトークンを設定（設定されていない場合、文末トークン（eos_token）を使うように設定）
                

        # device_map="auto" でロードすると、lm_head が float32 になることがある。
        # パラメータのデータ型を「インプレース」で bfloat16 にキャストし、初期状態を安定させる。
        if hasattr(self.llm, 'lm_head'):
            print("Initial load: Casting lm_head to bfloat16 (in-place) to ensure dtype consistency...")
            for param in self.llm.lm_head.parameters():
                param.data = param.data.to(torch.bfloat16)


        print("Base model and tokenizer loaded successfully.")#モデルとトークナイザの準備が完了
        self.cfr_nets = {}#DEEP CFRのニューラルネットワークを、役職ごとに保存
        self.kmeans_models = {}#ステップ1で学習したk-meansクラスタリングモデルを、役職ごとに保存

    def train(self, start_game_idx=0, num_games_limit=None):#中核部分
        for i in range(self.config.NUM_ITERATIONS):#反復回数だけ学習ループを繰り返す（現在3）
            # 既存の学習結果(AdapterとKMeans)が存在するか確認し、あればスキップ・ロードする (Resume機能)
            adapter_path = os.path.join(self.config.MODEL_SAVE_DIR, f"lspo_adapter_iter_{i}")
            kmeans_path = os.path.join(self.config.MODEL_SAVE_DIR, f"kmeans_models_iter_{i}.pkl")

            if os.path.exists(adapter_path) and os.path.exists(kmeans_path):
                print(f"\n[RESUME] Artifacts for Iteration {i+1} found at '{adapter_path}'. Skipping training steps.")
                
                # 次のイテレーションのために、スキップしたイテレーションの成果物(Adapter)をロードする
                print(f"Loading existing adapter '{adapter_path}' to prepare model state for next iteration...")

                # メモリリーク防止のため、現在のモデルを破棄
                del self.llm
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

                # ベースモデルを再ロード
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.config.BASE_LLM_MODEL, 
                    quantization_config=self.quantization_config, 
                    device_map="auto", 
                    dtype=torch.bfloat16
                )
                
                # lm_headの型合わせ
                if hasattr(self.llm, 'lm_head'):
                    for param in self.llm.lm_head.parameters():
                        param.data = param.data.to(torch.bfloat16)

                # 発見されたAdapterをロード
                self.llm = PeftModel.from_pretrained(self.llm, adapter_path)
                self.llm.eval()
                
                print(f"Model state restored to end of Iteration {i+1}. Proceeding to next loop.")
                continue


            print(f"--- Starting LSPO Iteration {i+1}/{self.config.NUM_ITERATIONS} ---")#1/3といった形式で表示
            
            games_to_generate = num_games_limit if num_games_limit is not None else self.config.GAMES_PER_ITERATION_FOR_DATA #追加

            # ステップ1：潜在戦略空間の形成
            print("Step 1: Constructing Latent Space...")
            data_filename = os.path.join(self.config.MODEL_SAVE_DIR, f"discussion_data_iter_{i}.pkl")  #修正点
            #data_filename = f"discussion_data_iter_{i}.pkl"#各イテレーションのデータファイルを定義

            print(f"Generating new self-play data for iteration {i} (forcing re-generation)...")

            # クリーン・リロード戦略により、ここでの self.llm は常に bfloat16 に統一されている
            agent_components = {'llm': self.llm, 'tokenizer': self.tokenizer, 'device': self.device}#言語モデル/トークナイザ/CPUおよびGPUの設定
            game_config = {'NUM_PLAYERS': self.config.NUM_PLAYERS, 'ROLES': self.config.ROLES}#プレイヤー数、役職の設定
            discussion_data = generate_self_play_data(LSPOAgent, agent_components, game_config, self.config.GAMES_PER_ITERATION_FOR_DATA, iteration_num=i, start_game_idx=start_game_idx)
            #自己対戦の中核部分 エージェントのクラス/必要な構成要素(2行上に記載)ゲーム設定/イテレーション数/イテレーション回数の記録
            with open(data_filename, 'wb') as f: pickle.dump(discussion_data, f)#書き込み用にファイルを開く/データをそのまま保存/with...as...:によって、開いた後自動的に閉じる
            
            for role in self.config.ROLES:#役職ごとにループ処理開始
                #num_clusters = (self.config.INITIAL_K_WEREWOLF if role == "werewolf" else self.config.INITIAL_K_VILLAGE) + i#クラスタリングの数を決定
                #self.kmeans_models[role] = construct_latent_space(discussion_data, role, num_clusters)#生成した潜在戦略空間を役職ごとにクラスタリング
                print(f"Constructing latent space for role: {role}...")
                self.kmeans_models[role] = construct_latent_space(
                    discussion_data, 
                    role, 
                    num_clusters=None  # Noneを渡して自動決定をトリガー
                )
        
            
            print("Latent space constructed successfully.")


            #新しくk-meansのモデルをファイルに生成（dataファイルと同様）
            kmeans_filename = os.path.join(self.config.MODEL_SAVE_DIR, f"kmeans_models_iter_{i}.pkl")  #修正点
            #kmeans_filename = f"kmeans_models_iter_{i}.pkl"
            print(f"Saving kmeans models for iteration {i} to '{kmeans_filename}'...")
            with open(kmeans_filename, 'wb') as f:
                pickle.dump(self.kmeans_models, f)
            print("Saved kmeans models successfully.")


            # === Step 2: 潜在空間での方針最適化 (Policy Optimization) ===
            print("Step 2: Policy Optimization in Latent Space...")
            self.cfr_nets = run_policy_optimization(
                self.config,
                self.kmeans_models,
                discussion_data,
                self.device
            )

            print(f"Saving CFR networks for iteration {i} to project root...")
            for role, net in self.cfr_nets.items():
                # ファイル名にイテレーション番号を追加 (例: seer_net_iter_0.pth)
                net_path = os.path.join(self.config.MODEL_SAVE_DIR, f'{role}_net_iter_{i}.pth') #修正点
                #net_path = f'{role}_net_iter_{i}.pth'
                torch.save(net.state_dict(), net_path)
            
            print("Saved CFR networks successfully.")

            # === Step 3: 潜在空間の拡張 (Latent Space Expansion) ===
            print("Step 3: Latent Space Expansion via DPO Fine-Tuning...")
            print("Preparing DPO preference dataset...")
            dpo_dataset = prepare_dpo_dataset(
                discussion_data,      # ステップ1のデータ
                self.cfr_nets,         # 引数名を合わせる: role_cfr_nets
                self.kmeans_models,    # 引数名を合わせる: kmeans_models
                self.device            # self.config を削除
            )
            
            if dpo_dataset is None or len(dpo_dataset) == 0:
                print("Warning: No DPO data generated. Skipping fine-tuning for this iteration.")
            else:
                print(f"DPO dataset prepared with {len(dpo_dataset)} samples.")

                print("Starting DPO fine-tuning...")
                
                fine_tuned_model = fine_tune_with_dpo(
                    self.llm,
                    self.tokenizer,
                    dpo_dataset,
                    self.config # DPO/LoRA設定を渡す
                )
                

                
                # 1. アダプタをディスクに保存 (学習成果の確保)
                adapter_path = os.path.join(self.config.MODEL_SAVE_DIR, f"lspo_adapter_iter_{i}") #修正点
                #adapter_path = f"lspo_adapter_iter_{i}"
                if hasattr(fine_tuned_model, 'save_pretrained'):
                    fine_tuned_model.save_pretrained(adapter_path)
                    print(f"Saved DPO adapter to '{adapter_path}'")
                
                # 2. メモリ上の「汚れた」モデル(float32混在など)を完全破棄
                print("Resetting model for next iteration to clear DPO artifacts...")
                del self.llm
                del fine_tuned_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 3. ベースモデルをクリーンな状態で再ロード (__init__と同じ設定)
                print("Reloading clean base model...")
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.config.BASE_LLM_MODEL, 
                    quantization_config=self.quantization_config, 
                    device_map="auto", 
                    dtype=torch.bfloat16
                )
                
                # 4. 再ロードしたモデルの lm_head を bfloat16 に統一
                if hasattr(self.llm, 'lm_head'):
                    print("Casting reloaded lm_head to bfloat16 (in-place)...")
                    for param in self.llm.lm_head.parameters():
                        param.data = param.data.to(torch.bfloat16)

                # 5. 保存しておいたアダプタを装着して復元
                print(f"Loading adapter '{adapter_path}' to clean base model...")
                self.llm = PeftModel.from_pretrained(self.llm, adapter_path)
                
                # 6. 推論モードに設定
                self.llm.eval()
                
                print(f"Model successfully reloaded and updated for Iteration {i+1}.")

                
            print(f"--- Finished LSPO Iteration {i+1} ---")

        print("--- All LSPO iterations completed ---")

        # 全イテレーション完了後のアーカイブ移動処理
        try:
            timestamp = datetime.datetime.now().strftime('%m%d_%H%M') # 例: 0122_1430
            archive_path = os.path.join(self.config.MODEL_ARCHIVE_DIR, timestamp)
            
            print(f"\n[ARCHIVE] Moving trained models to archive: {archive_path}")
            
            # modelsディレクトリごと移動 (リネーム) してアーカイブ化
            shutil.move(self.config.MODEL_SAVE_DIR, archive_path)
            
            # 空になった models ディレクトリを再作成 (次回の新規学習用)
            os.makedirs(self.config.MODEL_SAVE_DIR, exist_ok=True)
            
            print("[ARCHIVE] Successfully archived models. 'models/' directory is now empty for new training.")
            
        except Exception as e:
            print(f"[ERROR] Failed to archive models: {e}")
            # 失敗しても学習データ自体は残るように、ここではエラーを表示して続行
