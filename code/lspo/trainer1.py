import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .latent_space import generate_self_play_data, construct_latent_space
from .policy_optimization import run_policy_optimization
from .expansion import prepare_dpo_dataset, fine_tune_with_dpo 
from game.environment import WerewolfGame
from agents.lspo_agent import LSPOAgent
import config
import pickle
import os

class LSPOTrainer:
    def __init__(self, config):#最初に呼び出される初期化項目
        self.config = config          #configから情報を得る
        self.device = config.DEVICE   #市様子rデバイス情報を得る
        print("Initializing LSPOTrainer...") #初期化開始を通達
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4") #4bit制度でロード/計算時は16bitを用いる/精度の劣化を抑える量子化方式
        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_LLM_MODEL)#対応するトークナイザをhuggingfaceからダウンロード
        self.llm = AutoModelForCausalLM.from_pretrained(config.BASE_LLM_MODEL, quantization_config=quantization_config, device_map="auto", dtype=torch.bfloat16) #torch_dtype→dtypeに修正
        #4bit量子化を適用/モデルをGPUやCPUに自動で配置（1枚のGPUに乗らない大きなモデルも扱える）/モデルの重みを16bit浮動小数店で扱うように指示, メモリ使用量を削減/
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token #トークナイザのパディングトークンを設定（設定されていない場合、文末トークン（eos_token）を使うように設定）
                
        # ⬇⬇⬇ [修正] 根本原因の解決 ⬇⬇⬇
        # device_map="auto" でロードすると、lm_head が float32 になることがある。
        # .to() でのモジュール置換は accelerate 管理下では失敗するため、
        # パラメータのデータ型を「インプレース」で bfloat16 にキャストする。
        if hasattr(self.llm, 'lm_head'):
            print("Casting lm_head to bfloat16 (in-place) to ensure dtype consistency from the start...")
            for param in self.llm.lm_head.parameters():
                param.data = param.data.to(torch.bfloat16)
        # ⬆⬆⬆ [修正] 根本原因の解決 ⬆⬆⬆

        print("Base model and tokenizer loaded successfully.")#モデルとトークナイザの準備が完了
        self.cfr_nets = {}#DEEP CFRのニューラルネットワークを、役職ごとに保存
        self.kmeans_models = {}#ステップ1で学習したk-meansクラスタリングモデルを、役職ごとに保存

    def train(self):#中核部分
        for i in range(self.config.NUM_ITERATIONS):#反復回数だけ学習ループを繰り返す（現在3）
            print(f"--- Starting LSPO Iteration {i+1}/{self.config.NUM_ITERATIONS} ---")#1/3といった形式で表示
            
            # ステップ1：潜在戦略空間の形成
            print("Step 1: Constructing Latent Space...")
            data_filename = f"discussion_data_iter_{i}.pkl"#各イテレーションのデータファイルを定義

            print(f"Generating new self-play data for iteration {i} (forcing re-generation)...")

            agent_components = {'llm': self.llm, 'tokenizer': self.tokenizer, 'device': self.device}#言語モデル/トークナイザ/CPUおよびGPUの設定
            game_config = {'NUM_PLAYERS': self.config.NUM_PLAYERS, 'ROLES': self.config.ROLES}#プレイヤー数、役職の設定
            discussion_data = generate_self_play_data(LSPOAgent, agent_components, game_config, self.config.GAMES_PER_ITERATION_FOR_DATA, iteration_num=i)
            #自己対戦の中核部分 エージェントのクラス/必要な構成要素(2行上に記載)ゲーム設定/イテレーション数/イテレーション回数の記録
            with open(data_filename, 'wb') as f: pickle.dump(discussion_data, f)#書き込み用にファイルを開く/データをそのまま保存/with...as...:によって、開いた後自動的に閉じる
            
            for role in self.config.ROLES:#役職ごとにループ処理開始
                num_clusters = (self.config.INITIAL_K_WEREWOLF if role == "werewolf" else self.config.INITIAL_K_VILLAGE) + i#クラスタリングの数を決定
                self.kmeans_models[role] = construct_latent_space(discussion_data, role, num_clusters)#生成した潜在戦略空間を役職ごとにクラスタリング
            print("Latent space constructed successfully.")


            #新しくk-meansのモデルをファイルに生成（dataファイルと同様）
            kmeans_filename = f"kmeans_models_iter_{i}.pkl"
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
                net_path = f'{role}_net_iter_{i}.pth'
                torch.save(net.state_dict(), net_path)
            
            print("Saved CFR networks successfully.")

            # === Step 3: 潜在空間の拡張 (Latent Space Expansion) ===
            print("Step 3: Latent Space Expansion via DPO Fine-Tuning...")
            print("Preparing DPO preference dataset...")
            dpo_dataset = prepare_dpo_dataset(
                discussion_data,      # ステップ1のデータ
                self.cfr_nets,         # [修正] 引数名を合わせる: role_cfr_nets
                self.kmeans_models,    # [修正] 引数名を合わせる: kmeans_models
                self.device            # [修正] self.config を削除
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
                # ⬇⬇⬇ 修正ブロック ⬇⬇⬇
                # merge_and_unload() は型エラー (Float/BFloat16) と 
                # キャストエラー (ValueError) の両方の原因となるため、
                # トレーニングループの中では実行しない。
                
                # 1. アダプタを保存する
                adapter_path = f"lspo_adapter_iter_{i}"
                if hasattr(fine_tuned_model, 'save_pretrained'):
                    fine_tuned_model.save_pretrained(adapter_path)
                    print(f"Saved DPO adapter to '{adapter_path}'")
                
                # 2. 古いLLM（前のイテレーションのモデル）をVRAMから削除
                del self.llm
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 3. DPOから返された PeftModel を self.llm に設定
                #    (merge_and_unload は実行しない)
                self.llm = fine_tuned_model
                
                # 4. (最重要) DPO(prepare_model...)により float32 になった
                #    PeftModel の「ベースモデル」の lm_head を、
                #    次の推論 (Iter 2, Step 1) のために bfloat16 にキャストし直す。
                if hasattr(self.llm, 'base_model') and hasattr(self.llm.base_model, 'lm_head'):
                    print("Casting lm_head of PeftModel's base_model back to bfloat16 for next iteration's inference...")
                    self.llm.base_model.lm_head = self.llm.base_model.lm_head.to(torch.bfloat16)
                

                print(f"Model updated for Iteration {i+1} (using 4-bit Base + Adapter '{adapter_path}').")
                # ⬆⬆⬆ 修正ブロック ⬆⬆⬆
                
            print(f"--- Finished LSPO Iteration {i+1} ---")

        print("--- All LSPO iterations completed ---")