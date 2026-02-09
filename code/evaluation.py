import torch
import os
import pickle
import numpy as np
import gc
import datetime
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# プロジェクトモジュールのインポート
from game.environment import WerewolfGame
from agents.lspo_agent import LSPOAgent
from agents.base_agent import BaselineAgent
from utils.network import CFRNet
from utils.data_utils import format_obs_to_vector


def load_model_for_eval(model_path, device):
    """
    指定されたパスのモデルをロードする汎用関数。
    """
    print(f"\n[System] Loading Model: {model_path}")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        print(f"[Error] Failed to load tokenizer from {model_path}")
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- 【修正】GPU数に応じた device_map の決定 ---
    # GPUが1枚しか見えていない場合、"auto" だと誤ってCPUオフロードしようとしてエラーになることがあるため、
    # {"": 0} で強制的にGPU 0に割り当てる。
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        print("[System] Single GPU detected. Forcing device_map to GPU 0 or 1.")
        device_map = {"": 0}
    else:
        print("[System] Multiple GPUs detected. Using device_map='auto'.")
        device_map = "auto"
    # ----------------------------------------------
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=device_map,
        dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    if hasattr(model, 'lm_head'):
        for param in model.lm_head.parameters():
            param.data = param.data.to(torch.bfloat16)
            
    print("[System] Model Loaded Successfully.")
    return model, tokenizer


def setup_lspo_components(base_model, tokenizer, iteration, cfg, device, model_dir=None):
    """
    共有ベースモデルにアダプタとCFR/KMeansを追加ロードする。
    """
    iter_idx = iteration - 1
    save_dir = model_dir if model_dir else cfg.MODEL_SAVE_DIR
    
    print(f"\n[Setup] Attaching LSPO components for Iteration {iteration} from {save_dir}...")

    adapter_name = f"lspo_adapter_iter_{iter_idx}"
    kmeans_name = f"kmeans_models_iter_{iter_idx}.pkl"
    
    adapter_path = os.path.join(save_dir, adapter_name)
    kmeans_path = os.path.join(save_dir, kmeans_name)

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"[FATAL] LoRA Adapter not found at: {adapter_path}\nCannot evaluate LSPO agent without trained adapter.")
    
    if not os.path.exists(kmeans_path):
        raise FileNotFoundError(f"[FATAL] KMeans model not found at: {kmeans_path}\nCannot evaluate LSPO agent without latent space.")

    print(f"Loading LoRA adapter: {adapter_path}")
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)
    model_with_adapter.eval()
    
    print(f"Loading KMeans models: {kmeans_path}")
    with open(kmeans_path, 'rb') as f:
        kmeans_models = pickle.load(f)

    cfr_nets = {}
    for role in cfg.ROLES.keys():
        cfr_name = f"{role}_net_iter_{iter_idx}.pth"
        cfr_path = os.path.join(save_dir, cfr_name)
        
        if not os.path.exists(cfr_path):
             print(f"[Warning] CFR Net not found for {role} at {cfr_path}. Assuming not trained or pure LLM role.")
             continue
             
        state_dim = cfg.STATE_DIM
        action_dim = cfg.MAX_ACTION_DIM[role]
        net_device = device
        
        net = CFRNet(state_dim, action_dim).to(net_device)
        net.load_state_dict(torch.load(cfr_path, map_location=net_device))
        net.eval()
        cfr_nets[role] = net
        print(f"Loaded CFR Net for {role}")

    components = {
        'llm': model_with_adapter,
        'tokenizer': tokenizer,
        'device': device,
        'cfr_net': cfr_nets,
        'kmeans': kmeans_models
    }
    
    return components


def run_evaluation(lspo_iteration, baseline_iteration, num_games, cfg, model_dir=None, opponent_model_path=None, only_opponent=False):
    """
    評価実行のメインフロー。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    phase_a_model_path = opponent_model_path if opponent_model_path else cfg.BASE_LLM_MODEL
    phase_b_model_path = cfg.BASE_LLM_MODEL
    
    is_same_model = (phase_a_model_path == phase_b_model_path)
    
    base_model = None
    tokenizer = None

    # ==========================================
    # Phase A: Baseline/Opponent Evaluation
    # ==========================================
    #only_opponent フラグがある場合のみ Phase A を実行
    if only_opponent:
        model_name_a = os.path.basename(phase_a_model_path)
        print(f"\n{'='*40}")
        print(f"Phase A: Evaluating Opponent/Baseline ({model_name_a})")
        print(f"{'='*40}")

        base_model, tokenizer = load_model_for_eval(phase_a_model_path, device)

            # 念のためのガードレール（論理チェック）
        if base_model is None:
            raise RuntimeError(f"[FATAL] Failed to load base model for Phase A from {phase_a_model_path}")

        baseline_components = {
            'llm': base_model,
            'tokenizer': tokenizer,
            'device': device
        }

        print(f"[Log Info] OpponentAgent logs will be saved to: debug_logs_baseline/")

        baseline_results = run_game_loop(
            agent_class=BaselineAgent,
            agent_components=baseline_components,
            num_games=num_games,
            cfg=cfg,
            desc="Opponent Self-Play",
            agent_kwargs={'model_name': model_name_a},  # ここを追加
            log_save_dir="game_logs_baseline",  # 指定ディレクトリ
            log_prefix=model_name_a             # ファイル名の接頭辞
        )

        # 結果の保存と表示 (Phase A)
        process_and_save_results(baseline_results, f"Opponent_{model_name_a}", num_games)

        # 終了後にメモリ解放してリターン（Phase Bは行わない）
        print("\n[Config] Phase A finished. Skipping Phase B as only_opponent=True.")
        del base_model, tokenizer, baseline_components
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return




    '''
    if model_dir:
        save_path = os.path.join(model_dir, "eval_baseline_results.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(baseline_results, f)

    # Phase B スキップ判定
    if only_opponent:
        print("\n[Config] --only_opponent flag is set. Skipping LSPO Agent Evaluation.")
        del base_model, tokenizer, baseline_components
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return

    # 次のフェーズの準備
    if not is_same_model:
        print("[System] Switching models... Unloading Opponent Model.")
        del base_model, tokenizer, baseline_components
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        base_model, tokenizer = load_model_for_eval(phase_b_model_path, device)
    else:
        del baseline_components
        print("[System] Reusing loaded base model for LSPO phase.")
    '''


    # ==========================================
    # Phase B: LSPO Agent Evaluation
    # ==========================================
    #only_opponent が False の場合は最初からここを実行
    model_name_b = f"LSPO_Iter_{lspo_iteration}"
    print(f"\n{'='*40}")
    print(f"Phase B: Evaluating {model_name_b}")
    print(f"{'='*40}")

    base_model, tokenizer = load_model_for_eval(phase_b_model_path, device)

    # 念のためのガードレール（論理チェック）
    if base_model is None:
        raise RuntimeError(f"[FATAL] Failed to load base model for Phase B from {phase_b_model_path}")

    lspo_results = None
    print(f"[System] Iteration {lspo_iteration} detected. Loading trained components...")
    lspo_components = setup_lspo_components(
        base_model=base_model,
        tokenizer=tokenizer,
        iteration=lspo_iteration,
        cfg=cfg,
        device=device,
        model_dir=model_dir
    )
    lspo_results = run_game_loop(
        agent_class=LSPOAgent,
        agent_components=lspo_components,
        num_games=num_games,
        cfg=cfg,
        desc=f"LSPO Iter{lspo_iteration} Self-Play",
        agent_kwargs={'is_eval': True},
        log_save_dir="game_logs_lspo", # 別ディレクトリに保存
        log_prefix=f"LSPO_Iter{lspo_iteration}"
    )

    # 結果の保存と表示 (Phase B)
    process_and_save_results(lspo_results, model_name_b, num_games)

    if model_dir:
        save_path = os.path.join(model_dir, "eval_lspo_results.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(lspo_results, f)

    # 最終的なメモリ解放
    del base_model, tokenizer, lspo_components
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def run_game_loop(agent_class, agent_components, num_games, cfg, desc, agent_kwargs=None, log_save_dir=None, log_prefix="game"):
    """
    詳細な統計情報を収集するように拡張されたゲームループ
    """
    if agent_kwargs is None:
        agent_kwargs = {}

    game = WerewolfGame(num_players=cfg.NUM_PLAYERS, roles_config=cfg.ROLES)
    total_wins = {"werewolf": 0, "village": 0}
    
    # 予測精度の詳細集計用: prediction_stats[predictor_role][target_role] = {'correct': 0, 'total': 0}
    roles_list = ["villager", "werewolf", "seer", "doctor"]
    prediction_stats = {
        pred_role: {target_role: {'correct': 0, 'total': 0} for target_role in roles_list}
        for pred_role in roles_list
    }

    for g_idx in tqdm(range(num_games), desc=desc):
        game.reset()
        agents = {
            p.id: agent_class(p.id, p.role, agent_components, **agent_kwargs)
            for p in game.players
        }

        while not game.is_game_over():
            phase = game.phase
            
            # --- Discussion Phase ---
            if phase == "day_discussion":
                speakers_order = game.get_shuffled_alive_players()
                NUM_DISCUSSION_ROUNDS = 2
                for _ in range(NUM_DISCUSSION_ROUNDS):
                    for player_id in speakers_order:
                        if not game.players[player_id].is_alive: continue
                        obs = game.get_observation_for_player(player_id)
                        action = agents[player_id].get_action(obs, phase, [])
                        game.record_discussion_step(player_id, action.get("statement", "..."))
                game.phase = "day_voting"
                continue
            
            # --- Night / Voting Phase ---
            actor_ids = game.get_actors_for_phase()
            if not actor_ids:
                game.step({})
                continue

            # --- 予測精度の詳細計測 (Voting Phase) ---
            if phase == 'day_voting':
                living_players = game.get_living_players()
                true_roles = game.get_true_roles()
                
                for player_id in living_players:
                    predictor_role = agents[player_id].role
                    obs = game.get_observation_for_player(player_id)
                    predictions = agents[player_id].predict_roles(obs, game_idx=g_idx)
                    
                    for target_id, predicted_role in predictions.items():
                        # 自分自身、死亡者、正しくないIDはスキップ
                        if target_id not in true_roles or target_id not in living_players or target_id == player_id:
                            continue
                        
                        target_true_role = true_roles[target_id]
                        
                        # 統計更新
                        if predictor_role in prediction_stats and target_true_role in prediction_stats[predictor_role]:
                            prediction_stats[predictor_role][target_true_role]['total'] += 1
                            if predicted_role == target_true_role:
                                prediction_stats[predictor_role][target_true_role]['correct'] += 1

            # --- Action Execution ---
            actions_to_submit = {}
            for player_id in actor_ids:
                if not game.players[player_id].is_alive: continue
                obs = game.get_observation_for_player(player_id)
                avail_actions = game.get_available_actions(player_id)
                action = agents[player_id].get_action(obs, phase, avail_actions)
                actions_to_submit[player_id] = action
            
            game.step(actions_to_submit)

        winner = game.get_winner()
        if winner:
            total_wins[winner] += 1
        
        # ▼▼▼ 追加: ゲームログの保存 ▼▼▼
        if log_save_dir:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            # ファイル名に使えない文字を置換
            safe_prefix = log_prefix.replace("/", "_").replace(" ", "_")
            log_filename = f"{safe_prefix}_game_{num_games}_{timestamp}.json"
            log_path = os.path.join(log_save_dir, log_filename)
            
            log_data = {
                "game_id": num_games,
                "winner": winner,
                "roles": {p.id: p.role for p in game.players},
                "log": game.game_log
            }
            
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        
    results = {
        "total_games": num_games,
        "werewolf_wins": total_wins["werewolf"],
        "village_wins": total_wins["village"],
        "prediction_stats": prediction_stats
    }
    return results


def process_and_save_results(results, model_name, num_games):
    """
    結果を集計し、フォーマットして表示・ファイル保存を行う関数
    """
    if not results:
        print("No results to process.")
        return

    # --- 1. 勝率の計算 ---
    ww_wins = results["werewolf_wins"]
    v_wins = results["village_wins"]
    total = results["total_games"]
    
    ww_rate = (ww_wins / total) * 100 if total > 0 else 0
    v_rate = (v_wins / total) * 100 if total > 0 else 0
    
    # --- 2. 予測精度の計算 (全体 & マトリックス) ---
    pred_stats = results["prediction_stats"]
    
    total_correct = 0
    total_attempts = 0
    
    # 文字列バッファに結果を構築
    output_lines = []
    output_lines.append(f"Evaluation Results for: {model_name}")
    output_lines.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Total Games: {num_games}")
    output_lines.append("-" * 40)
    
    # 勝率表示
    output_lines.append("【勝率】")
    output_lines.append(f"  村人   : {v_rate:.2f}%")
    output_lines.append(f"  人狼   : {ww_rate:.2f}%")
    output_lines.append(f"  占い師 : {v_rate:.2f}%")
    output_lines.append(f"  Doctor : {v_rate:.2f}%")
    output_lines.append(f"  [村人派閥の勝率合計] : {v_rate:.2f}%")
    output_lines.append("-" * 40)

    # 予測精度マトリックス表示
    output_lines.append("【予測精度（各役職に対する予測精度）】")
    
    # ヘッダー
    roles = ["villager", "werewolf", "seer", "doctor"]
    roles_display = {"villager": "村人", "werewolf": "人狼", "seer": "占い師", "doctor": "Doc "}
    
    header = "          " + " | ".join([f"{roles_display[r]:^6}" for r in roles]) + " |"
    output_lines.append(header)
    output_lines.append("-" * len(header))
    
    for pred_role in roles:
        row_str = f"{roles_display[pred_role]:<8} :"
        for target_role in roles:
            # ▼▼▼ 修正: 不可能な予測（自分一人しかいない役職）を除外 ▼▼▼
            if pred_role == target_role and pred_role in ['seer', 'doctor']:
                row_str += f"   N/A |"
                continue

            stats = pred_stats[pred_role][target_role]
            correct = stats['correct']
            attempts = stats['total']
            
            total_correct += correct
            total_attempts += attempts
            
            acc = (correct / attempts * 100) if attempts > 0 else 0.0
            row_str += f" {acc:>5.1f}% |"
        output_lines.append(row_str)
    
    output_lines.append("-" * 40)
    
    # 全体予測精度
    total_acc = (total_correct / total_attempts * 100) if total_attempts > 0 else 0.0
    output_lines.append(f"予測精度の合計 : {total_acc:.2f}%")
    output_lines.append("=" * 40)

    # --- 3. 出力と保存 ---
    # コンソール出力
    full_output = "\n".join(output_lines)
    print("\n" + full_output)
    
    # ファイル保存
    os.makedirs("results", exist_ok=True)
    # ファイル名に使えない文字を置換
    safe_model_name = model_name.replace("/", "_").replace(" ", "_")
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"results/eval_{safe_model_name}_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_output)
    
    print(f"\n[System] Detailed results saved to: {filename}")
'''
import torch
import os
import pickle
import numpy as np
import gc
import argparse  # 追加
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# プロジェクトモジュールのインポート
from game.environment import WerewolfGame
from agents.lspo_agent import LSPOAgent
from agents.base_agent import BaselineAgent
from utils.network import CFRNet
from utils.data_utils import format_obs_to_vector


def load_shared_base_model(cfg, device):
    """
    ベースモデルを一度だけロードして共有するための関数。
    """
    print(f"\n[System] Loading Shared Base Model: {cfg.BASE_LLM_MODEL}")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_LLM_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_LLM_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.bfloat16
    )
    
    # 数値安定性のための処理
    if hasattr(model, 'lm_head'):
        for param in model.lm_head.parameters():
            param.data = param.data.to(torch.bfloat16)
            
    print("[System] Shared Base Model Loaded Successfully.")
    return model, tokenizer


def setup_lspo_components(base_model, tokenizer, iteration, cfg, device, model_dir=None):
    """
    共有ベースモデルにアダプタとCFR/KMeansを追加ロードする。
    """
    iter_idx = iteration - 1
    save_dir = model_dir if model_dir else cfg.MODEL_SAVE_DIR
    
    print(f"\n[Setup] Attaching LSPO components for Iteration {iteration} from {save_dir}...")

    # --- パス定義 ---
    adapter_name = f"lspo_adapter_iter_{iter_idx}"
    kmeans_name = f"kmeans_models_iter_{iter_idx}.pkl"
    
    adapter_path = os.path.join(save_dir, adapter_name)
    kmeans_path = os.path.join(save_dir, kmeans_name)

    # --- ファイルパスの厳格化 ---
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"[FATAL] LoRA Adapter not found at: {adapter_path}\nCannot evaluate LSPO agent without trained adapter.")
    
    if not os.path.exists(kmeans_path):
        raise FileNotFoundError(f"[FATAL] KMeans model not found at: {kmeans_path}\nCannot evaluate LSPO agent without latent space.")

    # --- 1. アダプタの装着 (PeftModel化) ---
    print(f"Loading LoRA adapter: {adapter_path}")
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)
    model_with_adapter.eval()
    
    # --- 2. KMeansのロード ---
    print(f"Loading KMeans models: {kmeans_path}")
    with open(kmeans_path, 'rb') as f:
        kmeans_models = pickle.load(f)

    # --- 3. CFRモデルのロード ---
    cfr_nets = {}
    for role in cfg.ROLES.keys():
        cfr_name = f"{role}_net_iter_{iter_idx}.pth"
        cfr_path = os.path.join(save_dir, cfr_name)
        
        if not os.path.exists(cfr_path):
             print(f"[Warning] CFR Net not found for {role} at {cfr_path}. Assuming not trained or pure LLM role.")
             continue
             
        state_dim = cfg.STATE_DIM
        action_dim = cfg.MAX_ACTION_DIM[role]
        net_device = device
        
        net = CFRNet(state_dim, action_dim).to(net_device)
        net.load_state_dict(torch.load(cfr_path, map_location=net_device))
        net.eval()
        cfr_nets[role] = net
        print(f"Loaded CFR Net for {role}")

    components = {
        'llm': model_with_adapter,
        'tokenizer': tokenizer,
        'device': device,
        'cfr_net': cfr_nets,
        'kmeans': kmeans_models
    }
    
    return components


def run_evaluation(lspo_iteration, baseline_iteration, num_games, cfg, model_dir=None):
    """
    評価実行のメインフロー。
    Iter 0 の場合は BaselineAgent を使用するように分岐を追加。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 共有ベースモデルのロード
    base_model, tokenizer = load_shared_base_model(cfg, device)
    
    baseline_results = None
    lspo_results = None

    # ==========================================
    # Phase A: Baseline Agent Evaluation (Iter 0)
    # ==========================================
    print(f"\n{'='*40}")
    print(f"Phase A: Evaluating Baseline Agent (Iter {baseline_iteration})")
    print(f"{'='*40}")

    baseline_components = {
        'llm': base_model,
        'tokenizer': tokenizer,
        'device': device
    }

    # BaselineAgentは "debug_logs_baseline" にログを出力する
    print(f"[Log Info] BaselineAgent logs will be saved to: debug_logs_baseline/")

    baseline_results = run_game_loop(
        agent_class=BaselineAgent,
        agent_components=baseline_components,
        num_games=num_games,
        cfg=cfg,
        desc="Baseline Self-Play"
    )

    if model_dir:
        save_path = os.path.join(model_dir, "eval_baseline_results.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(baseline_results, f)

    # メモリ掃除
    del baseline_components
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


    # ==========================================
    # Phase B: LSPO Agent Evaluation (Iter X or 0)
    # ==========================================
    print(f"\n{'='*40}")
    print(f"Phase B: Evaluating Target Agent (Iter {lspo_iteration})")
    print(f"{'='*40}")

    # ★修正ポイント: Iter 0 が指定された場合は BaselineAgent として動かす
    if lspo_iteration == 0:
        print(f"[Config] Iteration 0 detected. Using BaselineAgent (No Adapter) for Phase B.")
        print(f"[Log Info] Logs will be saved to: debug_logs_baseline/")
        
        # コンポーネントはBaseモデルそのまま
        lspo_components = {
            'llm': base_model,
            'tokenizer': tokenizer,
            'device': device
        }
        
        lspo_results = run_game_loop(
            agent_class=BaselineAgent,
            agent_components=lspo_components,
            num_games=num_games,
            cfg=cfg,
            desc="Iter 0 (Baseline) Self-Play"
        )
        
    else:
        # 通常のLSPO評価 (Iter 1以上)
        print(f"[Config] Iteration {lspo_iteration} detected. Loading Adapter and using LSPOAgent.")
        print(f"[Log Info] LSPOAgent logs will be saved to: debug_logs/ and prompt_logs/")
        
        lspo_components = setup_lspo_components(
            base_model=base_model,
            tokenizer=tokenizer,
            iteration=lspo_iteration,
            cfg=cfg,
            device=device,
            model_dir=model_dir
        )

        lspo_results = run_game_loop(
            agent_class=LSPOAgent,
            agent_components=lspo_components,
            num_games=num_games,
            cfg=cfg,
            desc="LSPO Self-Play",
            agent_kwargs={'is_eval': True}
        )

    if model_dir:
        save_path = os.path.join(model_dir, "eval_lspo_results.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(lspo_results, f)

    # ==========================================
    # Final Report
    # ==========================================
    print("\n" + "="*30)
    print("--- FINAL EVALUATION RESULTS ---")
    print(f"Baseline (Iter {baseline_iteration}):")
    print_results(baseline_results)
    print(f"Target (Iter {lspo_iteration}):")
    print_results(lspo_results)
    print("="*30)


def run_game_loop(agent_class, agent_components, num_games, cfg, desc, agent_kwargs=None):
    """
    汎用的なゲームループ
    """
    if agent_kwargs is None:
        agent_kwargs = {}

    game = WerewolfGame(num_players=cfg.NUM_PLAYERS, roles_config=cfg.ROLES)
    total_wins = {"werewolf": 0, "village": 0}
    all_accuracies = []

    for _ in tqdm(range(num_games), desc=desc):
        game.reset()
        
        agents = {
            p.id: agent_class(p.id, p.role, agent_components, **agent_kwargs)
            for p in game.players
        }
        
        game_accuracies = []

        while not game.is_game_over():
            phase = game.phase
            
            # --- 議論フェーズ ---
            if phase == "day_discussion":
                speakers_order = game.get_shuffled_alive_players()
                NUM_DISCUSSION_ROUNDS = 2
                
                for _ in range(NUM_DISCUSSION_ROUNDS):
                    for player_id in speakers_order:
                        if not game.players[player_id].is_alive:
                            continue
                        
                        obs = game.get_observation_for_player(player_id)
                        current_agent = agents[player_id]
                        
                        action = current_agent.get_action(obs, phase, [])
                        
                        statement = action.get("statement", "...")
                        game.record_discussion_step(player_id, statement)
                
                game.phase = "day_voting"
                continue
            
            # --- 夜・投票フェーズ ---
            actor_ids = game.get_actors_for_phase()
            if not actor_ids:
                game.step({})
                continue

            if phase == 'day_voting':
                living_players = game.get_living_players()
                true_roles = game.get_true_roles()
                
                for player_id in living_players:
                    agent = agents[player_id]
                    obs = game.get_observation_for_player(player_id)
                    predictions = agent.predict_roles(obs)
                    
                    correct = 0; total = 0
                    for target_id, predicted_role in predictions.items():
                        if target_id in true_roles and target_id in living_players and target_id != player_id:
                            total += 1
                            if true_roles[target_id] == predicted_role:
                                correct += 1
                    if total > 0:
                        game_accuracies.append(correct / total)

            actions_to_submit = {}
            for player_id in actor_ids:
                if not game.players[player_id].is_alive: continue
                
                agent = agents[player_id]
                obs = game.get_observation_for_player(player_id)
                avail_actions = game.get_available_actions(player_id)
                
                action = agent.get_action(obs, phase, avail_actions)
                actions_to_submit[player_id] = action
            
            game.step(actions_to_submit)

        winner = game.get_winner()
        if winner:
            total_wins[winner] += 1
        
        if game_accuracies:
            all_accuracies.append(np.mean(game_accuracies))

    results = {
        "total_games": num_games,
        "werewolf_wins": total_wins["werewolf"],
        "village_wins": total_wins["village"],
        "prediction_accuracy": np.mean(all_accuracies) if all_accuracies else 0
    }
    return results


def print_results(results):
    if not results:
        print("No results available.")
        return
    total = results["total_games"]
    ww_wins = results["werewolf_wins"]
    v_wins = results["village_wins"]
    
    ww_win_rate = (ww_wins / total) * 100 if total > 0 else 0
    v_win_rate = (v_wins / total) * 100 if total > 0 else 0
    
    print(f"  Win Rate (Werewolf Side): {ww_win_rate:.2f}% ({ww_wins}/{total})")
    print(f"  Win Rate (Village Side):  {v_win_rate:.2f}% ({v_wins}/{total})")
    print(f"  Prediction Accuracy:      {results['prediction_accuracy']:.2f}%")
'''