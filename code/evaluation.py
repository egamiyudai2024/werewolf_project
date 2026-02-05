# evaluation.py
import torch
import os
import pickle
import numpy as np
import gc  # メモリ解放用
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from utils.data_utils import format_obs_to_vector

# プロジェクトモジュールのインポート
from game.environment import WerewolfGame
from agents.lspo_agent import LSPOAgent
from utils.network import CFRNet 

def load_agent_components(iteration, cfg, device, model_dir=None):
    """
    指定されたイテレーションに必要なモデル（LLM, CFR, Kmeans）をロードします。
    config.py の MODEL_SAVE_DIR を参照するため、パスエラーが起きません。
    """
    print(f"Loading agent components for Iteration {iteration}...")

    # --- 1. LLMのロード (4bit量子化) ---
    print("Loading base model and tokenizer with 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_quant_type="nf4"
    )
    
    base_model_path = cfg.BASE_LLM_MODEL 
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto", 
        dtype=torch.bfloat16 
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 数値安定性のための処理 (Version Bの良い点を取り込み)
    if hasattr(model, 'lm_head'):
        for param in model.lm_head.parameters():
            param.data = param.data.to(torch.bfloat16)

    print("Base model and tokenizer loaded.")

    components = {
        'llm': model,
        'tokenizer': tokenizer,
        'device': device,
        'cfr_net': {},
        'kmeans': {}
    }

    # イテレーション0（ベースライン）の場合はここで終了
    if iteration == 0:
        print("Using Base Model (Iteration 0). No adapters or CFR models loaded.")
        return components

    # --- 2. モデルパスの特定 (Version Aのロジック + config参照) ---
    iter_idx = iteration - 1
    
    # 保存先ディレクトリ（指定があればそれを使用、なければデフォルト）
    if model_dir:
        save_dir = model_dir
        print(f"Using custom model directory: {save_dir}")
    else:
        save_dir = cfg.MODEL_SAVE_DIR

    # 保存先ディレクトリ（絶対パス）
    #save_dir = cfg.MODEL_SAVE_DIR
    
    # フォルダ/ファイル名
    adapter_name = f"lspo_adapter_iter_{iter_idx}"
    kmeans_name = f"kmeans_models_iter_{iter_idx}.pkl"
    
    # 絶対パスを作成
    adapter_path = os.path.join(save_dir, adapter_name)
    kmeans_path = os.path.join(save_dir, kmeans_name)

    # (A) LoRAアダプタのロード
    if os.path.exists(adapter_path):
        print(f"Loading LoRA adapter from: {adapter_path}")
        # merge_and_unload() は行わず、PeftModelとしてロード (Version Bの安全策)
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        components['llm'] = model
        print("Adapter loaded successfully.")
    else:
        print(f"Warning: Adapter path not found at '{adapter_path}'. Using base model.")

    # (B) KMeansモデルのロード
    if os.path.exists(kmeans_path):
        with open(kmeans_path, 'rb') as f:
            components['kmeans'] = pickle.load(f)
        print(f"KMeans models loaded from: {kmeans_path}")
    else:
        print(f"Warning: KMeans models not found at '{kmeans_path}'.")

    dummy_game = WerewolfGame(num_players=cfg.NUM_PLAYERS, roles_config=cfg.ROLES)
    dummy_obs = dummy_game.get_observation_for_player(0)
    actual_state_dim = len(format_obs_to_vector(dummy_obs)) # ここで「85」が取得される
    print(f"DEBUG: Detected State Dimension for Evaluation: {actual_state_dim}")

    # (C) CFRモデル（役職ごと）のロード
    for role in cfg.ROLES.keys():
        cfr_name = f"{role}_net_iter_{iter_idx}.pth"
        cfr_path = os.path.join(save_dir, cfr_name)
        
        if os.path.exists(cfr_path):
            #state_dim = cfg.STATE_DIM 
            state_dim = actual_state_dim
            action_dim = cfg.MAX_ACTION_DIM[role]
            
            # モデルと同じデバイスに配置
            net_device = model.device if hasattr(model, 'device') else device
            net = CFRNet(state_dim, action_dim).to(net_device)
            
            net.load_state_dict(torch.load(cfr_path, map_location=net_device))
            net.eval()
            components['cfr_net'][role] = net
            print(f"CFR model for {role} loaded from: {cfr_path}")
        else:
            print(f"Warning: CFR model not found: {cfr_name}")

    return components


def run_evaluation(lspo_iteration, baseline_iteration, num_games, cfg, model_dir=None):
    """
    評価プロセスのメイン関数。
    メモリ管理（逐次実行）を徹底しています。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ==========================================
    # 1. LSPOエージェント（学習済み）の評価
    # ==========================================
    print(f"\n--- Evaluating LSPO Agent (Iter {lspo_iteration}) Self-Play ---")
    
    # 学習済みモデルをロード
    lspo_components = load_agent_components(lspo_iteration, cfg, device, model_dir=model_dir)
    #lspo_components = load_agent_components(lspo_iteration, cfg, device)
    
    # is_eval=True で実行 (CFR戦略を使用)
    lspo_results = run_game_loop(lspo_components, num_games, cfg, is_eval=True)
    
    if model_dir is not None:
        save_path = os.path.join(model_dir, "eval_lspo_results.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(lspo_results, f)
        print(f"[Saved] LSPO evaluation results -> {save_path}")

    
    # ★メモリ解放 (重要: これにより2つのモデルを同時に持たなくて済む)
    del lspo_components
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ==========================================
    # 2. ベースラインエージェント（未学習）の評価
    # ==========================================
    print(f"\n--- Evaluating Baseline Agent (Iter {baseline_iteration}) Self-Play ---")
    
    # ベースラインモデルをロード
    #baseline_components = load_agent_components(baseline_iteration, cfg, device)
    baseline_components = load_agent_components(baseline_iteration, cfg, device, model_dir=model_dir)

    # is_eval=False で実行 (ランダム戦略)
    baseline_results = run_game_loop(baseline_components, num_games, cfg, is_eval=False)
    
    if model_dir is not None:
        save_path = os.path.join(model_dir, "eval_baseline_results.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(baseline_results, f)
        print(f"[Saved] Baseline evaluation results -> {save_path}")
        
    # メモリ解放
    del baseline_components
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ==========================================
    # 3. 最終結果の表示
    # ==========================================
    print("\n" + "="*30)
    print("--- FINAL EVALUATION RESULTS ---")
    print(f"Baseline Agent (Iter {baseline_iteration}) ({num_games} games):")
    print_results(baseline_results)
    
    print(f"\nLSPO Agent (Iter {lspo_iteration}) ({num_games} games):")
    print_results(lspo_results)
    print("="*30)


def run_game_loop(agent_components, num_games, cfg, is_eval=False):
    """
    指定されたコンポーネントを使って自己対戦を実行。
    Version Bの「議論フェーズの修正」も取り込んでいます。
    """
    game = WerewolfGame(num_players=cfg.NUM_PLAYERS, roles_config=cfg.ROLES)
    
    total_wins = {"werewolf": 0, "village": 0}


    # 【修正】役職別の正解数と総数を管理する辞書
    # キー: target_role (werewolf, seer, doctor, villager)
    # 値: [correct_count, total_count]
    role_prediction_stats = {
        "werewolf": [0, 0],
        "seer":     [0, 0],
        "doctor":   [0, 0],
        "villager": [0, 0]
    }


    all_accuracies = []
    
    for _ in tqdm(range(num_games), desc="Playing Games"):
        game.reset()
        
        # エージェント作成
        agents = {
            p.id: LSPOAgent(p.id, p.role, agent_components, is_eval=is_eval) 
            for p in game.players
        }
        
        game_accuracies = [] 

        while not game.is_game_over():
            phase = game.phase
            
            # --- 議論フェーズ (Version Bの明示的なロジックを採用) ---
            if phase == "day_discussion":
                speakers_order = game.get_shuffled_alive_players()
                NUM_DISCUSSION_ROUNDS = 2
                
                for _ in range(NUM_DISCUSSION_ROUNDS):
                    for player_id in speakers_order:
                        if not game.players[player_id].is_alive:
                            continue
                        
                        obs = game.get_observation_for_player(player_id)
                        current_agent = agents[player_id]
                        
                        # 議論アクション決定
                        action = current_agent.get_action(obs, phase, [])
                        statement = action.get("statement", "...")
                        
                        # 発言を即座に記録
                        game.record_discussion_step(player_id, statement)
                
                game.phase = "day_voting"
                continue
            
            # --- その他のフェーズ (一斉処理) ---
            actor_ids = game.get_actors_for_phase()

            if not actor_ids:
                game.step({})
                continue

            # 予測精度の測定 (投票前)
            if phase == 'day_voting':
                living_players = game.get_living_players()
                true_roles = game.get_true_roles()
                
                for player_id in living_players:
                    agent = agents[player_id]
                    obs = game.get_observation_for_player(player_id)
                    predictions = agent.predict_roles(obs)


                    # --- [修正後] (役職ごとの統計に加算) ---
                    for target_id, predicted_role in predictions.items():
                        # 生きていて、自分以外で、正しい役職リストにある場合のみ評価
                        if target_id in true_roles and target_id in living_players and target_id != player_id:
                            actual_role = true_roles[target_id]
                            
                            # 統計辞書に加算
                            if actual_role in role_prediction_stats:
                                role_prediction_stats[actual_role][1] += 1 # Total加算
                                if actual_role == predicted_role:
                                    role_prediction_stats[actual_role][0] += 1 # Correct加算


                    #correct = 0
                    #total = 0
                    #for target_id, predicted_role in predictions.items():
                    #    if target_id in true_roles and target_id in living_players and target_id != player_id:
                    #        total += 1
                    #        if true_roles[target_id] == predicted_role:
                    #            correct += 1
                    #if total > 0:
                    #    game_accuracies.append(correct / total)

            # アクション実行
            actions_to_submit = {}
            for player_id in actor_ids:
                if not game.players[player_id].is_alive:
                    continue 
                    
                agent = agents[player_id]
                obs = game.get_observation_for_player(player_id)
                avail_actions = game.get_available_actions(player_id)
                
                action = agent.get_action(obs, phase, avail_actions)
                actions_to_submit[player_id] = action

                # ▼▼▼ 2. CFR投票精度の測定 (投票実行時) ▼▼▼
                # (戦略モデルとしての正しさを測る)
                if phase == 'day_voting':
                    my_role = true_roles[player_id]
                    # 村人陣営のみ評価対象 (人狼を見つけられているか)
                    if my_role != 'werewolf':
                        vote_target = action_dict.get('vote')
                        if vote_target is not None:
                            # 投票先が人狼であれば正解 (1.0)、そうでなければ不正解 (0.0)
                            if vote_target in werewolf_ids:
                                game_cfr_acc.append(1.0)
                            else:
                                game_cfr_acc.append(0.0)
            
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
    """結果表示用"""
    total = results["total_games"]
    ww_wins = results["werewolf_wins"]
    v_wins = results["village_wins"]
    
    ww_win_rate = (ww_wins / total) * 100 if total > 0 else 0
    v_win_rate = (v_wins / total) * 100 if total > 0 else 0
    
    print(f"  Win Rate (Werewolf Side): {ww_win_rate:.2f}% ({ww_wins}/{total})")
    print(f"  Win Rate (Village Side):  {v_win_rate:.2f}% ({v_wins}/{total})")
    print(f"  Prediction Accuracy:      {results['prediction_accuracy']:.2f}%")
    # --- [修正後] (役職ごとの精度を表示) ---
    print("  [LLM Role Prediction Accuracy]")
    stats = results.get("role_prediction_stats", {})
    all_correct = 0
    all_total = 0
    
    for role in ["werewolf", "seer", "doctor", "villager"]:
        if role in stats:
            corr, tot = stats[role]
            acc = (corr / tot * 100) if tot > 0 else 0.0
            print(f"    - Target {role.capitalize()}: {acc:.2f}% ({corr}/{tot})")
            all_correct += corr
            all_total += tot
            
    overall_acc = (all_correct / all_total * 100) if all_total > 0 else 0.0
    print(f"    - Overall:          {overall_acc:.2f}%")

    print(f"  [Strategic Voting Accuracy]")
    print(f"    - Villager Side vs Werewolf: {results['cfr_voting_accuracy']:.2f}%")
