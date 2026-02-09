import torch
import os
import gc
import json
import datetime
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from game.environment import WerewolfGame
from agents.lspo_agent import LSPOAgent
from agents.base_agent import BaselineAgent
from evaluation import setup_lspo_components

# ==========================================
# ### MODIFIED ###: 新しいディレクトリの定義
# ==========================================
H2H_GAME_LOG_DIR = "game_logs_h2h"
H2H_PREDICT_LOG_DIR = "debug_predict_h2h"
os.makedirs(H2H_GAME_LOG_DIR, exist_ok=True)
os.makedirs(H2H_PREDICT_LOG_DIR, exist_ok=True)


def load_competition_model(model_spec, cfg, device_id, model_dir=None):
    """
    指定されたGPU IDにモデルをロードする。
    """
    device = f"cuda:{device_id}"
    
    if model_spec['type'] == 'deepseek':
        path = os.path.join(cfg.ROOT_DIR, "pretrained_models", "DeepSeek-R1-Distill-Qwen-32B")
    else:
        path = cfg.BASE_LLM_MODEL 

    print(f"\n[H2H] Loading {model_spec['type']} onto GPU {device_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        path,
        quantization_config=quant_config,
        device_map={"": device_id}, 
        dtype=torch.bfloat16,
        trust_remote_code=True
    )

    if hasattr(model, 'lm_head'):
        for param in model.lm_head.parameters():
            param.data = param.data.to(torch.bfloat16)

    if model_spec['type'] == 'lspo':
        components = setup_lspo_components(model, tokenizer, model_spec['iter'], cfg, device, model_dir)
        return components
    else:
        return {'llm': model, 'tokenizer': tokenizer, 'device': device}

def run_competition(model1_spec, model2_spec, num_games, cfg, model_dir=None):
    """
    モデル対戦のメインエントリ。
    """
    comp1 = load_competition_model(model1_spec, cfg, device_id=0, model_dir=model_dir)
    comp2 = load_competition_model(model2_spec, cfg, device_id=1, model_dir=model_dir)

    # --- Match A: M1 (人狼) vs M2 (村人) ---
    print(f"\n[Match A] {model1_spec['type']} (WW) vs {model2_spec['type']} (Village)")
    res_a = run_mixed_game_loop(comp1, comp2, num_games, cfg, "Match_A_M1WW", m1_spec=model1_spec, m2_spec=model2_spec)
    
    # --- Match B: M2 (人狼) vs M1 (村人) ---
    print(f"\n[Match B] {model2_spec['type']} (WW) vs {model1_spec['type']} (Village)")
    res_b = run_mixed_game_loop(comp2, comp1, num_games, cfg, "Match_B_M2WW", m1_spec=model2_spec, m2_spec=model1_spec)

    print_h2h_results(model1_spec, model2_spec, res_a, res_b)

def run_mixed_game_loop(ww_comp, village_comp, num_games, cfg, desc, m1_spec, m2_spec):
    game = WerewolfGame(num_players=cfg.NUM_PLAYERS, roles_config=cfg.ROLES)
    wins = {"werewolf": 0, "village": 0}
    
    '''
    # クロス予測統計: {predictor_model: {target_model: {correct: 0, total: 0}}}
    cross_prediction_stats = {
        m1_spec['type']: {m2_spec['type']: {'correct': 0, 'total': 0}},
        m2_spec['type']: {m1_spec['type']: {'correct': 0, 'total': 0}}
    }
    '''
    # ### MODIFIED ###: 役職ごとの詳細統計を持つように多層化
    roles_list = ["werewolf", "seer", "doctor", "villager"]
    cross_prediction_stats = {
        m1_spec['type']: {
            m2_spec['type']: {r: {'correct': 0, 'total': 0} for r in roles_list}
        },
        m2_spec['type']: {
            m1_spec['type']: {r: {'correct': 0, 'total': 0} for r in roles_list}
        }
    }


    for g_idx in tqdm(range(num_games), desc=desc):
        game.reset()
        agents = {}
        player_model_map = {} # player_id -> model_type
        
        for p in game.players:
            target_comp = ww_comp if p.team == "werewolf" else village_comp
            m_type = m1_spec['type'] if p.team == "werewolf" else m2_spec['type']
            player_model_map[p.id] = m_type
            
            agent_class = LSPOAgent if 'kmeans' in target_comp else BaselineAgent
            agents[p.id] = agent_class(p.id, p.role, target_comp)
            if hasattr(agents[p.id], 'is_eval'): agents[p.id].is_eval = True

            # ### MODIFIED ###: 予測ログの保存先をH2H専用ディレクトリに強制上書き
            agents[p.id].predict_log_dir = H2H_PREDICT_LOG_DIR

        while not game.is_game_over():
            phase = game.phase
            if phase == "day_discussion":
                speakers = game.get_shuffled_alive_players()
                for _round in range(2):
                    for pid in speakers:
                        if not game.players[pid].is_alive: continue
                        obs = game.get_observation_for_player(pid)
                        action = agents[pid].get_action(obs, phase, [])
                        game.record_discussion_step(pid, action.get("statement", "..."))
                game.phase = "day_voting"
                continue

            # --- 予測フェーズ (投票直前) ---
            if phase == 'day_voting':
                living_pids = game.get_living_players()
                true_roles = game.get_true_roles()
                for pid in living_pids:
                    my_model = player_model_map[pid]
                    opp_model = m2_spec['type'] if my_model == m1_spec['type'] else m1_spec['type']
                    
                    # 予測実行 (game_idxを渡してログ生成)
                    obs = game.get_observation_for_player(pid)
                    predictions = agents[pid].predict_roles(obs, game_idx=g_idx+1)
                    
                    for target_id, pred_role in predictions.items():
                        # フィルタリング: 対象が生存しており、かつ自分と異なるモデルの場合のみ集計
                        if target_id in living_pids and player_model_map[target_id] == opp_model:
                            true_role = true_roles[target_id]
                            cross_prediction_stats[my_model][opp_model][true_role]['total'] += 1
                            if pred_role == true_roles[target_id]:
                                cross_prediction_stats[my_model][opp_model][true_role]['correct'] += 1

            actor_ids = game.get_actors_for_phase()
            if not actor_ids:
                game.step({})
                continue

            actions_to_submit = {}
            for pid in actor_ids:
                obs = game.get_observation_for_player(pid)
                avail = game.get_available_actions(pid)
                action = agents[pid].get_action(obs, phase, avail)
                actions_to_submit[pid] = action
            game.step(actions_to_submit)

        winner = game.get_winner()
        if winner: wins[winner] += 1

        timestamp = datetime.datetime.now().strftime('%H%M%S')
        log_filename = f"log_{desc}_G{g_idx+1}_{timestamp}.json"
        log_path = os.path.join(H2H_GAME_LOG_DIR, log_filename)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "match": desc,
                "game_idx": g_idx + 1,
                "winner": winner,
                "player_models": player_model_map,
                "roles": {p.id: p.role for p in game.players},
                "history": game.game_log
            }, f, indent=2, ensure_ascii=False)
        
        
    return {"wins": wins, "pred_stats": cross_prediction_stats}

def print_h2h_results(m1, m2, res_a, res_b):
    output_lines = []
    output_lines.append("\n" + "="*60)
    output_lines.append(f" FINAL HEAD-TO-HEAD MATRIX: {m1['type']} vs {m2['type']}")
    output_lines.append("="*60)
    
    m1_name = f"{m1['type']}_iter{m1.get('iter',0)}"
    m2_name = f"{m2['type']}_iter{m2.get('iter',0)}"

    output_lines.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"{'Role Configuration':<35} | {'Winner':<15}")

    # ### 厳密な勝敗判定 ###
    def get_winner_str(ww_wins, v_wins, ww_model, v_model):
        if ww_wins > v_wins: return ww_model
        if v_wins > ww_wins: return v_model
        return "Draw"
    
    win_a = m1['type'] if res_a['wins']['werewolf'] > res_a['wins']['village'] else m2['type']
    win_b = m2['type'] if res_b['wins']['werewolf'] > res_b['wins']['village'] else m1['type']
    
    output_lines.append(f"{m1_name}(WW) vs {m2_name}(V): {win_a:<15}")
    output_lines.append(f"{m2_name}(WW) vs {m1_name}(V): {win_b:<15}")

    m1_total_wins = res_a['wins']['werewolf'] + res_b['wins']['village']
    m2_total_wins = res_a['wins']['village'] + res_b['wins']['werewolf']
    
    output_lines.append("-" * 60)
    output_lines.append(f"TOTAL AGGREGATED WINS(Side-Switch Adjusted):")
    output_lines.append(f" - {m1_name}: {m1_total_wins} wins")
    output_lines.append(f" - {m2_name}: {m2_total_wins} wins")

    # ==========================================
    # ### ADDED ###: 結論の出力 (ここに記載)
    # ==========================================
    if m1_total_wins > m2_total_wins:
        conclusion = f"{m1_name} is STRONGER than {m2_name}"
    elif m2_total_wins > m1_total_wins:
        conclusion = f"{m2_name} is STRONGER than {m1_name}"
    else:
        conclusion = "Both models are EQUALLY strong"
    
    output_lines.append(f"\nCONCLUSION: {conclusion}")
    output_lines.append("-" * 60) # 区切り線を追加して見やすく

    full_output = "\n".join(output_lines)
    print(full_output)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"results/h2h_{m1['type']}_vs_{m2['type']}_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_output)
    print(f"\n[System] H2H competition results saved to: {filename}")

'''
    

    # ### MODIFIED ###: クロス予測精度の詳細表示
    output_lines.append("\nCROSS-MODEL PREDICTION ACCURACY (Insight vs Deception):")
    roles_list = ["werewolf", "seer", "doctor", "villager"]
    
    for predictor in [m1['type'], m2['type']]:
        target = m2['type'] if predictor == m1['type'] else m1['type']
        
        # 合計値の計算
        total_correct = 0
        total_attempts = 0
        detail_lines = []
        
        for r in roles_list:
            # Match A と Match B の統計を合算
            c = res_a['pred_stats'][predictor][target][r]['correct'] + res_b['pred_stats'][predictor][target][r]['correct']
            t = res_a['pred_stats'][predictor][target][r]['total'] + res_b['pred_stats'][predictor][target][r]['total']
            total_correct += c
            total_attempts += t
            acc = (c / t * 100) if t > 0 else 0.0
            detail_lines.append(f"    > {r:<10}: {acc:>5.1f}% ({c}/{t})")
        
        overall_acc = (total_correct / total_attempts * 100) if total_attempts > 0 else 0.0
        output_lines.append(f" - {predictor:<10} predicting {target:<10} : {overall_acc:>5.1f}% ({total_correct}/{total_attempts})")
        output_lines.extend(detail_lines)
    
    output_lines.append("="*60)

    full_output = "\n".join(output_lines)
    print(full_output)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"results/h2h_{m1['type']}_vs_{m2['type']}_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_output)
    print(f"\n[System] H2H competition results saved to: {filename}")
'''