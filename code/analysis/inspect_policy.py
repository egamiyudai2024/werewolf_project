#inspect_policy.py
import torch
import numpy as np
import pickle
import sys
import os
import random
import argparse

# プロジェクトのルートディレクトリをPythonのパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.network import CFRNet
from utils.data_utils import format_obs_to_vector
from lspo.abstracted_environment import AbstractedWerewolfGame
from lspo.api_utils import get_embeddings 

# --- 設定 ---
ITERATION = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ▼▼▼ 追加・修正 (引数処理とディレクトリ決定ロジック) ▼▼▼
# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="Inspect learned policies.")
parser.add_argument('--model_dir', type=str, default=None, help='Path to the model directory.')
args = parser.parse_args()


def get_utterances_by_cluster(kmeans_model, discussion_data_for_role):
    if not discussion_data_for_role: return {}
    all_utterances = [cand for entry in discussion_data_for_role for cand in entry['candidates']]
    all_embeddings = np.array(get_embeddings(all_utterances))
    if all_embeddings.ndim == 1: all_embeddings = all_embeddings.reshape(1, -1)
    if len(all_embeddings) == 0: return {}
    cluster_labels = kmeans_model.predict(all_embeddings)
    utterance_map = {}
    for i in range(kmeans_model.n_clusters):
        utterances_in_cluster = [utt for utt, label in zip(all_utterances, cluster_labels) if label == i]
        utterance_map[i] = utterances_in_cluster
    return utterance_map

# --- 1. モデルとデータのロード ---
print("Loading artifacts...")

# 引数で指定があればそれを使い、なければデフォルト(../../models)を使う
if args.model_dir:
    base_dir = args.model_dir
    print(f"Using specified model directory: {base_dir}")
else:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
    print(f"Using default model directory: {base_dir}")


#base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
kmeans_path = os.path.join(base_dir, f'kmeans_models_iter_{ITERATION}.pkl')
data_path = os.path.join(base_dir, f'discussion_data_iter_{ITERATION}.pkl')
#kmeans_path = f'../kmeans_models_iter_{ITERATION}.pkl'
#data_path = f'../discussion_data_iter_{ITERATION}.pkl'
try:
    with open(kmeans_path, 'rb') as f: kmeans_models = pickle.load(f)
    with open(data_path, 'rb') as f: discussion_data = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: Could not find .pkl file. {e}")
    exit()

print("Mapping utterances to latent strategies...")
strategy_maps = {}
for role in config.ROLES.keys():
    if role in kmeans_models and role in discussion_data:
        strategy_maps[role] = get_utterances_by_cluster(kmeans_models[role], discussion_data[role])

# --- 2. ポリシーの分析ループ ---
print("\n" + "="*50)
print(f"  Inspecting Learned Policies for Iteration {ITERATION+1}")
print("="*50 + "\n")

# 1. 動作確認用のシナリオを定義（特定の局面を作り出す）
SCENARIOS = [
    {
        "name": "Scenario 1: Start of Game (Night 1)",
        "round": 1,
        "phase": "night",
        "alive": [0, 1, 2, 3, 4, 5, 6],
        "last_votes": {},
        "last_dead": None,
        "my_secret": None
    },
    {
        "name": "Scenario 2: Day 2 Discussion (Player 1 killed, suspicious atmosphere)",
        "round": 2,
        "phase": "day_discussion",
        "alive": [0, 2, 3, 4, 5, 6], # Player 1 is dead
        "last_votes": {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1}, # Dummy votes
        "last_dead": 1, 
        "my_secret": 0  # e.g., Seer checked 0
    },
    {
        "name": "Scenario 3: Day 2 Voting (Need to eliminate someone)",
        "round": 2,
        "phase": "day_voting",
        "alive": [0, 2, 3, 4, 5, 6],
        "last_votes": {},
        "last_dead": 1,
        "my_secret": None
    }
]

roles_to_inspect = ['werewolf', 'seer', 'doctor', 'villager']

for role in roles_to_inspect:
    print(f"\n=== Role: {role.upper()} ===")
    
    cfr_net_path = os.path.join(base_dir, f'{role}_net_iter_{ITERATION}.pth')
    
    # --- モデルのロード (既存コードの try-except ブロックと同じ) ---
    try:
        # ゲーム環境の初期化 (次元数計算用)
        game_for_dim = AbstractedWerewolfGame(config.NUM_PLAYERS, config.ROLES, kmeans_models, discussion_data)
        dummy_obs = game_for_dim.get_observation_for_player(0)
        state_dim = len(format_obs_to_vector(dummy_obs)) # 85次元になっているはず
        
        # モデル準備
        discussion_actions = kmeans_models[role].n_clusters if role in kmeans_models else 0
        night_and_voting_actions = config.NUM_PLAYERS
        action_dim = max(discussion_actions, night_and_voting_actions)
        
        model = CFRNet(state_dim, action_dim).to(DEVICE)
        model.load_state_dict(torch.load(cfr_net_path, map_location=DEVICE))
        model.eval()
    except (FileNotFoundError, KeyError) as e:
        print(f"Could not load model for role '{role}'. Skipping. Reason: {e}")
        continue

    # --- シナリオごとの評価ループ (ここがメインの変更点) ---
    for scenario in SCENARIOS:
        print(f"\n  -- {scenario['name']} --")
        
        # 1. ゲーム状態の強制セットアップ
        game = AbstractedWerewolfGame(config.NUM_PLAYERS, config.ROLES, kmeans_models, discussion_data)
        game.reset()
        
        # 【重要】生の変数を強制的に上書き (environment.py の変数名と一致させる)
        # これにより、任意のゲーム途中経過を再現する
        game.round = scenario['round']
        game.phase = scenario['phase']
        game.last_voting_results = scenario['last_votes']
        game.last_announced_dead = scenario['last_dead']
        
        # 生存情報の反映
        for p in game.players:
            p.is_alive = (p.id in scenario['alive'])

        # ターゲットプレイヤーの特定
        player_id = next((p.id for p in game.players if p.role == role), None)
        if player_id is None: continue 
        
        # 自分の秘密行動履歴をセット
        if scenario['my_secret'] is not None:
            game.last_secret_actions[player_id] = scenario['my_secret']

        # 2. アクション空間の取得
        num_actions = game.get_latent_action_space(role)
        if num_actions == 0:
            print("     No available actions in this phase.")
            continue

        # 3. 観測ベクトルの生成と推論
        obs = game.get_observation_for_player(player_id)
        state_vector = format_obs_to_vector(obs)
        state_tensor = torch.from_numpy(state_vector).float().to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            regret_values = model(state_tensor).cpu().numpy().flatten()

        # 4. ポリシー(確率)の計算
        regret_values = regret_values[:num_actions]
        
        # ▼▼▼ [修正] アクションマスクの適用 ▼▼▼
        # policy_optimization.py と同じロジックを適用する
        available_actions = game.get_available_actions(player_id)
        
        masked_regrets = np.full(num_actions, -1e9) # 初期値は負の無限大(選択不可)
        
        if game.phase == 'day_discussion':
            # 議論フェーズは全ての潜在戦略が有効
            masked_regrets = regret_values 
        else:
            # 夜・投票フェーズは有効なターゲットのみ許可
            if available_actions:
                for action in available_actions:
                    if action < num_actions: # 安全策
                        masked_regrets[action] = regret_values[action]
            else:
                pass # 有効アクションなし(パス)

        # マスク適用後の値を使って確率計算
        positive_regrets = np.maximum(masked_regrets, 0)
        regret_sum = np.sum(positive_regrets)
        
        policy = np.zeros(num_actions)
        if regret_sum > 0:
            policy = positive_regrets / regret_sum
        else:
            # Regretが全て負の場合、有効アクション内での一様分布にする
            if game.phase != 'day_discussion' and available_actions:
                valid_count = len([a for a in available_actions if a < num_actions])
                if valid_count > 0:
                    prob = 1.0 / valid_count
                    for action in available_actions:
                        if action < num_actions:
                            policy[action] = prob
            elif game.phase == 'day_discussion':
                 policy = np.ones(num_actions) / num_actions



        #positive_regrets = np.maximum(regret_values, 0)
        #regret_sum = np.sum(positive_regrets)
        #policy = (positive_regrets / regret_sum) if regret_sum > 0 else (np.ones(num_actions) / num_actions)

        # 5. 結果の表示 (上位3つの戦略のみ表示して見やすくする)
        available_actions = game.get_available_actions(player_id)
        
        # 確率の高い順にソート
        top_indices = np.argsort(policy)[::-1][:3]
        
        for i in top_indices:
            prob = policy[i]
            if prob < 0.01: continue # 1%未満は無視
            
            action_meaning = ""
            if game.phase == 'day_discussion':
                action_meaning = f"(Latent Strategy {i})"
            elif i in available_actions:  # 'elif' かつ 'in' を使用
                action_meaning = f"(Target Player {i})"
            else:
                action_meaning = f"(Invalid Target Player {i})"
            
            print(f"     {prob:.1%} : {action_meaning}")
            
            # 議論フェーズなら発言内容のサンプリングを表示
            if game.phase == 'day_discussion' and role in strategy_maps:
                utterances = strategy_maps[role].get(i, [])
                if utterances:
                    sample = random.choice(utterances)
                    # 長すぎるので先頭100文字だけ表示
                    print(f"           -> \"{sample[:100]}...\"")

