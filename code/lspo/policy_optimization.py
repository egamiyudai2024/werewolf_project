#policy_optimization.py
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils.network import CFRNet #utills/network.pyで定義された後悔値を予測するためのニューラルネットワークモデル
from utils.data_utils import format_obs_to_vector
from .abstracted_environment import AbstractedWerewolfGame #発話アクションが「潜在戦略ID」に置き換えられた、cfr学習用の特別なゲーム環境
import random
import copy

class DeepCFRTrainer:
    def __init__(self, abstracted_game_env, role_models, device, config, max_action_dims): #抽象化されたゲーム環境/各役職に対応するCFRNetモデル/CPUおよびGPUの設定/各種設定情報/各役職の最大アクション数
        self.game_env = abstracted_game_env
        self.role_models = role_models
        self.optimizers = {role: optim.Adam(model.parameters(), lr=config.CFR_LEARNING_RATE) for role, model in role_models.items()}
        self.memory = {role: [] for role in role_models}
        self.device = device
        self.config = config
        self.loss_fn = nn.MSELoss()
        self.max_action_dims = max_action_dims # 各役割の最大アクション数を保持

    def update_networks(self):
        for role, model in self.role_models.items():
            if len(self.memory[role]) < self.config.CFR_BATCH_SIZE: continue
            batch = random.sample(self.memory[role], self.config.CFR_BATCH_SIZE)
            state_batch, regret_batch = zip(*batch)
            state_tensor = torch.tensor(np.array(state_batch), dtype=torch.float32).to(self.device)
            regret_tensor = torch.tensor(np.array(regret_batch), dtype=torch.float32).to(self.device)
            self.optimizers[role].zero_grad()
            predicted_regrets = model(state_tensor)
            
            # 両方のテンソルの次元が揃っているので、シンプルな損失計算に戻す
            loss = self.loss_fn(predicted_regrets, regret_tensor)
            loss.backward()
            self.optimizers[role].step()

    def _rollout(self, game_state):
        """
        深さ制限に達した際、現在のポリシー（ニューラルネットワーク）に従って
        ゲーム終了までシミュレーションを行い、報酬を返す。 
        """
        # ロールアウト中は探索（分岐）しないため、状態を直接進めて良い
        # (呼び出し元で set_state して復元されるため破壊してOK)
        
        while not game_state.is_game_over():
            actions = {}
            acting_players = game_state.get_actors_for_phase()
            
            for pid in acting_players:
                player = game_state.players[pid]
                role = player.role
                
                # --- ポリシー（戦略）の計算 ---
                # 現在のネットワークを使って「どの行動をとるべきか」の確率分布を出す
                model = self.role_models.get(role)
                num_actions = game_state.get_latent_action_space(role)
                
                chosen_action = 0
                if model and num_actions > 0:
                    # 1. 観測の取得
                    obs = game_state.get_observation_for_player(pid)
                    sv = format_obs_to_vector(obs)
                    st = torch.from_numpy(sv).float().to(self.device).unsqueeze(0)
                    
                    # 2. 推論 (Regret予測)
                    with torch.no_grad():
                        regrets = model(st).cpu().numpy().flatten()[:num_actions]
                    
                    # 3. Regret Matching で確率分布に変換
                    # (後悔が大きい行動ほど選ばれやすくする)
                    positive_regrets = np.maximum(regrets, 0)
                    sum_R = np.sum(positive_regrets)
                    
                    if sum_R > 0:
                        probs = positive_regrets / sum_R
                    else:
                        probs = np.ones(num_actions) / num_actions # 一様ランダム
                    
                    # 4. 確率に従ってサンプリング
                    chosen_action = np.random.choice(num_actions, p=probs)
                
                else:
                    # モデルがない場合やアクションがない場合はランダム (フォールバック)
                    if num_actions > 0:
                        chosen_action = random.randrange(num_actions)
                
                # --- アクションの登録 ---
                if game_state.phase == "day_discussion":
                    actions[pid] = chosen_action
                else:
                    # 夜・投票フェーズ: 潜在ID = ターゲットID
                    # ただし有効なターゲットかチェックが必要
                    avail = game_state.get_available_actions(pid)
                    # モデルが選んだIDが有効範囲外ならランダムな有効アクションに置換
                    if avail and chosen_action in avail:
                        actions[pid] = chosen_action
                    elif avail:
                        actions[pid] = random.choice(avail)
                    else:
                        actions[pid] = None # パス
            
            # ゲーム進行
            game_state.step_with_latent_actions(actions)

        # 終了時の報酬を返す
        rewards = game_state._get_rewards()
        return np.array([rewards.get(p.id, 0) for p in game_state.players])

    def traverse_game_tree(self, game_state, reach_probabilities, depth=0):
        
        # 1. ゲーム終了判定
        if game_state.game_over:
            final_rewards = game_state._get_rewards()
            return np.array([final_rewards.get(p.id, 0) for p in game_state.players])
        
        # 2. 深さ制限チェック (Rolloutへ移行)
        if depth >= self.config.CFR_MAX_DEPTH:
            return np.zeros(self.config.NUM_PLAYERS)
        
        # 3. 行動プレイヤー特定

        current_phase = game_state.phase
        if current_phase == "day_discussion":
            acting_player_ids = [p.id for p in game_state._get_alive_players()]
        else:
            acting_player_ids = [p.id for p in game_state._get_alive_players() if game_state.get_available_actions(p.id)]

        if not acting_player_ids:
            snapshot = game_state.get_state()
            game_state.step({})
            util = self.traverse_game_tree(game_state, reach_probabilities, depth)
            game_state.set_state(snapshot)
            return util
        
        #if not acting_player_ids:
        #    game_state.step({})
        #    return self.traverse_game_tree(game_state, reach_probabilities, depth + 1)

        cfr_player_idx = acting_player_ids[0]
        cfr_player = game_state.players[cfr_player_idx]
        role = cfr_player.role
        
        if not self.role_models.get(role):
             snapshot = game_state.get_state()
             game_state.step({})
             util = self.traverse_game_tree(game_state, reach_probabilities, depth)
             game_state.set_state(snapshot)
             return util
        #     game_state.step({})
        #     return self.traverse_game_tree(game_state, reach_probabilities, depth + 1)

        # --- Regret Matching (現在の方策計算) ---
        current_observation = game_state.get_observation_for_player(cfr_player_idx)
        state_vector = format_obs_to_vector(current_observation)
        num_actions = game_state.get_latent_action_space(role)
        
        if num_actions == 0:
            snapshot = game_state.get_state()
            game_state.step({})
            util = self.traverse_game_tree(game_state, reach_probabilities, depth)
            game_state.set_state(snapshot)
            return util
        #    game_state.step({})
        #    return self.traverse_game_tree(game_state, reach_probabilities, depth + 1)

        state_tensor = torch.from_numpy(state_vector).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            regret_values = self.role_models[role](state_tensor).cpu().numpy().flatten()
        
        regret_values = regret_values[:num_actions]
        
        positive_regrets = np.maximum(regret_values, 0)
        regret_sum = np.sum(positive_regrets)
        current_policy = (positive_regrets / regret_sum) if regret_sum > 0 else (np.ones(num_actions) / num_actions)
        
        child_node_utilities = np.zeros((num_actions, self.config.NUM_PLAYERS))
        node_utility = np.zeros(self.config.NUM_PLAYERS)

        # 状態のバックアップ (軽量スナップショット)
        snapshot = game_state.get_state()

        for action in range(num_actions):
            # 自分(CFRプレイヤー)のアクション
            actions_for_phase = {cfr_player_idx: action}
            
            # 他プレイヤーのアクション決定 (サンプリング)
            # ※ここでこそ「現在の戦略」を使ってサンプリングする (External Sampling的アプローチ)
            #other_acting_players = [pid for pid in acting_player_ids if pid != cfr_player_idx]hypothetical_game = copy.deepcopy(game_state)
            other_acting_players = [pid for pid in acting_player_ids if pid != cfr_player_idx]
            
            for other_pid in other_acting_players:
                other_role = game_state.players[other_pid].role
                #other_role = hypothetical_game.players[other_pid].role
                if self.role_models.get(other_role):
                    other_num_actions = game_state.get_latent_action_space(other_role)
                    #other_num_actions = hypothetical_game.get_latent_action_space(other_role)
                    if other_num_actions > 0:
                        other_obs = game_state.get_observation_for_player(other_pid)
                        #other_obs = hypothetical_game.get_observation_for_player(other_pid)
                        other_sv = format_obs_to_vector(other_obs)
                        other_st = torch.from_numpy(other_sv).float().to(self.device).unsqueeze(0)
                        with torch.no_grad():
                            other_rv = self.role_models[other_role](other_st).cpu().numpy().flatten()
                        other_rv = other_rv[:other_num_actions]
                        other_pr = np.maximum(other_rv, 0)
                        other_rs = np.sum(other_pr)
                        other_policy = (other_pr / other_rs) if other_rs > 0 else (np.ones(other_num_actions) / other_num_actions)
                        other_action = np.random.choice(len(other_policy), p=other_policy)
                        actions_for_phase[other_pid] = other_action
                    else:
                        actions_for_phase[other_pid] = 0
                else:
                    other_num_actions = game_state.get_latent_action_space(other_role)
                    #other_num_actions = hypothetical_game.get_latent_action_space(other_role)
                    if other_num_actions > 0:
                        actions_for_phase[other_pid] = random.randrange(other_num_actions)

            #hypothetical_game.step_with_latent_actions(actions_for_phase)
            # 1. ゲーム進行 (In-place) - game_state を直接進める
            game_state.step_with_latent_actions(actions_for_phase)
            
            # 2. 再帰呼び出し (参照渡し)
            child_node_utilities[action] = self.traverse_game_tree(game_state, reach_probabilities, depth + 1)
            
            # 3. 状態復元 (Undo) - snapshotを使って元に戻す
            game_state.set_state(snapshot)
            
            # 期待値計算
            node_utility += current_policy[action] * child_node_utilities[action]

        reach_prob_opponent = np.prod(np.delete(reach_probabilities, cfr_player_idx))
        counterfactual_regrets = np.zeros(num_actions)
        for action in range(num_actions):
            regret = (child_node_utilities[action][cfr_player_idx] - node_utility[cfr_player_idx]) * reach_prob_opponent
            counterfactual_regrets[action] = regret
        
        # 【修正点3】後悔量をパディングしてからメモリに保存する
        max_dim = self.max_action_dims[role]
        padded_regrets = np.zeros(max_dim)
        padded_regrets[:num_actions] = counterfactual_regrets
        
        if len(self.memory[role]) >= self.config.CFR_BUFFER_SIZE: self.memory[role].pop(0)
        self.memory[role].append((state_vector, padded_regrets))
        return node_utility

    def train_cfr(self, num_iterations):
        # ... (このメソッドは変更なし) ...
        print("Starting Deep CFR training...")
        for _ in tqdm(range(num_iterations), desc="Deep CFR Training"):
            game_state = copy.deepcopy(self.game_env)
            game_state.reset()
            initial_reach_probs = np.ones(self.config.NUM_PLAYERS)
            self.traverse_game_tree(game_state=game_state, reach_probabilities=initial_reach_probs, depth=0)
            self.update_networks()
        print("Deep CFR training finished.")


def run_policy_optimization(config, kmeans_models, discussion_data, device):
    print("Initializing components for policy optimization...")
    abstracted_game = AbstractedWerewolfGame(config.NUM_PLAYERS, config.ROLES, kmeans_models, discussion_data)
    
    dummy_obs = abstracted_game.get_observation_for_player(0)
    state_dim = len(format_obs_to_vector(dummy_obs))
    print(f"State vector dimension set to: {state_dim}")
    
    role_models = {}
    max_action_dims = {} 
    
    for role, role_count in config.ROLES.items():
        if role_count > 0:
            abstracted_game.phase = "day_discussion"
            discussion_actions = abstracted_game.get_latent_action_space(role)
            
            # 最大生存者数(num_players)を仮定して夜と投票のアクション数を計算
            # 実際のget_available_actionsは自分以外の生存者数を返すため、-1する→doctorが自分自身を蘇生可能なため、-1を削除
            night_and_voting_actions = config.NUM_PLAYERS

            action_dim = max(discussion_actions, night_and_voting_actions)
            max_action_dims[role] = action_dim # 辞書に保存
            print(f"Role: {role}, Max Action Dimension: {action_dim}")

            if action_dim > 0:
                role_models[role] = CFRNet(state_dim, action_dim).to(device)

    if not role_models:
        print("Warning: No models to train for CFR. Skipping policy optimization.")
        return {}
    
    # DeepCFRTrainerにmax_action_dimsを渡す
    cfr_trainer = DeepCFRTrainer(abstracted_game, role_models, device, config, max_action_dims)
    cfr_trainer.train_cfr(num_iterations=config.CFR_TRAIN_ITERATIONS)
    print("Policy optimization step finished.")
    return role_models
