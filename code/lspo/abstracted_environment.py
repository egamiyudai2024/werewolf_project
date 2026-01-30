# lspo/abstracted_environment.py
import numpy as np
from sklearn.cluster import KMeans
from game.environment import WerewolfGame 
from .api_utils import get_embeddings
import copy 
import random 
from dataclasses import dataclass
from typing import List, Dict, Set, Any


#動的状態のみを保持する軽量クラス定義
@dataclass
class WerewolfGameState:
    """
    ゲームの動的な状態のみを保持する軽量スナップショット。
    Deep CFRの探索中にコピーコストを削減するために使用する。
    """
    round: int
    phase: str
    game_over: bool
    winner: Any
    players_alive: Dict[int, bool]          # プレイヤーID -> 生存フラグ
    game_log: List[Dict]                    # ログデータ（テキスト）
    intermediate_rewards: Dict[int, float]  # 報酬マップ
    seer_checked_players: Set[int]          # 占い済みリスト
    player_action_histories: Dict[int, List[Dict]] # 行動履歴


class AbstractedWerewolfGame(WerewolfGame):
    """
    CFRの学習のために、議論アクションを離散的な潜在戦略IDに置き換えたゲーム環境.
    """
    def __init__(self, num_players, roles_config, kmeans_models, discussion_data):
        print("DEBUG: Initializing AbstractedWerewolfGame (This should happen only ONCE per iter!)") # 追加
        super().__init__(num_players, roles_config)
        self.kmeans_models = kmeans_models
        # discussion_dataの構造が変更されたため、新しいマッピング方法に変更
        self._build_latent_to_text_map_from_candidates(discussion_data)

    def __deepcopy__(self, memo):
        """
        copy.deepcopyがこのオブジェクトをコピーする際の挙動をカスタマイズする.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ['kmeans_models', 'latent_map']:
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def _build_latent_to_text_map_from_candidates(self, discussion_data):
        """ 新しいdiscussion_data構造からマッピングを構築 """
        self.latent_map = {}
        for role, data_list in discussion_data.items():
            if not data_list or self.kmeans_models.get(role) is None:
                continue
            
            self.latent_map[role] = {}
            kmeans = self.kmeans_models[role]
            
            # 全ての候補発言をリストに集める
            all_utterances = [candidate for entry in data_list for candidate in entry['candidates']]
            
            if not all_utterances or len(all_utterances) < kmeans.n_clusters:
                continue

            embeddings = get_embeddings(all_utterances)
            if embeddings is None or len(embeddings) == 0:
                continue

            labels = kmeans.predict(embeddings)

            # 各クラスタに属する発言をマッピング
            for i in range(kmeans.n_clusters):
                cluster_utterances = [utt for utt, label in zip(all_utterances, labels) if label == i]
                self.latent_map[role][i] = cluster_utterances
    
    # （追加）状態の保存(get)と復元(set)メソッド
    
    def get_state(self) -> WerewolfGameState:
        """
        現在のゲーム状態の軽量スナップショットを作成して返す。
        巨大な静的データ（KMeansモデルやEmbeddingマップ）は含まない。
        """
        return WerewolfGameState(
            round=self.round,
            phase=self.phase,
            game_over=self.game_over,
            winner=self.winner,
            # Playerオブジェクト自体ではなく、変動する生存フラグのみをIDでマッピングして保存
            players_alive={p.id: p.is_alive for p in self.players},
            
            # リスト/辞書/セットはミュータブルなのでディープコピーして保存する
            # (テキストデータ等の軽量なコピーなので高速)
            game_log=copy.deepcopy(self.game_log),
            intermediate_rewards=copy.deepcopy(self.intermediate_rewards),
            seer_checked_players=copy.deepcopy(self.seer_checked_players),
            player_action_histories=copy.deepcopy(self.player_action_histories)
        )

    def set_state(self, state: WerewolfGameState):
        """
        スナップショットからゲーム状態を復元する。
        """
        self.round = state.round
        self.phase = state.phase
        self.game_over = state.game_over
        self.winner = state.winner
        
        # プレイヤーの生存状態を復元 (Playerオブジェクト自体は再生成せずステータスのみ更新)
        for p in self.players:
            if p.id in state.players_alive:
                p.is_alive = state.players_alive[p.id]
        
        # コンテナデータの復元
        # (ここでもdeepcopyすることで、復元後の進行がスナップショットに影響を与えないようにする)
        self.game_log = copy.deepcopy(state.game_log)
        self.intermediate_rewards = copy.deepcopy(state.intermediate_rewards)
        self.seer_checked_players = copy.deepcopy(state.seer_checked_players)
        self.player_action_histories = copy.deepcopy(state.player_action_histories)

    def transition(self, state: WerewolfGameState, latent_actions: Dict[int, int]) -> WerewolfGameState:
        """
        純粋関数的な状態遷移メソッド（Engine機能）。
        
        引数で渡された `state` に対して `latent_actions` を適用した場合の
        「新しい状態（スナップショット）」を計算して返します。
        
        このメソッドは実行前後でインスタンスの内部状態を変更しません（副作用なし）。
        Deep CFRの探索で「仮定の未来」をシミュレーションするために使用します。
        """
        # 1. 現在の内部状態をバックアップ（副作用を防ぐため）
        #    ※最適化の余地はありますが、安全性を優先して Memento パターンを適用します
        backup_state = self.get_state()
        
        try:
            # 2. 指定された「過去/現在の状態」をセット
            self.set_state(state)
            
            # 3. アクションを実行してゲームを1ステップ進める
            #    (内部的には self.players や self.game_log が更新される)
            self.step_with_latent_actions(latent_actions)
            
            # 4. 更新された状態をスナップショットとして取得（これが戻り値）
            next_state = self.get_state()
            
            return next_state
            
        finally:
            # 5. 元の状態に復元（呼び出し元のコンテキストを壊さない）
            self.set_state(backup_state)


    def get_latent_action_space(self, role):
        """
        現在のフェーズと役職に基づいて、取りうる潜在アクションの数を返す.
        """
        # 議論フェーズでは、クラスタ数をアクション数とする
        if self.phase == "day_discussion":
            if role in self.kmeans_models and self.kmeans_models[role] is not None:
                return self.kmeans_models[role].n_clusters
            return 0 # モデルがない場合はアクションなし

        # 【修正】夜・投票フェーズは「プレイヤー人数（固定）」を返す
        # これにより、Action ID = Player ID という絶対的な対応関係を作る
        # (available_actionsの長さではなく、全プレイヤーIDの範囲をカバーする)
        return self.num_players

    def step_with_latent_actions(self, latent_actions):
        """潜在アクション（整数）を実際のゲームアクションに変換してステップを進める"""
        full_actions = {}
        for player_id, latent_id in latent_actions.items():
            player = self.players[player_id]
            if not player.is_alive:
                continue

            if self.phase == "day_discussion":
                # クラスタIDに対応する発言リストを取得し、その中からランダムに1つ選ぶ
                utterance_list = self.latent_map.get(player.role, {}).get(latent_id, ["Default statement."])
                action_text = random.choice(utterance_list) if utterance_list else "Default statement."
                full_actions[player_id] = {"statement": action_text}
            
            else:
                 # 【修正】夜または投票フェーズ
                 # latent_id (CFRが出力したAction ID) をそのままターゲットのPlayer IDとして扱う
                 target_id = latent_id
                 available_actions = self.get_available_actions(player_id)
                 
                 # CFRが探索で選んだ target_id が、現在の有効なアクションに含まれているか確認
                 # (例: 死んだプレイヤーを選んでいないかチェック)
                 if target_id in available_actions:
                     chosen_action = target_id
                 else:
                     # 無効なアクション（死んだ人への投票など）を選んだ場合のフォールバック
                     # 学習を止めないため、ランダムに有効なアクションを選ぶ
                     if available_actions:
                         chosen_action = random.choice(available_actions)
                     else:
                         chosen_action = None

                 if chosen_action is not None:
                     if self.phase == "night":
                         full_actions[player_id] = {"target": chosen_action}
                     elif self.phase == "day_voting":
                         full_actions[player_id] = {"vote": chosen_action}
                         
        return self.step(full_actions)
