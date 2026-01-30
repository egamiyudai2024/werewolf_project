#environment.py
import random
from collections import Counter

class Player:
    # ... (このクラスは変更なし) ...
    def __init__(self, player_id, role):
        self.id = player_id
        self.role = role
        self.is_alive = True
        self.team = "werewolf" if role == "werewolf" else "village"

    def __repr__(self):
        return f"Player_{self.id}({self.role}, {'Alive' if self.is_alive else 'Dead'})"

class WerewolfGame:
    def __init__(self, num_players, roles_config):
        self.num_players = num_players
        self.roles_config = roles_config
        self.intermediate_rewards = {} # ラウンドごとの報酬を記録

        self.last_voting_results = {}      # {voter_id: target_id}
        self.last_announced_dead = None    # player_id or None
        self.last_secret_actions = {}      # {actor_id: target_id} (最新ラウンド用)
        self.reset()

    def reset(self):
        self.players = self._assign_roles()
        self.game_log = []
        self.round = 1
        self.phase = "night"
        self.game_over = False
        self.winner = None
        self.seer_id = next((p.id for p in self.players if p.role == 'seer'), None)
        self.doctor_id = next((p.id for p in self.players if p.role == 'doctor'), None)
        self.intermediate_rewards = {p.id: 0 for p in self.players}

        #行動履歴管理 (重複占い防止 & プロンプト反映用)
        self.seer_checked_players = set()
        # プレイヤーごとの行動履歴をリストで保持
        self.player_action_histories = {p.id: [] for p in self.players}

        #リセット時も初期化
        self.last_voting_results = {}
        self.last_announced_dead = None
        self.last_secret_actions = {}

        return self._get_initial_observations()

    def _get_rewards(self):
        final_rewards = self.intermediate_rewards.copy()
        if self.game_over:
            for player in self.players:
                if player.team == self.winner:
                    final_rewards[player.id] += 300
                else:
                    final_rewards[player.id] -= 300
        return final_rewards
    
    def _update_round_rewards(self):
        for player in self._get_alive_players():
            self.intermediate_rewards[player.id] += 5

    def _process_voting(self, actions):
        votes_by_player = {pid: act.get('vote') for pid, act in actions.items() if act and self.players[pid].is_alive}
        #生の投票データを保存
        self.last_voting_results = votes_by_player
        self.game_log.append({"type": "vote_summary", "round": self.round, "votes": votes_by_player})
        
        # --- 投票報酬の計算 ---
        for voter_id, voted_id in votes_by_player.items():
            if voted_id is None:
                print("プレイヤーによる投票がスキップされました.")
                continue
            voter = self.players[voter_id]
            voted_player = self.players[voted_id]
            if voter.team == 'village':
                if voted_player.role == 'werewolf':
                    self.intermediate_rewards[voter_id] += 20
                else:
                    self.intermediate_rewards[voter_id] -= 20
        
        # ここで除外しないと Counter([None, None]) となり、None が最多得票になってクラッシュする
        valid_votes = [v for v in votes_by_player.values() if v is not None and isinstance(v, int)]

        if not valid_votes:
            # 有効票がゼロの場合（全員棄権、または全員パース失敗）
            self.game_log.append({"type": "elimination", "round": self.round, "text": "No voting occurred (no valid votes)."})
            return
        

        # 【変更点】棄権が選択できないため、votesリストが空になることはない（生存者が2人以上いる限り）
        #votes = list(votes_by_player.values())
        #if not votes:
        #    # 生存者が1人以下など、投票が発生しない稀なケース
        #    self.game_log.append({"type": "elimination", "round": self.round, "text": "No voting occurred."})
        #    return

        vote_counts = Counter(valid_votes)
        max_votes = max(vote_counts.values())
        tied_players = [pid for pid, count in vote_counts.items() if count == max_votes]
        voted_out_player_id = random.choice(tied_players)

        if self.players[voted_out_player_id].is_alive:
            self.players[voted_out_player_id].is_alive = False
            self.game_log.append({"type": "elimination", "round": self.round, "text": f"Player {voted_out_player_id} was voted out."})
            
            # 1. 処刑された本人へのペナルティ (-10)
            self.intermediate_rewards[voted_out_player_id] -= 10
            
            eliminated_player = self.players[voted_out_player_id]
            for player in self._get_alive_players():
                if player.team != eliminated_player.team:
                    self.intermediate_rewards[player.id] += 5
                else:
                    self.intermediate_rewards[player.id] -= 5
    
    def step(self, actions):
        if self.game_over:
            return {p.id: self.get_observation_for_player(p.id) for p in self.players}

        if self.phase == "night":
            self._process_night_actions(actions)
            self.phase = "day_announcement"
        elif self.phase == "day_announcement":
            self.phase = "day_discussion"
        elif self.phase == "day_discussion":
            for pid, action in sorted(actions.items()):
                if action and self.players[pid].is_alive:
                    self.game_log.append({"type": "discussion", "round": self.round, "player_id": pid, "statement": action.get("statement")})
            self.phase = "day_voting"
        elif self.phase == "day_voting":
            self._process_voting(actions)
            self._check_win_condition()
            if not self.game_over:
                self._update_round_rewards()
                self.round += 1
                self.phase = "night"

        return {p.id: self.get_observation_for_player(p.id) for p in self.players}
    
    # -----------------------------------------------------------------
    # ⬇⬇⬇ [追加] 逐次議論のためのヘルパーメソッド ⬇⬇⬇
    
    def get_shuffled_alive_players(self):
        """
        生存プレイヤーのIDリストをランダムにシャッフルして返す。
        これを1日の議論の順番として使用する。
        """
        alive_ids = [p.id for p in self._get_alive_players()]
        random.shuffle(alive_ids)
        return alive_ids

    def record_discussion_step(self, player_id, statement):
        """
        単一のプレイヤーの発言を即座にゲームログに記録する。
        これにより、次のプレイヤーは直前の発言を含んだログを観測できる。
        """
        if self.players[player_id].is_alive:
            self.game_log.append({
                "type": "discussion", 
                "round": self.round, 
                "player_id": player_id, 
                "statement": statement
            })
            
    # ⬆⬆⬆ [追加] ここまで ⬆⬆⬆
    # -----------------------------------------------------------------

    def _assign_roles(self):
        roles = []
        for role, count in self.roles_config.items():
            roles.extend([role] * count)
        random.shuffle(roles)
        return [Player(i, roles[i]) for i in range(self.num_players)]

    def _get_alive_players(self, team=None):
        players = [p for p in self.players if p.is_alive]
        if team:
            return [p for p in players if p.team == team]
        return players

    def _check_win_condition(self):
        if self.game_over:
            return
        alive_werewolves = len(self._get_alive_players("werewolf"))
        alive_villagers = len(self._get_alive_players("village"))
        if alive_werewolves == 0:
            self.game_over = True
            self.winner = "village"
            self.game_log.append({"type": "game_end", "text": "All werewolves are eliminated. Village team wins."})
        elif alive_werewolves >= alive_villagers:
            self.game_over = True
            self.winner = "werewolf"
            self.game_log.append({"type": "game_end", "text": "Werewolves equal or outnumber villagers. Werewolf team wins."})
    
    def get_available_actions(self, player_id):
        player = self.players[player_id]
        if not player.is_alive:
            return []
        
        alive_players_except_self = [p.id for p in self._get_alive_players() if p.id != player_id]
        
        if self.phase == "night":
            # 村人(villager)を含む全役職が、夜に誰かを選ぶ（疑う、占う、守る、襲う）ことができる
            # Doctorのみ自分も選べる、それ以外は自分以外
            if player.role == "doctor":
                return [p.id for p in self._get_alive_players()]
            else:
                return alive_players_except_self if alive_players_except_self else [player_id]
        
        if self.phase == "day_voting":
            return alive_players_except_self
            
        return []

    def _get_public_log(self):
        return [log for log in self.game_log if log.get('type') != 'private']

    def get_observation_for_player(self, player_id):
        player = self.players[player_id]
        obs = {
            "player_id": player.id, "role": player.role, "is_alive": player.is_alive,
            "round": self.round, "phase": self.phase,
            "alive_players": [p.id for p in self._get_alive_players()],
            "game_log": self._get_public_log(),
            "private_log": [log for log in self.game_log if log.get('type') == 'private' and player_id in log.get('visible_to', [])],
	        "action_history": self.player_action_histories[player_id],
            #CFR用の変数を観測データに追加
            "last_voting_results": self.last_voting_results,        # {voter_id: target_id}
            "last_announced_dead": self.last_announced_dead,        # int or None
            "my_last_secret_target": self.last_secret_actions.get(player_id), # int or None
        }
        if player.role == "werewolf":
            teammates = [p.id for p in self.players if p.role == "werewolf" and p.id != player.id]
            obs["teammates"] = teammates
        return obs

    def _get_initial_observations(self):
        return {p.id: self.get_observation_for_player(p.id) for p in self.players}

    def _process_night_actions(self, actions):
        werewolf_targets = [act['target'] for pid, act in actions.items() if act and self.players[pid].is_alive and self.players[pid].role == 'werewolf']
        seer_target = next((act['target'] for pid, act in actions.items() if act and pid == self.seer_id and self.players[pid].is_alive), None)
        doctor_target = next((act['target'] for pid, act in actions.items() if act and pid == self.doctor_id and self.players[pid].is_alive), None)
        
        # ⬇⬇⬇ 【追加】秘密行動の生データを保存 (自分の行動参照用) ⬇⬇⬇
        self.last_secret_actions = {}
        
        # 占い師の行動
        if seer_target is not None:
            self.last_secret_actions[self.seer_id] = seer_target
        
        # 医者の行動
        if doctor_target is not None:
            self.last_secret_actions[self.doctor_id] = doctor_target
            
        # 人狼の行動 (Werewolf全員に、決定されたターゲットを紐付ける)
        # 以前のコードロジックに基づき、most_common(1)で選ばれたターゲットを取得
        kill_target = Counter(werewolf_targets).most_common(1)[0][0] if werewolf_targets else None
        
        if kill_target is not None:
            for p in self.players:
                if p.role == 'werewolf':
                    self.last_secret_actions[p.id] = kill_target
        # ⬆⬆⬆ 追加ここまで ⬆⬆⬆

        # --- Seerの行動記録 ---
        if seer_target is not None and self.players[seer_target].is_alive:
            self.seer_checked_players.add(seer_target) # 重複防止セットに追加
            target_role = self.players[seer_target].role
            is_ww = (target_role == 'werewolf')

            # プロンプト用の構造化履歴に追加
            self.player_action_histories[self.seer_id].append({
                "round": self.round,
                "action_type": "investigate",
                "target": seer_target,
                "result": "Werewolf" if is_ww else "Not Werewolf"
            })


            self.game_log.append({"type": "private","round": self.round, "visible_to": [self.seer_id], "text": f"You investigated Player {seer_target}. They are {'a Werewolf' if is_ww else 'not a Werewolf'}."})

        # --- Doctorの行動記録 ---
        if doctor_target is not None and self.players[self.doctor_id].is_alive:
            # プロンプト用の構造化履歴に追加
            self.player_action_histories[self.doctor_id].append({
                "round": self.round,
                "action_type": "protect",
                "target": doctor_target
            })
            # Doctorは自分が誰を守ったかを知っているべき
            # これにより、game_logで医者が誰を守ったか確認できる
            self.game_log.append({
                "type": "private",
                "round": self.round,
                "visible_to": [self.doctor_id],
                "text": f"You chose to protect Player {doctor_target}."
            })
        
        # --- Werewolfの行動記録 (チーム全員に追加) ---
        kill_target = Counter(werewolf_targets).most_common(1)[0][0] if werewolf_targets else None
        
        if kill_target is not None:
             for p in self.players:
                if p.role == 'werewolf':
                    self.player_action_histories[p.id].append({
                        "round": self.round,
                        "action_type": "attack",
                        "target": kill_target
                    })


        # エージェント側では「疑い」としてターゲットを選ばせているため、
        # ここで kill_target を None にすることで、誰も死なないようにする。
        #if self.round == 1:
        #    kill_target = None
        
        if kill_target is not None and self.players[kill_target].is_alive and kill_target != doctor_target:
            self.players[kill_target].is_alive = False
            self.game_log.append({"type": "announcement", "round": self.round, "text": f"Player {kill_target} was killed last night."})
            #死亡者を保存
            self.last_announced_dead = kill_target
        else:
            self.game_log.append({"type": "announcement", "round": self.round, "text": "No one was killed last night."})
            #死亡者なし
            self.last_announced_dead = None

        
        self._check_win_condition()
    

# -----------------------------------------------------------------
# ⬇⬇⬇ environment.py に以下のメソッドを追加 ⬇⬇⬇
    def is_game_over(self):
        """
        評価ループがゲームの終了を検知するために使用します。
        """
        return self.game_over

    def get_winner(self):
        """
        評価ループが勝率を計算するために使用します。
        """
        return self.winner

    def get_living_players(self):
        """
        評価ループが予測精度の計算対象を絞り込むために使用します。
        """
        return [p.id for p in self._get_alive_players()]

    def get_true_roles(self):
        """
        評価ループが予測精度を採点するために使用します。
        """
        return {p.id: p.role for p in self.players}

    def get_actors_for_phase(self):
        """
        評価ループが、現在のフェーズで行動が必要なプレイヤーを
        特定するために使用します。
        """
        # アナウンスフェーズは自動進行
        if self.phase == "day_announcement":
            return []
        
        # それ以外のフェーズ（night, day_discussion, day_voting）では、
        # 生存プレイヤー全員が行動決定（または何もしない決定）を行う必要がある
        return [p.id for p in self._get_alive_players()]



 
