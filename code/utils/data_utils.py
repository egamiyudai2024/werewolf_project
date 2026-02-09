# utils/data_utils.py
import numpy as np
import torch
from lspo.listener import Listener # 追加
import config # 追加


# --- 定数定義 (論文 Appendix C.1 より) ---
SYSTEM_PROMPT = """You are an expert in playing the social deduction game named Werewolf. The game has seven roles
including two Werewolves, one Seer, one Doctor, and three Villagers. There are seven players including
player_0, player_1, player_2, player_3, player_4, player_5, and player_6.
At the beginning of the game, each player is assigned a hidden role which divides them into
the Werewolves and the Villagers (Seer, Doctor, Villagers). Then the game alternates between the night
round and the day round until one side wins the game.
In the night round: the Werewolves choose one player to kill; the Seer chooses one player to
see if they are a Werewolf; the Doctor chooses one player including themselves to save without knowing
who is chosen by the Werewolves; the Villagers do nothing.
In the day round: three phases including an announcement phase, a discussion phase, and a
voting phase are performed in order.
In the announcement phase, an announcement of last night's result is made to all players. If player_i was attacked by the Werewolves and not saved by the Doctor last night, the announcement will be "player_i was killed"; if a player was attacked by the Werewolves but saved by the Doctor last night, the announcement will be "no player was killed".
In the discussion phase, the speaking order is randomized. The discussion consists of 2 rounds, so you will have the opportunity to speak twice to discuss who might be the Werewolves.
In the voting phase, each player must votes for one player. The player with the most
votes is eliminated and the game continues to the next night round.
"IMPORTANT: The Werewolves win the game if the number of remaining Werewolves is equal to the number of
remaining Seer, Doctor, and Villagers. The Seer, Doctor, and Villagers win the game if all Werewolves
are eliminated."""

# LSPOの探索空間を拡張するための3つの戦略指針
STRATEGY_DEFINITIONS = {
    0: "STRATEGY: Be AGGRESSIVE. Actively suspect others, lead the discussion, and accuse someone strongly based on the logs.",
    1: "STRATEGY: Be DEFENSIVE and COOPERATIVE. Focus on establishing your own credibility and building trust with others.\n"
        "Avoid making unnecessary enemies. Support logical opinions from others and prioritize the stability of the village discussion.",
    2: "STRATEGY: Be STRATEGIC and TECHNICAL. Use game mechanics like Role Claim (CO) or bluffing. Look for logical contradictions. Play deceptively if you are a Werewolf."
}


def one_hot(index, size):
    vec = np.zeros(size)
    if index is not None and 0 <= index < size:
        vec[index] = 1
    return vec

# Listenerのインスタンス化
listener_instance = Listener(embedding_dim=config.EMBEDDING_DIM)

def format_obs_to_vector(observation):
    NUM_PLAYERS = 7
    NUM_ROLES = 4
    NUM_PHASES = 3
    ROLES_MAP = {"werewolf": 0, "seer": 1, "doctor": 2, "villager": 3}
    PHASES_MAP = {"night": 0, "day_announcement": 0, "day_discussion": 1, "day_voting": 2}

    player_id_vec = one_hot(observation.get('player_id'), NUM_PLAYERS)
    role_vec = one_hot(ROLES_MAP.get(observation.get('role')), NUM_ROLES)
    curr_round_val = observation.get('round', 1)  #追加分
    current_round = np.array([curr_round_val - 1])
    current_phase_vec = one_hot(PHASES_MAP.get(observation.get('phase')), NUM_PHASES)
    alive_players_vec = np.zeros(NUM_PLAYERS)
    for p_id in observation.get('alive_players', []):
        alive_players_vec[p_id] = 1

    MAX_ROUNDS = config.MAX_ROUNDS  # 例: 3

    game_log = observation.get('game_log', [])
    
    # データを解析しやすい形に整理
    # 1. 自分の過去のアクション (action_historyから抽出)
    my_actions_map = {}
    for entry in observation.get('action_history', []):
        r = entry.get('round')
        tgt = entry.get('target')
        if r is not None:
            my_actions_map[r] = tgt

    # 2. 過去の投票とアナウンス (game_logから抽出)
    votes_map = {}
    dead_map = {}
    
    for log in game_log:
        r = log.get('round')
        ltype = log.get('type')
        
        if ltype == 'vote_summary':
            # {'voter_id': target_id}
            votes_map[r] = log.get('votes', {})
            
        elif ltype == 'announcement':
            # "Player X was killed" / "No one..."
            text = log.get('text', "")
            if "was killed" in text and "Player" in text:
                # 簡易パース: "Player X" の数字を取り出す
                import re
                match = re.search(r'Player\s*(\d+)', text)
                if match:
                    dead_map[r] = int(match.group(1))
            elif "No one" in text:
                dead_map[r] = -1 # 特殊値: 誰も死ななかった

    # 履歴スロットの構築 (Round 1 〜 MAX_ROUNDS)
    history_vecs = []
    
    for r in range(1, MAX_ROUNDS + 1):
        # A. Secret Action (7次元)
        # 過去のラウンド、または現在ラウンドで既に夜のアクションが終わっている場合
        vec_sec = np.zeros(NUM_PLAYERS)
        # 現在ラウンドより前、または現在ラウンドかつ夜フェーズではない(＝行動済み)と仮定できる場合
        # あるいは単純に action_history に存在すれば埋める方針で統一
        if r in my_actions_map:
            tgt = my_actions_map[r]
            if tgt is not None and isinstance(tgt, int) and 0 <= tgt < NUM_PLAYERS:
                vec_sec[tgt] = 1

        # B. Announcement (7次元) - 昨夜の死者
        vec_dead = np.zeros(NUM_PLAYERS)
        if r in dead_map:
            tgt = dead_map[r]
            if tgt >= 0 and tgt < NUM_PLAYERS:
                vec_dead[tgt] = 1
        
        # C. Voting Result (49次元)
        vec_vote = np.zeros(NUM_PLAYERS * NUM_PLAYERS)
        if r in votes_map:
            votes = votes_map[r]
            for v_str, t_val in votes.items():
                try:
                    if t_val is not None:
                        v_id = int(v_str)
                        t_id = int(t_val)
                        if 0 <= v_id < NUM_PLAYERS and 0 <= t_id < NUM_PLAYERS:
                            idx = v_id * NUM_PLAYERS + t_id
                            vec_vote[idx] = 1
                except:
                    pass
                    
        history_vecs.extend([vec_sec, vec_dead, vec_vote])

    history_flat = np.concatenate(history_vecs)

    # --- 会話情報 (Listener) ---
    
    # 3. 当日の議論 (environment.pyから渡された生ログを使用)
    # config.py の DISCUSSION_TURNS, EMBEDDING_DIM を使用
    #curr_logs = observation.get('current_discussion_logs', [])
    curr_logs = [
        log for log in game_log 
        if log.get('round') == curr_round_val and log.get('type') == 'discussion'
    ]

    discussion_vec = listener_instance.process_current_discussion(
        curr_logs, 
        num_players=NUM_PLAYERS, 
        max_turns=config.DISCUSSION_TURNS
    )

    # 4. 過去の議論ベクトル化
    # 過去の全ラウンドについて完全スロットを作成 (MAX_ROUNDS-1) * 896 次元
    past_discussion_vec = listener_instance.summarize_past_discussion(
        game_log, 
        curr_round_val
    )

    # 全て結合
    final_vector = np.concatenate([
        player_id_vec,
        role_vec,
        current_round,
        current_phase_vec,
        alive_players_vec,
        history_flat,     # 63×5=315
        discussion_vec,   # 896
        past_discussion_vec  
    ])

    
    
    # 次元数チェックとログ出力
    actual_dim = len(final_vector)
    expected_dim = config.STATE_DIM
    
    if actual_dim != expected_dim:
        raise ValueError(f"[FATAL ERROR] Vector dimension mismatch! Actual: {actual_dim}, Expected from Config: {expected_dim}")
    
    # デバッグ用: 初回または稀なタイミングで確認ログを出す（ここでは毎回出すと遅くなるため、エラー時以外はコメントアウト推奨だが、確認のため入れる）
    # print(f"[DEBUG] Vector dim: {actual_dim} (OK)")
    
    return final_vector.astype(np.float32)


    
def format_obs_to_prompt(observation):
    """
    情報を [Secret Private Memory], [Public Game Log], [Current Situation] に構造化して提示します。
    """
    p_id = observation.get('player_id')
    role = observation.get('role')
    current_round = observation.get('round')
    phase = observation.get('phase')
    alive_players = observation.get('alive_players', [])
    
    # プロンプト全体の初期化
    prompt = f"{SYSTEM_PROMPT}\n\n"

    # =================================================================
    # 1. [Secret Private Memory] (最重要・他言無用)
    # =================================================================
    prompt += "=== [Secret Private Memory] (ONLY YOU KNOW THIS) ===\n"
    prompt += f"You are player_{p_id}.\n"
    prompt += f"Your Role: {role.capitalize()} (Keep this secret!)\n"

    # 人狼のチームメイト情報
    if role == 'werewolf' and 'teammates' in observation:
        teammates = observation['teammates']
        if teammates:
            teammates_str = ", ".join([f"player_{t}" for t in teammates])
            prompt += f"Your Teammate: {teammates_str} (Do not betray them unless necessary).\n"

    # 行動履歴 (自然言語化)
    action_history = observation.get('action_history', [])
    if action_history:
        prompt += "Your Past Actions:\n"
        for entry in action_history:
            r = entry['round']
            act_type = entry['action_type']
            target = entry['target']
            
            if act_type == "investigate":
                result = entry['result']
                # 例: Night 1 Action: You investigated player_2 and found they are a Werewolf.
                prompt += f"- Night {r} Action: You investigated player_{target} and found they are {result}.\n"
            elif act_type == "protect":
                prompt += f"- Night {r} Action: You chose to protect player_{target}.\n"
            elif act_type == "attack":
                prompt += f"- Night {r} Action: You and your teammate chose to attack player_{target}.\n"
    else:
        prompt += "(No actions taken yet.)\n"

    prompt += "\n"

    # =================================================================
    # 2. [Public Game Log] & 3. [Current Situation] の構築
    # =================================================================
    # ログを過去（Public）と現在（Current）に振り分ける
    game_log = observation.get('game_log', [])
    private_log = observation.get('private_log', [])

    # 0: Announcement
    # 1: Discussion
    # 2: Vote Summary (Details)
    # 3: Elimination (Result)
    type_order = {
        "announcement": 0,
        "discussion": 1,
        "vote_summary": 2,
        "elimination": 3
    }

    # type_orderにないものは -1 (先頭) として扱う
    all_logs = sorted(
        game_log + private_log, 
        key=lambda x: (x.get('round', 0), type_order.get(x.get('type'), -1))
    )

    # ログをソート
    #all_logs = sorted(game_log + private_log, key=lambda x: (x.get('round', 0), 0 if x.get('type')!='discussion' else 1))

    public_log_str = ""
    current_log_str = ""
    
    current_processing_round = 0

    for entry in all_logs:
        r = entry.get('round')
        type_ = entry.get('type')
        text = entry.get('text')
        
        # ログ行の生成
        line = ""
        if type_ == "announcement":
            line = f"[Day {r} Announcement]: {text}\n"
        elif type_ == "discussion":
            pid = entry.get('player_id')
            stmt = entry.get('statement')
            line = f"player_{pid} said: \"{stmt}\"\n"
        elif type_ == "elimination":
            line = f"[Day {r} Voting Result]: {text}\n"
        elif type_ == "vote_summary":
            votes = entry.get('votes', {})
            vote_details = []
            # 投票データを文字列化 (例: player_0 voted for player_1)
            # 見やすさのためプレイヤーID順にソート
            for v_id in sorted(votes.keys(), key=lambda k: int(k)):
                t_id = votes[v_id]
                if t_id is not None:
                    vote_details.append(f"player_{v_id} voted for player_{t_id}")
            
            if vote_details:
                line = f"[Day {r} Voting Details]: {', '.join(vote_details)}\n"
            else:
                continue 
        
        # 振り分けロジック
        if r < current_round:
            # 過去のラウンド -> Public Log
            if r > current_processing_round:
                public_log_str += f"-- Round {r} --\n"
                current_processing_round = r
            public_log_str += line
        else:
            # 現在のラウンド -> Current Situation
            current_log_str += line

    # =================================================================
    # セクションの結合
    # =================================================================
    
    # 2. [Public Game Log]
    prompt += "=== [Public Game Log] (KNOWN BY EVERYONE) ===\n"
    if public_log_str:
        prompt += public_log_str
    else:
        prompt += "(Game just started, no history yet.)\n"
    prompt += "\n"

    # 3. [Current Situation]
    prompt += "=== [Current Situation] (LIVE UPDATES) ===\n"
    alive_str = ", ".join([f"player_{p}" for p in sorted(alive_players)])
    prompt += f"Current Phase: {phase} {current_round}\n"
    prompt += f"Alive Players: {alive_str}\n"
    prompt += "Latest Events & Discussion:\n"
    
    if current_log_str:
        prompt += current_log_str
    else:
        prompt += "(No events yet in this round.)\n"

    # =================================================================
    # 4. マーカー (YOU ARE HERE)
    # =================================================================
    if phase == "night":
        prompt += "\n--> IT IS NIGHT. CHOOSE YOUR ACTION.\n"
    elif phase == "day_discussion":
        prompt += "\n--> YOU ARE HERE. IT IS YOUR TURN TO SPEAK.\n"
    elif phase == "day_voting":
        prompt += "\n--> IT IS VOTING TIME. MAKE YOUR DECISION.\n"
    
    return prompt

def get_action_prompt(observation, available_actions, strategy_id=None, is_deepseek=False):
    """
    論文 Appendix C.2, C.3, C.4 に基づくアクション選択プロンプトを生成します。
    Instruction Block の上書きバグを修正し、+= で結合するように変更しています。
    """
    phase = observation.get('phase')
    n_round = observation.get('round')
    p_id = observation.get('player_id')
    role = observation.get('role')
    known_whites = []
    known_blacks = []
    investigated_history = []

    action_history = observation.get('action_history', [])
    for entry in action_history:
        if entry.get('action_type') == 'investigate':
            target = entry.get('target')
            result = entry.get('result')
            investigated_history.append(f"Player {target} ({result})")
            if result == 'Not Werewolf':
                known_whites.append(target)
            elif result == 'Werewolf':
                known_blacks.append(target)

    # Role Instruction (変更なし)
    role_instruction = ""
    if role == "werewolf":
        role_instruction = (
            f"IMPORTANT: You are player_{p_id} and a Werewolf. To win, you must HIDE your identity. "
            "You can pretend to be a Villager, a Seer, or a Doctor depending on the situation. "
            "Do NOT reveal that you are a Werewolf. "
            "You may sacrifice your teammate if necessary to gain trust."
        )
    elif role == "seer":
        role_instruction = (
            f"IMPORTANT: You are player_{p_id} and the Seer. Your inspection is the Village's most powerful weapon. "
             "Remaining passive or hiding information often benefits the Werewolves. "
            "It is highly recommended to reveal your identity (Coming Out) if you find a Werewolf, as this is the best way to convince others. "
            "Even if you only find Villagers, clearing them helps narrow down suspects. Aim to guide the discussion actively to ensure the Village's victory."
        )
    elif role == "doctor":
        role_instruction = (
            f"IMPORTANT: You are player_{p_id} and the Doctor. Your goal is to protect the Seer or confirmed innocents. "
            "You can also protect yourself to survive. "
            "Especially, if the Seer reveals themselves, protecting them contributes significantly to the Village's victory. "
            "Therefore, while you should generally keep your role secret (e.g., by pretending to be a Villager), "
            "revealing your role (Coming Out) is also a valid strategy if you judge it beneficial for the Village side."
        )
    else: # Villager
        role_instruction = (
            f"IMPORTANT: You are player_{p_id} and a Villager. Reason logically to find the Werewolves based on facts and inconsistencies. "
            "Listen to others carefully and vote for the most suspicious player."
        )
    
    prompt = ""
    
    instruction_block = (
        f"\n====YOUR ROLE & OBJECTIVE====\n"
        f"{role_instruction}\n"
    )

    # --- C.2 Prompt for Secret Actions (Night) ---
    if phase == "night":
        if role == "villager":
            return None
            
        action_verb = "kill"
        if role == "seer": action_verb = "see"
        if role == "doctor": action_verb = "save"
        
        actions_str = ", ".join([f"{action_verb} player_{p}" for p in available_actions if p is not None])
        
        prompt += instruction_block
        if n_round == 1:
             prompt += (
                "\n[NIGHT 1 CONSTRAINT]\n"
                "This is the very first night. There are NO past interactions to consider.\n"
                "Do NOT arbitrarily create imaginary stories or scenarios about past events to justify your choice.\n"
                "Make your decision based on pure strategy or intuition (randomness).\n"
             )
        prompt += f"Now it is night {n_round} round, you (and your teammate) should choose one player to {action_verb}.\n"
        prompt += f"you should first reason about the current situation, then choose from the following actions: {actions_str}\n"
        if role == "seer" and (known_whites or known_blacks):
                prompt += f"\n[MEMORY] You have already investigated players: {investigated_history}. Do not investigate them again.\n"
        # ▼▼▼ モデルに応じた制約メッセージの定義 ▼▼▼
        if is_deepseek:
            # DeepSeek用: 思考暴走を抑える強い制約
            prompt+="You should only respond in JSON format as described below.\n"
            prompt+="IMPORTANT: Keep the 'reasoning' field CONCISE and SHORT (less than 50 words). "
            prompt+="Do not explain your entire thought process inside the JSON.\n"
            prompt+="Response Format:\n"
        else:
            # 通常用 (Llama, Qwenなど): 標準的な制約
            prompt+="You should only respond in JSON format as described below.\n"
            prompt+="Response Format:\n"
        prompt += "{\n"
        prompt += '  "reasoning": "reason about the current situation",\n'
        prompt += f'  "action": "{action_verb} player_i"\n'
        prompt += "}\n"
        prompt += "Ensure the response can be parsed by Python json.loads. Do not write any code or explanations outside the JSON. Do not use Markdown code blocks.\nJSON:"

    # --- C.3 Prompt for Discussion Actions (Day) ---
    elif phase == "day_discussion":
        prompt += instruction_block
        # 初日(Round 1)専用の強力な制約プロンプト（初日に殺されたプレイヤーへの議論を制限するため））
        if n_round == 1:
             prompt += (
                "\n[DAY 1 CONSTRAINT]\n"
                "There is NO pre-game conversation (Day 0) in this game. "
                "Therefore, there are NO past interactions regarding the player who died last night (whoever it may be).\n"
                "Do NOT arbitrarily create imaginary stories or scenarios about past events to discuss.\n"
                "Instead, discuss based ONLY on current facts (who was killed, who is alive) and future motives.\n\n"
             )
        prompt += f"Now it is day {n_round} discussion phase and it is your turn to speak.\n"
        prompt += f"Before speaking to the other players, you should first reason the current situation only to yourself, and then speak to all other players.\n"

        # strategy_id が指定されている場合はその戦略を入れる
        if strategy_id is not None and strategy_id in STRATEGY_DEFINITIONS:
            # 指定された戦略を挿入
            prompt += f"\n{STRATEGY_DEFINITIONS[strategy_id]}\n"
        
        # ▼▼▼ モデルに応じた制約メッセージの定義 ▼▼▼
        if is_deepseek:
            # DeepSeek用: 思考暴走を抑える強い制約
            prompt+="You should only respond in JSON format as described below.\n"
            prompt+="IMPORTANT: Keep the 'reasoning' field CONCISE and SHORT (less than 50 words). "
            prompt+="Do not explain your entire thought process inside the JSON.\n"
            prompt+="Response Format:\n"
        else:
            # 通常用 (Llama, Qwenなど): 標準的な制約
            prompt+="You should only respond in JSON format as described below.\n"
            prompt+="Response Format:\n"
        prompt += "{\n"
        prompt += '  "reasoning": "reason about the current situation only to yourself",\n'
        prompt += '  "statement": "speak to all other players"\n'
        prompt += "}\n"
        prompt += "Ensure the response can be parsed by Python json.loads. Do not write any code or explanations outside the JSON. Do not use Markdown code blocks.\nJSON:"

    # --- C.4 Prompt for Voting Actions (Day) ---
    elif phase == "day_voting":
        objective = "maximize the Werewolves' benefit" if role == "werewolf" else "vote for one player that is most likely to be a Werewolf"
        actions_str = ", ".join([f"vote for player_{p}" for p in available_actions if p is not None])
        
        prompt += instruction_block
        prompt += f"Now it is day {n_round} voting phase, you must vote for one player to {objective}.\n"
        prompt += f"You should first reason about the current situation, and then choose from the following actions: {actions_str}\n"
        # ▼▼▼ モデルに応じた制約メッセージの定義 ▼▼▼
        if is_deepseek:
            # DeepSeek用: 思考暴走を抑える強い制約
            prompt+="You should only respond in JSON format as described below.\n"
            prompt+="IMPORTANT: Keep the 'reasoning' field CONCISE and SHORT (less than 50 words). "
            prompt+="Do not explain your entire thought process inside the JSON.\n"
            prompt+="Response Format:\n"
        else:
            # 通常用 (Llama, Qwenなど): 標準的な制約
            prompt+="You should only respond in JSON format as described below.\n"
            prompt+="Response Format:\n"
        prompt += "{\n"
        prompt += '  "reasoning": "reason about the current situation",\n'
        prompt += '  "action": "vote for player_i"\n'
        prompt += "}\n"
        prompt += "Ensure the response can be parsed by Python json.loads. Do not write any code or explanations outside the JSON. Do not use Markdown code blocks.\nJSON:"

    return prompt
