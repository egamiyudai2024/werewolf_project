# lspo/listener.py
import numpy as np
import config
from .api_utils import get_embeddings

class Listener:
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim
        # 1ラウンドあたりの議論ベクトルサイズ: 7人 * 2ターン * 64次元
        # 空間化されたスロットの合計サイズを計算
        self.round_slot_size = config.NUM_PLAYERS * config.DISCUSSION_TURNS * self.embedding_dim

    def process_current_discussion(self, discussion_logs, num_players=7, max_turns=2):
        """
        当日の議論ログを解析し、[Player][Turn]ごとの発言Embeddingスロットを作成する。
        発言の順序と話者を保存する「空間化」処理を行う。
        
        Args:
            discussion_logs (list): 当日の議論ログのみを含むリスト
            num_players (int): プレイヤー数 (Default: 7)
            max_turns (int): 1日あたりの発言回数 (Default: 2)
            
        Returns:
            np.array: (num_players * max_turns * embedding_dim) の1次元配列
        """
        # 固定スロットの確保: (7, 2, 64)
        # これにより、「誰が」「何ターン目に」話したかという位置情報が固定される
        slots = np.zeros((num_players, max_turns, self.embedding_dim), dtype=np.float32)
        
        # プレイヤーごとの発言回数カウンタ（何番目のスロットに入れるか判定用）
        turn_counts = {i: 0 for i in range(num_players)}

        for log in discussion_logs:
            pid = log.get('player_id')
            statement = log.get('statement')
            
            # 有効なプレイヤーIDかつ発言内容がある場合のみ処理
            if pid is not None and 0 <= pid < num_players and statement:
                current_turn = turn_counts[pid]
                
                # 規定ターン数（スロット数）を超えていない場合のみ格納
                if current_turn < max_turns:
                    # テキストをベクトル化
                    emb = self._get_vector(statement)
                    # 指定位置に格納
                    slots[pid, current_turn] = emb
                    # カウンタを進める
                    turn_counts[pid] += 1

        # 1次元に平坦化して返す (CFRネットワークの入力形式に合わせる)
        return slots.flatten()

    def summarize_past_discussion(self, game_log, current_round):
        """
        過去の全ラウンド(1 ~ current_round-1)の議論内容をベクトル化する。
        【重要】要約や平均化を行わず、全ての過去ラウンドに対してフルサイズの
        スロット([Player][Turn])を割り当て、文脈（時系列情報）を空間情報として完全に保持する。
        
        Args:
            game_log (list): ゲーム全体のログ
            current_round (int): 現在のラウンド数
            
        Returns:
            np.array: ((MAX_ROUNDS - 1) * round_slot_size) の1次元配列
        """
        # 過去スロット数: MAX_ROUNDS - 1
        # config.MAX_ROUNDS=5 の場合、過去4ラウンド分を保持
        max_past_slots = config.MAX_ROUNDS - 1
        
        # 出力用配列の初期化 (全てゼロ)
        # 次元数: (4ラウンド分) * (896次元) = 3584次元
        past_slots_flat = np.zeros(max_past_slots * self.round_slot_size, dtype=np.float32)
        
        # Day 1 (Round 1) までは過去が存在しないため、ゼロベクトルを返す
        if current_round <= 1:
            return past_slots_flat

        # 1. 過去ログをラウンドごとに分類して抽出
        past_logs_by_round = {r: [] for r in range(1, current_round)}
        
        for log in game_log:
            r = log.get('round')
            ltype = log.get('type')
            # 議論ログのみ、かつ現在より前のラウンドのもの
            if ltype == 'discussion' and r is not None and r < current_round:
                past_logs_by_round[r].append(log)
        
        # 2. 各過去ラウンドについてベクトルを作成し、所定の位置に埋め込む
        # r=1 (Day 1) -> slot index 0
        # r=2 (Day 2) -> slot index 1
        # ...
        for r in range(1, config.MAX_ROUNDS):
            slot_idx = r - 1
            
            # スロット範囲内であれば処理
            if slot_idx < max_past_slots:
                # そのラウンドが現在のラウンドより過去である場合
                if r < current_round:
                    target_logs = past_logs_by_round.get(r, [])
                    
                    # process_current_discussionと同じロジック（空間化）を使用
                    # これにより、過去のラウンドも「誰が」「いつ」話したかが保存される
                    round_vec = self.process_current_discussion(
                        target_logs, 
                        num_players=config.NUM_PLAYERS, 
                        max_turns=config.DISCUSSION_TURNS
                    )
                    
                    # 3. 巨大な1次元配列上の適切な位置(オフセット)にコピー
                    start = slot_idx * self.round_slot_size
                    end = start + self.round_slot_size
                    past_slots_flat[start:end] = round_vec
                    
        return past_slots_flat

    def _get_vector(self, text):
        """
        テキストをEmbeddingし、指定次元にリサイズして返すヘルパー関数。
        APIエラーや空の結果に対する頑健性を持つ。
        """
        try:
            embs = get_embeddings([text])
            if embs and len(embs) > 0:
                return self._resize_vector(embs[0])
        except Exception:
            # エラー時はゼロベクトルでフォールバック（学習を止めないため）
            pass
        return np.zeros(self.embedding_dim, dtype=np.float32)

    def _resize_vector(self, vec):
        """
        取得したベクトルを self.embedding_dim (64次元) に厳密に合わせる。
        通常1536次元等のAPI出力を、先頭64次元へのスライシングまたはパディングで調整する。
        """
        vec = np.array(vec, dtype=np.float32)
        current_dim = len(vec)
        
        if current_dim > self.embedding_dim:
            # 次元が多い場合はスライス (先頭の成分を使用)
            return vec[:self.embedding_dim]
        elif current_dim < self.embedding_dim:
            # 次元が足りない場合はゼロパディング
            return np.pad(vec, (0, self.embedding_dim - current_dim))
        
        return vec