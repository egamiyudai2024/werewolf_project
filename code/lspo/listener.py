# lspo/listener.py (New File)
import numpy as np
from .api_utils import get_embeddings

class Listener:
    def __init__(self, embedding_dim=64): # 軽量化のため64次元を採用（configと合わせる）
        self.embedding_dim = embedding_dim

    def process_discussion(self, discussion_logs, num_players=7, max_turns=2):
        """
        当日の議論ログを解析し、プレイヤーごとの発言Embeddingスロットを作成する。
        Return: (num_players * max_turns * embedding_dim) の1次元配列
        """
        # スロット初期化: [Player0_Turn1, Player0_Turn2, ..., Player6_Turn2]
        slots = np.zeros((num_players, max_turns, self.embedding_dim), dtype=np.float32)
        
        # プレイヤーごとの発言カウンタ
        turn_counts = {i: 0 for i in range(num_players)}

        # ログを走査して埋める
        for log in discussion_logs:
            pid = log.get('player_id')
            statement = log.get('statement')
            
            if pid is not None and 0 <= pid < num_players:
                current_turn = turn_counts[pid]
                if current_turn < max_turns:
                    # Embedding取得 (API経由)
                    emb = get_embeddings([statement]) # list[list[float]]
                    if emb and len(emb) > 0:
                        # 次元圧縮が必要な場合はここでPCA等をするが、
                        # 今回はAPI側で次元指定するか、単純に先頭を使うか、設定に依存。
                        # ※ configで指定した次元数に合わせる処理が必要
                        # 簡易実装: 取得したベクトルをスライシングして格納
                        vec = np.array(emb[0])[:self.embedding_dim]
                        
                        # パディング（もし次元が足りない場合）
                        if len(vec) < self.embedding_dim:
                            vec = np.pad(vec, (0, self.embedding_dim - len(vec)))
                            
                        slots[pid, current_turn] = vec
                    
                    turn_counts[pid] += 1

        return slots.flatten()

    def summarize_history(self, full_game_log, current_round):
        """
        過去の日数分の要約Embeddingを作成する。
        (Day 1 Summary, Day 2 Summary...)
        """
        # 今回は簡易的に「過去のVoting Result」や「死者情報」は
        # 既存のベクトルにあるため、ここでは「過去の会話の雰囲気」等は
        # 複雑になるため一旦ゼロベクトルでプレースホルダーとするか、
        # 必要に応じて実装する。
        # ※ ユーザーとの合意に基づき、ここもスロットとして用意する。
        
        # 履歴スロット数: 3日分 (Day1, Day2, Day3)
        history_slots = np.zeros((3, self.embedding_dim), dtype=np.float32)
        
        # 本来はここで LLM に "Summarize Day 1..." と投げるが、
        # 処理時間を考慮し、今回は実装枠のみ確保する。
        return history_slots.flatten()