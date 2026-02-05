#latent_space.py
import os
from tqdm import tqdm#時間のかかるループ処理の進捗状況をプログレスバーで表示し、視覚的に分かりやすく
import numpy as np
from sklearn.cluster import KMeans #k-meansクラスタリングを実行するためのクラス
from sklearn.metrics import silhouette_score
from game.environment import WerewolfGame
from agents.lspo_agent import LSPOAgent
import config
import json
from .api_utils import get_embeddings #テキストをベクトルに変換する関数

def generate_self_play_data(agent_class, agent_components, game_config, num_games, iteration_num, start_game_idx=0):
    #エージェントのクラス：LSPOAgent/エージェントを動かすのに必要なもの（言語モデル/トークナイザ/CPUおよびGPUの設定）/ゲーム設定/シミュレートするゲーム総数/現在のLSPOイテレーション番号
    print(f"Generating self-play data for {num_games} games (Starting from ID {start_game_idx})...")
    all_discussion_data = {role: [] for role in game_config['ROLES']}#役職ごとに発話データを格納するためのからの辞書を作成
    log_dir = "game_logs"; os.makedirs(log_dir, exist_ok=True) #log_dirの作成及び存在しない場合はgame_logsというディレクトリを作成
    #for game_idx in tqdm(range(num_games), desc="Self-Play Simulation"):#指定されたゲーム回数だけシミュレーションを繰り返す/tqdmによってループの進捗バー（プログレスバー）を表示、見やすくする/descバーの左側に表示する説明文
    for i in tqdm(range(num_games), desc="Self-Play Simulation"):
        current_game_idx = start_game_idx + i
        game = WerewolfGame(game_config['NUM_PLAYERS'], game_config['ROLES']); game.reset() #ゲーム環境インスタンスを割くせし、reset()でゲームの初期化
        players = game.players
        agents = {p.id: agent_class(p.id, p.role, agent_components) for p in players} #プレイヤーごとに、プレイヤーID/プレイヤーの枠色/言語モデル,トークナイザ等の設定を割り当てる
        while not game.game_over: #ゲームが終了するまでループを実行
            current_phase = game.phase; actions_to_execute = {}#現在のフェーズを取得し、各プレイヤーの行動を格納するための空の辞書を初期化
            if current_phase == "day_announcement":
                game.step({}) # 空のアクションを渡して時間を進める
                continue
# ⬇⬇⬇ [修正] 議論フェーズの逐次・複数ラウンド処理 ⬇⬇⬇
            if current_phase == "day_discussion":
                # 1. 発言順をシャッフル (日付ごとにランダム)
                speakers_order = game.get_shuffled_alive_players()
                
                # 2. 複数ラウンド (今回は2回) ループ
                # これにより、1周目の他者の発言を踏まえて2周目に発言できる
                NUM_DISCUSSION_ROUNDS = 2
                
                for _ in range(NUM_DISCUSSION_ROUNDS):
                    for player_id in speakers_order:
                        # 死亡確認 (議論中に死亡することはないが、念のため)
                        if not game.players[player_id].is_alive:
                            continue

                        # 直前の発言が含まれた最新の観測情報を取得
                        obs = game.get_observation_for_player(player_id)
                        current_agent = agents[player_id]
                        
                        # アクション決定
                        # ここで available_actions は空リストで渡す(議論フェーズは自由発話なので選択肢不要)
                        action = current_agent.get_action(obs, current_phase, [])
                        
                        # 発言を即座に環境に記録
                        statement = action.get("statement", "...")
                        game.record_discussion_step(player_id, statement)
                        
                        # 学習用データの保存 (発言候補がある場合)
                        if "statement_candidates" in action:
                            role = players[player_id].role
                            # ここでのobsは「自分の発言直前」の状態なので学習データとして適切
                            data_point = {"observation": obs, "candidates": action["statement_candidates"]}
                            all_discussion_data[role].append(data_point)
                
                # 3. 議論終了後、フェーズを強制的に投票へ進める
                game.phase = "day_voting"
            
            # ⬇⬇⬇ それ以外のフェーズ（夜・投票）は従来通りの一斉処理 ⬇⬇⬇
            else:
                players_to_act = [p.id for p in game._get_alive_players()]
                
                # 夜フェーズで行動可能なプレイヤーのみに絞る処理は environment.get_available_actions に任せる
                # ただし、agentに渡すときは全員ループで回し、agent側で空なら何もしない判断をする
                
                for player_id in players_to_act:
                    obs = game.get_observation_for_player(player_id) 
                    available_actions = game.get_available_actions(player_id) 
                    
                    current_agent = agents[player_id]
                    action = current_agent.get_action(obs, current_phase, available_actions)
                    actions_to_execute[player_id] = action 
                
                # 一斉に時間を進める
                game.step(actions_to_execute) 
            
            if game.round > 10: print(f"Warning: Game {current_game_idx+1} reached 10 rounds, forcing end."); break 
            
        #log_filename = os.path.join(log_dir, f"game_log_iter_{iteration_num}_game_{game_idx+1}.json") 
        log_filename = os.path.join(log_dir, f"game_log_iter_{iteration_num}_game_{current_game_idx+1}.json")
        with open(log_filename, 'w', encoding='utf-8') as f:
            log_data = {"roles": {p.id: p.role for p in players}, "winner": game.winner, "log": game.game_log}
            json.dump(log_data, f, indent=2, ensure_ascii=False)
            
    print("Finished generating self-play data.")
    return all_discussion_data

# ▼▼▼ [追加] 最適なKを探すヘルパー関数 ▼▼▼
def _find_optimal_k(embeddings_np, role_name):
    """
    シルエット分析を用いて、指定された範囲内で最適なクラスタ数Kを決定する。
    """
    min_k = config.MIN_CLUSTERS
    max_k = config.MAX_CLUSTERS
    n_samples = len(embeddings_np)

    # 安全策: データ数が少なすぎる場合の処理
    # データ数が min_k 未満なら、データ数そのものをKとする（全点をクラスタ化）
    if n_samples < min_k:
        print(f"[LatentSpace] Warning: Not enough data for {role_name} (n={n_samples}). Setting k={n_samples}.")
        return max(1, n_samples)
    
    # 上限の調整: データ数以上のクラスタは作れない
    effective_max_k = min(max_k, n_samples - 1)
    if effective_max_k < min_k:
        return min_k

    best_k = min_k
    best_score = -1.0
    
    print(f"[LatentSpace] Analyzing optimal K for role '{role_name}' (Range: {min_k}-{effective_max_k})...")

    # 指定範囲で探索
    for k in range(min_k, effective_max_k + 1):
        try:
            # 乱数シードを固定して再現性を確保
            kmeans = KMeans(n_clusters=k, random_state=config.SEED, n_init='auto')
            labels = kmeans.fit_predict(embeddings_np)
            
            # シルエット係数を計算 (-1 to 1, 1が良い)
            # データ数が多すぎる(>10000)場合は計算コスト削減のためサンプリングしても良いが、
            # 今回は「妥協なし」のため全データで計算する。
            score = silhouette_score(embeddings_np, labels)
            
            # print(f"  K={k}: Silhouette Score = {score:.4f}") # 詳細デバッグ用
            
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            print(f"  Warning: Clustering failed for K={k}: {e}")
            continue

    print(f"[LatentSpace] >> Selected Optimal K={best_k} (Score: {best_score:.4f}) for {role_name}")
    return best_k

def construct_latent_space(discussion_data, role, num_clusters=None):
    """
    収集した発話をK-Meansでクラスタリングし、潜在戦略空間を構築する。
    num_clusters が None の場合、シルエット分析により自動決定する。
    """
    # 1. データの収集
    all_utterances = [cand for item in discussion_data.get(role, []) for cand in item['candidates']]
    
    if not all_utterances:
        print(f"[LatentSpace] No utterances found for role: {role}")
        return None

    # 2. Embeddingの取得
    embeddings = get_embeddings(all_utterances) 
    if not embeddings: 
        return None 
    
    embeddings_np = np.array(embeddings)
    
    # 3. クラスタ数(K)の決定
    if num_clusters is None:
        # 自動決定ロジック (今回追加)
        final_k = _find_optimal_k(embeddings_np, role)
    else:
        # 指定された場合 (従来のロジックや強制指定)
        final_k = num_clusters
        # データ数チェック
        if len(embeddings_np) < final_k:
            print(f"[LatentSpace] Warning: Data count ({len(embeddings_np)}) < Requested K ({final_k}).")
            return None

    # 4. 最終的なK-Meansモデルの作成
    kmeans = KMeans(n_clusters=final_k, random_state=config.SEED, n_init='auto') 
    kmeans.fit(embeddings_np)
    
    return kmeans
    
#def construct_latent_space(discussion_data, role, num_clusters=None): #収集した発話をk-means方でクラスタリングすることで、潜在戦略空間に抽象化
#    all_utterances = [cand for item in discussion_data.get(role, []) for cand in item['candidates']]#roleをキーにして、itemの中にあるcandidatesを取り出して、一つずつcandに展開　→　例）seerの前候補発言を1つのリストにまとめた
#    embeddings = get_embeddings(all_utterances) #全ての発話テキストを数値ベクトルにエンコーディング
#    if not embeddings: return None #エラー処理
#    embeddings_np = np.array(embeddings) #データ数がクラスタ数より少ない場合はクラスタリングができないため、処理を中断
#        #get_embeddings が失敗した場合や、データ数がクラスタ数より少ない場合はクラスタリングができないため、処理を中断
#    if len(embeddings_np) < num_clusters: return None
#    kmeans = KMeans(n_clusters=num_clusters, random_state=config.SEED, n_init='auto') #k-meansモデルを初期化 クラスタ数k/乱数のシードを固定し、毎度同じ初期値でクラスタリングを開始/自動的に初期化回数を選ぶ
#    kmeans.fit(embeddings_np)
#    return kmeans
