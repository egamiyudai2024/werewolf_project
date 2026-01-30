#latent_space.py
import os
from tqdm import tqdm#時間のかかるループ処理の進捗状況をプログレスバーで表示し、視覚的に分かりやすく
import numpy as np
from sklearn.cluster import KMeans #k-meansクラスタリングを実行するためのクラス
from game.environment import WerewolfGame
from agents.lspo_agent import LSPOAgent
import config
import json
from .api_utils import get_embeddings #テキストをベクトルに変換する関数

def generate_self_play_data(agent_class, agent_components, game_config, num_games, iteration_num):
    #エージェントのクラス：LSPOAgent/エージェントを動かすのに必要なもの（言語モデル/トークナイザ/CPUおよびGPUの設定）/ゲーム設定/シミュレートするゲーム総数/現在のLSPOイテレーション番号
    print(f"Generating self-play data for {num_games} games...")
    all_discussion_data = {role: [] for role in game_config['ROLES']}#役職ごとに発話データを格納するためのからの辞書を作成
    log_dir = "game_logs"; os.makedirs(log_dir, exist_ok=True) #log_dirの作成及び存在しない場合はgame_logsというディレクトリを作成
    for game_idx in tqdm(range(num_games), desc="Self-Play Simulation"):#指定されたゲーム回数だけシミュレーションを繰り返す/tqdmによってループの進捗バー（プログレスバー）を表示、見やすくする/descバーの左側に表示する説明文
        game = WerewolfGame(game_config['NUM_PLAYERS'], game_config['ROLES']); game.reset() #ゲーム環境インスタンスを割くせし、reset()でゲームの初期化
        players = game.players
        agents = {p.id: agent_class(p.id, p.role, agent_components) for p in players} #プレイヤーごとに、プレイヤーID/プレイヤーの枠色/言語モデル,トークナイザ等の設定を割り当てる
        while not game.game_over: #ゲームが終了するまでループを実行
            current_phase = game.phase; actions_to_execute = {}#現在のフェーズを取得し、各プレイヤーの行動を格納するための空の辞書を初期化
            if current_phase == "day_discussion": players_to_act = [p.id for p in game._get_alive_players()] #昼の行動は全員行動するので生存プレイヤー全てが対象
            else: players_to_act = [p.id for p in game._get_alive_players() if game.get_available_actions(p.id)] #夜の行動は特定のプレイヤーのみが対象なのでその判定後、生存プレイヤーが対象
            for player_id in players_to_act:
                obs = game.get_observation_for_player(player_id) #対象となるプレイヤーが観測できる情報を取得　情報の内容：プレイヤーID/役割/生存者/現在のラウンド/現在のフェーズ/生存プレイヤー/ゲームログ/プライベートログ
                available_actions = game.get_available_actions(player_id) #今できる行動の一覧を取得（発言、投票、夜のアクション等）
                if not available_actions and current_phase != "day_discussion": continue #行動できることが何もなく、昼の話し合いフェーズでない場合はスキップ
                if not available_actions: available_actions = [p.id for p in game._get_alive_players() if p.id != player_id] or [player_id] #行動できることがなく、昼のフェーズ（村人の話）の場合、生きていりうプレイヤーの中から自分以外を選択（いない場合は自分自身）
                current_agent = agents[player_id]#AIエージェントを取り出す
                action = current_agent.get_action(obs, current_phase, available_actions)#今の状況（観測状況・フェーズ・可能な行動）を渡して、どの行動を選ぶか決定させる
                actions_to_execute[player_id] = action #選んだ行動を保存
                if current_phase == "day_discussion" and "statement_candidates" in action:#ディスカッション時、エージェントのステートメントを観測情報とともに保存
                #昼のフェーズかつ発言候補が存在する場合
                    role = players[player_id].role
                    data_point = {"observation": obs, "candidates": action["statement_candidates"]} #他のプレイヤーの発言履歴や情報を含む観測情報/エージェントが生成した発言候補のリスト 入力と出力のペアを作成
                    all_discussion_data[role].append(data_point) #そのデータをデータファイルに追加
            game.step(actions_to_execute) #ゲームを1ステップ進める
            if game.round > 10: print(f"Warning: Game {game_idx+1} reached 10 rounds, forcing end."); break #ゲームが10ラウンドを超えた場合、強制的に終了
        log_filename = os.path.join(log_dir, f"game_log_iter_{iteration_num}_game_{game_idx+1}.json") #ゲーム終了後、全記録をjsonファイルに保存
        with open(log_filename, 'w', encoding='utf-8') as f:#encoding：日本語等も文字映えせず保存可能/with分によって書き込み後に自動的に閉じる
            log_data = {"roles": {p.id: p.role for p in players}, "winner": game.winner, "log": game.game_log}
            json.dump(log_data, f, indent=2, ensure_ascii=False)#ensure_ascii=Falseによって日本語等も文字化けせず保存可能
    print("Finished generating self-play data.")
    return all_discussion_data

def construct_latent_space(discussion_data, role, num_clusters): #収集した発話をk-means方でクラスタリングすることで、潜在戦略空間に抽象化
    all_utterances = [cand for item in discussion_data.get(role, []) for cand in item['candidates']]#roleをキーにして、itemの中にあるcandidatesを取り出して、一つずつcandに展開　→　例）seerの前候補発言を1つのリストにまとめた
    embeddings = get_embeddings(all_utterances) #全ての発話テキストを数値ベクトルにエンコーディング
    if not embeddings: return None #エラー処理
    embeddings_np = np.array(embeddings) #データ数がクラスタ数より少ない場合はクラスタリングができないため、処理を中断
        #get_embeddings が失敗した場合や、データ数がクラスタ数より少ない場合はクラスタリングができないため、処理を中断
    if len(embeddings_np) < num_clusters: return None
    kmeans = KMeans(n_clusters=num_clusters, random_state=config.SEED, n_init='auto') #k-meansモデルを初期化 クラスタ数k/乱数のシードを固定し、毎度同じ初期値でクラスタリングを開始/自動的に初期化回数を選ぶ
    kmeans.fit(embeddings_np)
    return kmeans