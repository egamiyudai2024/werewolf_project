
#/werewolf_project/code/config.py
import os
import torch

# ==========================================
# 環境自動判定とパス設定
# ==========================================
# 現在のユーザー名を取得
CURRENT_USER = os.getenv("USER")

# 環境ごとのルートディレクトリ定義
if CURRENT_USER == "ku50002337":
    # --- [玄界 Genkai] ---
    print("Config: Detected GENKAI environment.")
    # 指定されたディレクトリ構造に基づくルート
    ROOT_DIR = "/home/pj25001061/ku50002337/werewolf_project"
    BASE_LLM_MODEL = os.path.join(ROOT_DIR, "pretrained_models", "Qwen2.5-32B-Instruct")

elif CURRENT_USER == "yudai2024":
    # --- [Shrike (Lab Server)] ---
    print("Config: Detected SHRIKE environment.")
    # 指定されたディレクトリ構造 (/v2lspo/...)
    ROOT_DIR = "/home/yudai2024/v2lspo/werewolf_project"
    
    # Shrikeの場合、「以下玄界と同じ」であればモデルも配置されている可能性がありますが、
    # 万が一同期されていない場合は HuggingFace Hub からダウンロードするようにフォールバックを設定
    local_model_path = os.path.join(ROOT_DIR, "pretrained_models", "Qwen2.5-32B-Instruct")
    if os.path.exists(local_model_path):
        BASE_LLM_MODEL = local_model_path
    else:
        # ローカルになければHFから取得 (Shrikeはネット接続可と想定)
        BASE_LLM_MODEL = "Qwen/Qwen2.5-32B-Instruct"

else:
    # --- [その他 Local] ---
    print("Config: Detected LOCAL environment.")
    ROOT_DIR = os.path.join(os.getenv("HOME"), "werewolf_project")
    BASE_LLM_MODEL = "Qwen/Qwen2.5-32B-Instruct"
    EMBEDDING_MODEL = "text-embedding-3-small"

EMBEDDING_MODEL = "text-embedding-3-small"

# ==========================================
# 共通パス設定
# ==========================================
# ログとモデルの保存先 (ROOT_DIR以下に作成)
LOG_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_SAVE_DIR = os.path.join(ROOT_DIR, "models")

# 完了した学習モデルを保管するアーカイブ用ディレクトリ
MODEL_ARCHIVE_DIR = os.path.join(ROOT_DIR, "models_archive")

# ディレクトリ自動作成
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_ARCHIVE_DIR, exist_ok=True)

# APIキー (Shrike/Local用)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==========================================
# 学習パラメータ設定
# ==========================================

# --- General ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_NAME = "werewolf_lspo"
SEED = 42

# --- Game Settings ---
NUM_PLAYERS = 7
ROLES = {
    "werewolf": 2,
    "seer": 1,
    "doctor": 1,
    "villager": 3
}

# --- LSPO Trainer Settings ---
NUM_ITERATIONS = 3 

# --- Latent Space Construction Settings ---
GAMES_PER_ITERATION_FOR_DATA = 10  # テスト実行用
CANDIDATE_ACTIONS_PER_TURN = 3    # 高速化のため2推奨
#INITIAL_K_WEREWOLF = 3 
#INITIAL_K_VILLAGE = 2 
MIN_CLUSTERS = 3
MAX_CLUSTERS = 7

# --- Policy Optimization (Deep CFR) Settings ---
# 【重要】高速化・最適化後の推奨値
CFR_TRAIN_ITERATIONS = 3000  # 理想は3000
CFR_BUFFER_SIZE = 200000      # 理想は200000. 5000
CFR_BATCH_SIZE = 512        #理想は512.   256
CFR_LEARNING_RATE = 1e-3
CFR_MAX_DEPTH = 5           # 深さ制限 (Rolloutへの切り替えライン)

# --- Network Settings ---
# 入力次元増大に伴い、隠れ層を拡張
CFR_HIDDEN_DIM = 2048


# --- Game Logic Settings ---
MAX_ROUNDS = 5           # 7人村ならDay 3で決着
DISCUSSION_TURNS = 2     # 1日あたりの発言ターン数

# --- Embedding Settings ---
EMBEDDING_DIM = 64

# --- State Dimension Calculation (妥協なき固定設計) ---
# 1. Basic Info (22次元)
#    - ID(7) + Role(4) + Round(1) + Phase(3) + Alive(7) = 22
dim_basic = 22

# 2. History Slots (各ラウンドごとの行動履歴)
#    - 各ラウンド: Secret(7) + Vote(49) + Dead(7) = 63
#    - 全体: 63 * 3ラウンド = 189
dim_history = 63 * MAX_ROUNDS

# 3. Discussion Slots (議論ベクトル)
# 1ラウンドあたりの次元数: 7人 * 2ターン * 64次元 = 896
dim_discussion_per_round = 7 * DISCUSSION_TURNS * EMBEDDING_DIM

# (A) Current Discussion Slots (当日の議論)
dim_discussion_current = dim_discussion_per_round

# (B) Past Discussion Slots (過去の議論)
# 変更: 平均化せず、過去のラウンド数分だけフルのスロットを用意する
# (5 - 1) * 896 = 3584 次元
dim_discussion_past = (MAX_ROUNDS - 1) * dim_discussion_per_round

# Total Dimension
# 22 + 315 + 896 + 3584 = 4817 次元
STATE_DIM = dim_basic + dim_history + dim_discussion_current + dim_discussion_past

#STATE_DIM = 85
MAX_ACTION_DIM = {
    "werewolf": 7,
    "seer": 7,
    "doctor": 7,
    "villager": 7
}




# --- Latent Space Expansion (DPO) Settings ---
DPO_BATCH_SIZE = 4       
DPO_EPOCHS = 2
DPO_LEARNING_RATE = 1e-6 # 論文推奨値
DPO_BETA = 0.1           

# QLoRA Settings
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

