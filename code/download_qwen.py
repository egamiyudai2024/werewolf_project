import os
from huggingface_hub import snapshot_download

# --- 設定 ---
# 保存したいモデルのID
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"

# 保存先パス (絶対パスで指定して間違いを防ぐ)
# ~/werewolf_project/models/Qwen-2.5-32B-Instruct に保存されます
SAVE_DIR = os.path.expanduser(f"~/werewolf_project/models/{MODEL_ID.split('/')[-1]}")

# フォルダ作成
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Start downloading {MODEL_ID} to {SAVE_DIR} ...")

# --- ダウンロード実行 ---
# resume_download=True: 通信が切れても途中から再開可能
# local_dir_use_symlinks=False: 実体をダウンロード（重要）
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=SAVE_DIR,
    local_dir_use_symlinks=False,
    resume_download=True
)

print("Download completed successfully!")
