import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys
import os
import argparse

# プロジェクトのルートディレクトリをPythonのパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lspo.api_utils import get_embeddings 

# --- 設定 ---
ITERATION = 2 # 確認したいイテレーション番号

# ▼▼▼ 修正箇所: 全役職をリスト化 ▼▼▼
ROLES_TO_INSPECT = ['werewolf', 'seer', 'doctor', 'villager']
# ▲▲▲ 修正箇所 ▲▲▲

# --- 引数処理 ---
parser = argparse.ArgumentParser(description="Visualize latent space.")
parser.add_argument('--model_dir', type=str, default=None, help='Path to the model directory.')
args = parser.parse_args()

# --- 1. 学習済みモデルとデータのロード (ループの外で一度だけ行う) ---
print("Loading artifacts...")

if args.model_dir:
    base_dir = args.model_dir
    print(f"Using specified model directory: {base_dir}")
else:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
    print(f"Using default model directory: {base_dir}")

kmeans_path = os.path.join(base_dir, f'kmeans_models_iter_{ITERATION}.pkl')
data_path = os.path.join(base_dir, f'discussion_data_iter_{ITERATION}.pkl')

try:
    with open(kmeans_path, 'rb') as f:
        kmeans_models = pickle.load(f)
    with open(data_path, 'rb') as f:
        discussion_data = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: Could not find file. {e}")
    exit()

# 出力先ディレクトリ作成
output_dir = 'analysis_results'
os.makedirs(output_dir, exist_ok=True)

# ▼▼▼ 修正箇所: ここから役職ごとのループ開始 ▼▼▼
print(f"Starting analysis for roles: {ROLES_TO_INSPECT}")

for role in ROLES_TO_INSPECT:
    print("\n" + "="*40)
    print(f"Processing Role: {role}")
    print("="*40)

    kmeans = kmeans_models.get(role)
    data = discussion_data.get(role)

    if kmeans is None or data is None:
        print(f"No data found for role: {role} in iteration {ITERATION}. Skipping.")
        continue # exit() ではなく continue にして次の役職へ

    # --- 2. データの準備 ---
    print(f"Preparing data for {role}...")
    all_utterances = [candidate for entry in data for candidate in entry['candidates']]

    if not all_utterances:
        print(f"No utterances found for {role}. Skipping.")
        continue

    print("Generating embeddings...")
    all_embeddings = np.array(get_embeddings(all_utterances))

    if all_embeddings is None or len(all_embeddings) == 0:
        print(f"Failed to generate embeddings for {role}. Skipping.")
        continue

    cluster_labels = kmeans.predict(all_embeddings)

    # --- 3. t-SNEによる次元削減 ---
    print("Running t-SNE...")
    perplexity_value = min(30, len(all_embeddings) - 1)
    if perplexity_value <= 0:
        print(f"Not enough data points to run t-SNE for {role}. Skipping.")
        continue
        
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # --- 4. 可視化 ---
    print(f"Plotting for {role}...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=cluster_labels,
        palette=sns.color_palette("hsv", n_colors=kmeans.n_clusters),
        legend="full"
    )
    plt.title(f"Latent Space Visualization for '{role}' (Iteration {ITERATION+1})")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    
    output_path = os.path.join(output_dir, f'latent_space_{role}_iter_{ITERATION+1}.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    plt.close() # 【重要】メモリリーク防止のため図を閉じる

    # --- 5. 各クラスタの代表的な発言を確認 ---
    print(f"--- Sample Utterances from Each Cluster ({role}) ---")
    for i in range(kmeans.n_clusters):
        print(f"\nCluster {i}:")
        utterances_in_cluster = [utt for utt, label in zip(all_utterances, cluster_labels) if label == i]
        sample_size = min(3, len(utterances_in_cluster)) # 表示数を少し減らして見やすく
        if sample_size > 0:
            samples = random.sample(utterances_in_cluster, sample_size)
            for sample in samples:
                print(f"  - {sample[:100]}...") # 長すぎる場合はカット

print("\nAll roles processed.")
# ▲▲▲ 修正箇所 ▲▲▲

'''
#check_latent_space.py
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys
import os
import argparse  # 引数処理用に追加

# プロジェクトのルートディレクトリをPythonのパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lspo.api_utils import get_embeddings 

# --- 設定 ---
ITERATION = 2 # 確認したいイテレーション番号 (0から始まる)
ROLES_TO_INSPECT = ['werewolf', 'seer', 'doctor', 'villager']

# ▼▼▼ 追加・修正 (引数処理とディレクトリ決定ロジック) ▼▼▼
# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="Visualize latent space.")
parser.add_argument('--model_dir', type=str, default=None, help='Path to the model directory (e.g., models_archive/xxxx).')
args = parser.parse_args()

print("Loading artifacts...")

# 引数で指定があればそれを使い、なければデフォルト(../../models)を使う
if args.model_dir:
    base_dir = args.model_dir
    print(f"Using specified model directory: {base_dir}")
else:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
    print(f"Using default model directory: {base_dir}")

# --- 1. 学習済みモデルとデータのロード ---
#print("Loading artifacts...")
# analysis/から見て一つ上の階層 (プロジェクトルート) を指すようにパスを修正
#base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
kmeans_path = os.path.join(base_dir, f'kmeans_models_iter_{ITERATION}.pkl')
data_path = os.path.join(base_dir, f'discussion_data_iter_{ITERATION}.pkl')
#kmeans_path = f'../kmeans_models_iter_{ITERATION}.pkl'
#data_path = f'../discussion_data_iter_{ITERATION}.pkl'

try:
    with open(kmeans_path, 'rb') as f:
        kmeans_models = pickle.load(f)
    with open(data_path, 'rb') as f:
        discussion_data = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: Could not find file. {e}")
    print(f"Please make sure you have run the training for iteration {ITERATION},")
    print(f"and that '{kmeans_path}' and '{data_path}' exist in the project root directory.")
    exit()

kmeans = kmeans_models.get(ROLE_TO_INSPECT)
data = discussion_data.get(ROLE_TO_INSPECT)

if kmeans is None or data is None:
    print(f"No data found for role: {ROLE_TO_INSPECT} in iteration {ITERATION}")
    exit()

# --- 2. データの準備 ---
print("Preparing data for visualization...")
all_utterances = [candidate for entry in data for candidate in entry['candidates']]

print("Generating embeddings for visualization...")
all_embeddings = np.array(get_embeddings(all_utterances))

if all_embeddings is None or len(all_embeddings) == 0:
    print("Failed to generate embeddings.")
    exit()

cluster_labels = kmeans.predict(all_embeddings)

# --- 3. t-SNEによる次元削減 ---
print("Running t-SNE... (This may take a while)")
# t-SNEのperplexityはサンプル数より小さい必要があるため、安全策を追加
perplexity_value = min(30, len(all_embeddings) - 1)
if perplexity_value <= 0:
    print("Not enough data points to run t-SNE.")
    exit()
tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
embeddings_2d = tsne.fit_transform(all_embeddings)

# --- 4. 可視化 ---
print("Plotting...")
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    hue=cluster_labels,
    palette=sns.color_palette("hsv", n_colors=kmeans.n_clusters),
    legend="full"
)
plt.title(f"Latent Space Visualization for '{ROLE_TO_INSPECT}' (Iteration {ITERATION+1})")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")

output_dir = 'analysis_results'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'latent_space_{ROLE_TO_INSPECT}_iter_{ITERATION+1}.png')
plt.savefig(output_path)
print(f"Plot saved to {output_path}")


#os.makedirs('analysis_results', exist_ok=True)
#plt.savefig(f'analysis_results/latent_space_{ROLE_TO_INSPECT}_iter_{ITERATION+1}.png')
#print(f"Plot saved to analysis_results/latent_space_{ROLE_TO_INSPECT}_iter_{ITERATION+1}.png")
plt.show()

# --- 5. 各クラスタの代表的な発言を確認 ---
print("\n--- Sample Utterances from Each Cluster ---")
for i in range(kmeans.n_clusters):
    print(f"\n--- Cluster {i} ---")
    utterances_in_cluster = [utt for utt, label in zip(all_utterances, cluster_labels) if label == i]
    sample_size = min(5, len(utterances_in_cluster))
    if sample_size > 0:
        samples = random.sample(utterances_in_cluster, sample_size)
        for sample in samples:
            print(f"  - {sample}")
'''