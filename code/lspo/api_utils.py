#api_utils.py
import os
import openai

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OpenAI key not found. Please set the OPENAI_API_KEY environment variable.")
openai.api_key = os.getenv("OPENAI_API_KEY")

_EMBEDDING_CACHE = {}

def get_embeddings(texts, model="text-embedding-3-small"):
    """
    OpenAI APIを使用してテキストの埋め込みベクトルを取得する.
    意味をなさないプレースホルダー文字列をフィルタリングで除去し、
    空の文字列は '[SILENCE]' という特別なトークンに置き換えて処理する.
    """
    if not texts:
        return []
 
    # 除去すべき、意味をなさないプレースホルダー文字列のブラックリストを定義
    INVALID_PLACEHOLDERS = {
        '...',
        'Your statement here'
    }

    # 包括的なクリーニング処理
    processed_texts = []

    to_fetch_indices = [] # APIで取得する必要があるテキストの(元のインデックス, テキスト)
    
    # 最終的な結果を格納するリスト (キャッシュ済み or 新規取得で埋める)
    final_embeddings = [None] * len(texts)

    for i, text in enumerate(texts):
        if text is None or not isinstance(text, str):
            final_embeddings[i] = [0.0] * 1536 # ダミー
            continue
            
        stripped = text.strip()
        if stripped in INVALID_PLACEHOLDERS:
            final_embeddings[i] = [0.0] * 1536
            continue

        target_text = "[SILENCE]" if not stripped else stripped
        
        # キャッシュにあるか確認
        if target_text in _EMBEDDING_CACHE:
            # キャッシュヒット: APIを呼ばずにメモリから取得
            final_embeddings[i] = _EMBEDDING_CACHE[target_text]
        else:
            # キャッシュミス: 後でまとめてAPIを呼ぶリストに追加
            to_fetch_indices.append((i, target_text))
            processed_texts.append(target_text)

    # 診断用のデバッグ出力を残しておく
    #print("\n--- [DEBUG] Cleaned data sent to OpenAI Embedding API ---")
    #print(processed_texts)
    #print("-------------------------------------------------------\n")

    if not processed_texts:
        # フィルターをかけた結果、リストが空になった場合
        return [e if e is not None else [0.0]*1536 for e in final_embeddings]
        
    try:
        response = openai.embeddings.create(input=processed_texts, model=model)

        for j, item in enumerate(response.data):
            original_idx, txt = to_fetch_indices[j]
            emb = item.embedding
            
            # キャッシュに登録
            _EMBEDDING_CACHE[txt] = emb 
            # 結果リストに格納
            final_embeddings[original_idx] = emb

        return final_embeddings
        
    except openai.BadRequestError as e:
        print(f"FATAL: Error calling OpenAI API even after comprehensive cleaning.")
        print(f"The data printed above is the cause. Please analyze it.")
        print(f"Original Error: {e}")
        return []
    except Exception as e:
        print(f"An unexpected API error occurred: {e}")
        return []
        
 
