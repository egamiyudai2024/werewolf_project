#api_utils.py
import os
import openai

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OpenAI key not found. Please set the OPENAI_API_KEY environment variable.")
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    for text in texts:
        # Noneでないことと、文字列であることを確認
        if text is None or not isinstance(text, str):
            continue
        
        # 空白を除去した文字列
        stripped_text = text.strip()

        # ブラックリストに含まれているかチェック
        if stripped_text in INVALID_PLACEHOLDERS:
            continue # ブラックリストの文字列は完全に無視する

        # 空文字列かチェックし、[SILENCE] に置き換える
        if not stripped_text:
            processed_texts.append("[SILENCE]")
        else:
            processed_texts.append(stripped_text)

    # 診断用のデバッグ出力を残しておく
    #print("\n--- [DEBUG] Cleaned data sent to OpenAI Embedding API ---")
    #print(processed_texts)
    #print("-------------------------------------------------------\n")

    if not processed_texts:
        # フィルターをかけた結果、リストが空になった場合
        return []
        
    try:
        response = openai.embeddings.create(input=processed_texts, model=model)
        return [item.embedding for item in response.data]
    except openai.BadRequestError as e:
        print(f"FATAL: Error calling OpenAI API even after comprehensive cleaning.")
        print(f"The data printed above is the cause. Please analyze it.")
        print(f"Original Error: {e}")
        return []
    except Exception as e:
        print(f"An unexpected API error occurred: {e}")
        return []
        
 
