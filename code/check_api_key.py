#check_api_key.py
import os
import openai

print("--- OpenAI API Key Check ---")

# 1. 環境変数からAPIキーを読み込む
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not found.")
    print("Please make sure you have set the environment variable correctly.")
else:
    print("SUCCESS: Found API key in environment variables.")
    # マスクしてキーの一部を表示（安全のため）
    print(f"   API Key starts with: {api_key[:5]}... and ends with: ...{api_key[-4:]}")

    try:
        # 2. OpenAIクライアントを初期化
        client = openai.OpenAI(api_key=api_key)
        print("\nAttempting to connect to OpenAI API...")

        # 3. シンプルで有効なリクエストを送信
        # 空のリストではなく、必ず成功するテキストを渡す
        test_input = ["hello world"]
        response = client.embeddings.create(
            input=test_input,
            model="text-embedding-3-small"
        )

        # 4. 成功したか確認
        embedding_vector = response.data[0].embedding
        print("SUCCESS: Successfully received a response from OpenAI!")
        print(f"   - Received an embedding vector with {len(embedding_vector)} dimensions.")
        print("   - This confirms your API key is working correctly.")

    except openai.AuthenticationError:
        print("ERROR: Authentication failed. Your API key is incorrect or invalid.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")

print("\n--- Check Finished ---")