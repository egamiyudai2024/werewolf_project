import os
import json
import time
from openai import OpenAI
from agents.base_agent import BaseAgent
from utils.data_utils import format_obs_to_prompt, get_action_prompt

class GPTAgent(BaseAgent):
    def __init__(self, player_id, role, model_name="gpt-4o"):
        super().__init__(player_id, role)
        # APIキーは環境変数から取得
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name

    def get_action(self, observation, phase, available_actions):
        """
        GPT-4oに状況を投げて、JSON形式で行動を受け取る
        """
        # 1. LSPOと同じ関数を使ってプロンプトを作成
        context_prompt = format_obs_to_prompt(observation)
        instruction_prompt = get_action_prompt(observation, available_actions)
        
        # 行動不要なフェーズならスキップ
        if instruction_prompt is None:
            return {"phase": phase}

        full_prompt = context_prompt + "\n\n" + instruction_prompt + "\n\nRespond strictly in JSON format."

        # 2. APIリクエスト (失敗時のリトライ付き)
        max_retries = 3
        for i in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a professional Werewolf player. Analyze the game logically. Output valid JSON only."},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                return json.loads(content)
            except Exception as e:
                print(f"GPT Error (Attempt {i+1}): {e}")
                time.sleep(2)

        # どうしてもダメな場合はパスする（エラー落ち防止）
        return {"statement": "Thinking...", "target": None}

    def predict_roles(self, observation):
        """
        役職予測（今回は簡易的に実装、必要ならプロンプトを書く）
        """
        return {}
