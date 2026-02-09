#agents/base_agent.py
import torch
import json
import re
import os
import time
from abc import ABC, abstractmethod
from utils.data_utils import format_obs_to_prompt, get_action_prompt

class BaseAgent(ABC):
    def __init__(self, player_id, role, agent_components=None):
        self.player_id = player_id
        self.role = role
        # コンポーネントが渡された場合は保持（BaselineAgent等で使用）
        if agent_components:
            self.llm = agent_components.get('llm')
            self.tokenizer = agent_components.get('tokenizer')
            self.device = agent_components.get('device')
        

    @abstractmethod
    def get_action(self, observation, phase, available_actions):
        """
        現在の観測情報と可能な行動リストから行動を決定する
        observation: 辞書型の観測データ
        phase: 現在のフェーズ ("night", "day_discussion", "day_voting")
        available_actions: 選択可能なアクションIDのリスト
        """
        pass

    @abstractmethod
    def predict_roles(self, observation):
        """
        現在の観測情報から他プレイヤーの役職を予測する（評価用）
        """
        pass


class BaselineAgent(BaseAgent):
    """
    未学習モデル（Iter 0）評価用のReActエージェント。
    CFRやクラスタリングを使用せず、プロンプトエンジニアリングのみで推論を行う。
    常に strategy_id=None (特定の戦略バイアスなし) で動作する。
    """
    def __init__(self, player_id, role, agent_components, model_name="baseline"):
        super().__init__(player_id, role, agent_components)
        self.model_name = model_name  # モデル名を保存
        self.log_dir = "debug_logs_baseline"
        os.makedirs(self.log_dir, exist_ok=True)
        #予測ログ専用ディレクトリ
        self.predict_log_dir = "debug_predict"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.predict_log_dir, exist_ok=True)

    def _query_llm(self, prompt, max_new_tokens=250):
        """LLMにクエリを投げ、テキスト応答を取得する"""
        try:
            tokenized_prompt = self.tokenizer.encode(prompt)
            # コンテキスト長対策
            if len(tokenized_prompt) > 28000:
                prompt = "..." + prompt[-27500:]
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
            
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7, # ベースラインは少し安定志向で
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # プロンプト部分を除去
            response_only = response_text[len(prompt):]
            return response_only
            
        except Exception as e:
            print(f"[Error] LLM Query Failed: {e}")
            return ""

    def _extract_json(self, text):
        # agents/base_agent.py 内
        """
        テキストからJSONオブジェクトを抽出する。
        DeepSeek-R1等の思考タグ対応版（思考終了後のテキストを優先し、後方から探索する）。
        """
        if not text: return None

        # --- 1. DeepSeek-R1対応: </think> で分割して後半を採用 ---
        if "</think>" in text:
            # タグがある場合、思考終了後の部分だけを取り出す
            text = text.split("</think>")[-1].strip()
        # -----------------------------------------------------

        # --- 2. 複数のJSON候補がある場合、最後尾のものを優先して探す ---
        # テキスト内のすべての '{' の位置を見つける
        start_indices = [i for i, char in enumerate(text) if char == '{']
        
        # 後ろから順にパースを試みる（最終回答が一番後ろにあるため）
        for start in reversed(start_indices):
            balance = 0
            for i in range(start, len(text)):
                if text[i] == '{': balance += 1
                elif text[i] == '}': balance -= 1
                
                if balance == 0:
                    candidate = text[start:i+1]
                    # クリーニング
                    candidate = re.sub(r'//.*', '', candidate)
                    try:
                        return json.loads(candidate)
                    except:
                        # パース失敗なら次の候補（より前のJSON）へ
                        break
        
        # --- 3. フォールバック: 正規表現でMarkdownコードブロック等を探す ---
        try:
            # ```json ... ``` のパターンを優先
            match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            # 単純なブロック検索
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass

        return None

    def _parse_discrete_action(self, phase, json_obj, available_actions):
        """JSONからアクションIDを抽出する"""
        if not json_obj:
            # JSONパース失敗時は警告を出し、Noneアクションを返す（キーは必ず含める）
            print(f"[BaselineAgent Warning] JSON parse failed in {phase}. Action will be None.")
            if phase == "night":
                return {"phase": phase, "target": None}
            elif phase == "day_voting":
                return {"phase": phase, "target": None}
            return {"phase": phase}
            #return {"phase": phase} # 棄権

        # 複数のキーパターンに対応
        chosen_action_str = json_obj.get("action") or json_obj.get("vote") or json_obj.get("target") or str(json_obj)
        
        # 数値の抽出
        match = re.search(r'(?:player_?|Player_?|\b)(\d+)', str(chosen_action_str))
        chosen_action = None
        
        if match:
            action_num = int(match.group(1))
            if action_num in available_actions:
                chosen_action = action_num
        
        if phase == "night":
            return {"phase": phase, "target": chosen_action}
        elif phase == "day_voting":
            return {"phase": phase, "target": chosen_action}
        
        return {"phase": phase}

    def get_action(self, observation, phase, available_actions):
        """
        観測情報に基づき、LLMに直接問い合わせてアクションを決定する。
        """
        # 1. 共通コンテキストプロンプト
        context_prompt = format_obs_to_prompt(observation)

        # ▼▼▼ 修正: DeepSeek判定を行い、引数として渡す ▼▼▼
        # モデル名に "deepseek" が含まれているかチェック (大文字小文字無視)
        is_deepseek = "deepseek" in self.model_name.lower()
        
        # 2. アクション指示プロンプト (strategy_id=None で純粋なReActを要求)
        instruction_prompt = get_action_prompt(observation, available_actions, strategy_id=None, is_deepseek=is_deepseek)
        
        # Villagerの夜フェーズなど、アクション不要な場合
        if instruction_prompt is None:
            return {"phase": phase}

        full_prompt = context_prompt + "\n" + instruction_prompt

        # ▼▼▼モデルに応じてトークン数を動的に決定 ▼▼▼
        if is_deepseek:
            # DeepSeekは思考プロセスが長いため大きく取る
            token_limit = 4096
        else:
            # Llama/Qwenは思考タグがないため、1024あれば十分余裕がある
            # (250だと理由説明が長くなった時に切れるリスクがあるため、1024推奨)
            token_limit = 1024

        # 3. 推論実行
        response_text = self._query_llm(full_prompt, max_new_tokens=token_limit)
        response_json = self._extract_json(response_text)

        # ログ保存 (デバッグ用)
        timestamp = int(time.time())
        round_num = observation.get('round', 0)

        # モデル名からファイル名に使えない文字を置換
        safe_name = self.model_name.replace("/", "_").replace(" ", "_")

        filename = f"{safe_name}_{phase}_round{round_num}_p{self.player_id}_ts{timestamp}.txt"
        with open(os.path.join(self.log_dir, filename), "w", encoding="utf-8") as f:
            f.write(f"PROMPT:\n{full_prompt}\n\nRAW LLM RESPONSE:\n{response_text}")

            # 2. 抽出結果 (修正確認用)
            f.write(f"=== EXTRACTED JSON ACTION ===\n")
            if response_json:
                f.write(json.dumps(response_json, indent=2, ensure_ascii=False))
            else:
                f.write("None (Extraction Failed)")

        # 4. レスポンス解析と返却
        if phase == "day_discussion":
            if response_json:
                return {
                    "phase": phase,
                    "statement": response_json.get("statement", "I have nothing to say."),
                    "reasoning": response_json.get("reasoning", "")
                }
            else:
                return {"phase": phase, "statement": "..."}
        
        else: # 夜・投票フェーズ
            return self._parse_discrete_action(phase, response_json, available_actions)

    def predict_roles(self, observation, game_idx=0):
        """役職予測の実行"""
        prompt = format_obs_to_prompt(observation)

        # 2. 情報抽出
        my_role = self.role
        my_id = self.player_id
        teammates = observation.get("teammates", [])

        prompt = prompt.replace("--> IT IS VOTING TIME. MAKE YOUR DECISION.", "")
        prompt += "\n[TASK]\nBased on the game history, predict the roles of all other players.\n"
        prompt += "\n[CONSTRAINTS]\n1. Use ONLY these four labels: 'werewolf', 'seer', 'doctor', 'villager'.\n"
        prompt += "2. Do NOT use 'unknown', 'dead', or multiple roles (e.g., 'villager or doctor').\n"
        prompt += "3. You must assign exactly one role per player, ensuring the total count matches the game setup (2 Werewolves, 1 Seer, 1 Doctor, 3 Villagers).\n"
        
        # 要件3: 役職内訳の提示
        prompt += "\n[GAME SETUP]\n There are 7 players total. The role distribution is:\n"
        prompt += "Total: 7 players. (2 Werewolves, 1 Seer, 1 Doctor, 3 Villagers)\n"

        # 要件2: 自己認識とチームメイトの提示 (Recency Bias対策)
        prompt += f"\n[IMPORTANT REMINDER]\n"
        prompt += f"You are player_{my_id}.\n"
        prompt += f"Your Role: {my_role.capitalize()}\n"


                # 人狼の場合、チームメイト情報を「予測のヒント」として再度強調する
        if self.role == "werewolf" and "teammates" in observation:
            teammates = observation["teammates"]
            if teammates:
                teammates_str = ", ".join([f"player_{t}" for t in teammates])
                # 「あなたはこれらを知っている」と明示する
                prompt += f"IMPORTANT HINT: You KNOW that {teammates_str} is your Werewolf teammate. Mark them as 'werewolf'.\n"
                prompt += "Use this knowledge to deduce who the Seer, Doctor and villager are among the remaining players.\n"
        #prompt += "Response JSON format: {\"player_id\": \"role\", ...}\n"
        prompt += "Response JSON format:\n"
        prompt += "{\n"
        prompt += '  "reasoning": "Analyze your previous actions and the players\' claims, voting patterns, and contradictions step-by-step.",\n'
        prompt += '  "player_0": "role_label",\n'
        prompt += '  "player_1": "role_label",\n'
        prompt += '  "player_2": "role_label",\n'
        prompt += '  "player_3": "role_label",\n'
        prompt += '  "player_4": "role_label",\n'
        prompt += '  "player_5": "role_label",\n'
        prompt += '  "player_6": "role_label"\n'
        prompt += "}\n"
        
        response_text = self._query_llm(prompt, max_new_tokens=500)
        response_json = self._extract_json(response_text)

        # ### MODIFIED ###: ログ保存処理
        timestamp = int(time.time())
        round_num = observation.get('round', 0)
        safe_model_name = self.model_name.replace("/", "_").replace(" ", "_")[:10]
        
        filename = f"pred_{safe_model_name}_G{game_idx}_R{round_num}_p{self.player_id}_ts{timestamp}.txt"
        log_path = os.path.join(self.predict_log_dir, filename)
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== ROLE PREDICTION LOG ===\n")
            f.write(f"Model: {self.model_name} | Game: {game_idx} | Round: {round_num} | Player: {self.player_id} ({self.role})\n\n")
            f.write(f"--- PROMPT ---\n{prompt}\n\n")
            f.write(f"--- RAW RESPONSE ---\n{response_text}\n\n")
            f.write(f"--- EXTRACTED JSON ---\n{json.dumps(response_json, indent=2, ensure_ascii=False) if response_json else 'Failed'}\n")
        # ### END MODIFIED ###
        
        predictions = {}
        valid_roles = {'werewolf', 'seer', 'doctor', 'villager'}
        
        if isinstance(response_json, dict):
            for key, value in response_json.items():
                # キーからIDを抽出 (例: "player_1" -> 1)
                pid_match = re.search(r'(\d+)', str(key))
                if pid_match and isinstance(value, str) and value.lower() in valid_roles:
                    predictions[int(pid_match.group(1))] = value.lower()
                    
        return predictions