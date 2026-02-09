# agents/lspo_agent.py
import torch
import json
import re
import random
import time
import numpy as np
import os
from agents.base_agent import BaseAgent
from utils.data_utils import format_obs_to_prompt, get_action_prompt, format_obs_to_vector
from lspo.api_utils import get_embeddings

class LSPOAgent(BaseAgent):
    def __init__(self, player_id, role, agent_components, is_eval=False):
        super().__init__(player_id, role)
        self.llm = agent_components['llm']
        self.tokenizer = agent_components['tokenizer']
        self.device = agent_components['device']
        self.is_eval = is_eval
        #予測ログ専用ディレクトリ
        self.predict_log_dir = "debug_predict"
        os.makedirs(self.predict_log_dir, exist_ok=True)
        
        # CFRネットワーク
        all_cfr_nets = agent_components.get('cfr_net', {})
        self.my_cfr_net = all_cfr_nets.get(role) if all_cfr_nets else None
        
        # KMeansモデル
        raw_kmeans = agent_components.get('kmeans')
        if isinstance(raw_kmeans, dict) and role in raw_kmeans:
            self.my_kmeans = raw_kmeans[role]
        else:
            self.my_kmeans = raw_kmeans
        
	# プロンプトログ保存用のディレクトリ作成
        self.log_dir = "prompt_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.debug_log_dir = "debug_logs"
        os.makedirs(self.debug_log_dir, exist_ok=True)

    def _query_llm(self, prompt, max_new_tokens=250, save_dir=None, filename=None):
        tokenized_prompt = self.tokenizer.encode(prompt)
        if len(tokenized_prompt) > 7500: 
            prompt = "..." + prompt[-7000:]
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.9,      # 温度0.9で多様性を確保
            top_p=0.9,
            repetition_penalty=1.1
        )
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_only = response_text[len(prompt):]
        
        # --- 指定されたディレクトリに生のログを保存（デバッグ用） ---
        if save_dir and filename:
            try:
                log_path = os.path.join(save_dir, filename)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("=== PROMPT ===\n")
                    f.write(prompt + "\n\n")
                    f.write("=== RAW LLM RESPONSE ===\n")
                    f.write(response_only + "\n")
            except Exception as e:
                print(f"[Warning] Failed to save debug log: {e}")
        # -------------------------------------------------------
        
        json_obj = self._extract_json(response_only)
        if json_obj: return json_obj
        return {}
    
    def _query_llm_batch(self, prompts, max_new_tokens=250, save_dir=None, filename_prefix=None):
        """
        複数のプロンプトをまとめてGPUで並列処理し、スループットを向上させる。
        """
        # 1. トークナイザ設定 (生成タスクでは左パディングが必須)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. バッチトークン化 (padding=Trueで最長系列に合わせる)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.llm.device)
        
        # 3. 生成実行
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.9, # 多様性確保のための温度
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        results = []
        # 4. 結果のデコードと抽出
        for i, output in enumerate(outputs):
            # プロンプト部分を除去して応答のみを取り出す
            # (入力トークン長を使ってスライスするのが最も正確)
            input_len = inputs.input_ids[i].shape[0]
            generated_tokens = output[input_len:]
            
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # JSON抽出
            json_obj = self._extract_json(response_text)
            results.append(json_obj)

            # --- ログ保存 (デバッグ用) ---
            if save_dir and filename_prefix:
                try:
                    log_path = os.path.join(save_dir, f"{filename_prefix}_{i}.txt")
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(f"=== PROMPT {i} ===\n{prompts[i]}\n\n=== RESPONSE ===\n{response_text}\n")
                except Exception:
                    pass
                
        return results


    def _extract_json(self, text):
        # 既存のロジックを維持
        start_indices = [i for i, char in enumerate(text) if char == '{']
        for start in start_indices:
            balance = 0; in_string = False; escape = False
            for i in range(start, len(text)):
                char = text[i]
                if in_string:
                    if escape: escape = False
                    elif char == '\\': escape = True
                    elif char == '"': in_string = False
                else:
                    if char == '"': in_string = True
                    elif char == '{': balance += 1
                    elif char == '}':
                        balance -= 1
                        if balance == 0:
                            candidate = text[start : i+1]
                            candidate_clean = re.sub(r'//.*', '', candidate)
                            candidate_clean = re.sub(r'/[*].*?[*]/', '', candidate_clean, flags=re.DOTALL)
                            #candidate_clean = re.sub(r'/\*.*?\*/', '', candidate_clean, flags=re.DOTALL)
                            try: return json.loads(candidate_clean)
                            except: pass
                            break
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match: return json.loads(match.group(0))
        except: pass
        return None

    def get_action(self, observation, phase, available_actions, candidate_actions_per_turn=None):
        # 1. 評価モードかつCFRがある場合、夜・投票はCFRのみで決定 (LLM不使用)
        #if self.is_eval and self.my_cfr_net and phase in ["night", "day_voting"]:
        #    return self._get_action_via_cfr_direct(observation, phase, available_actions)

        # 1. 評価モードの際、LLMのみ、かつループ回数1回で自己対戦を行う
        if self.is_eval: #and self.my_cfr_net and phase in ["night", "day_voting"]:
            candidate_actions_per_turn = 1
            #print("[Eval Mode] Setting candidate_actions_per_turn to 1 for evaluation.")
        elif candidate_actions_per_turn is None:
            candidate_actions_per_turn = 3  # デフォルトは3候補生成

        # 2. プロンプト生成とバッチ推論の準備
        batch_prompts = []
        
        # 共通部分の作成
        context_prompt = format_obs_to_prompt(observation)
        #constraint_prompt = "\nIMPORTANT: Output ONLY the JSON object. Do not output any thinking process outside the JSON. Do not use Markdown code blocks.\nJSON:"

        # A. バッチプロンプトの構築ループ
        for i in range(candidate_actions_per_turn):
            # 戦略IDを決定 (0: Aggressive, 1: Defensive, 2: Strategic)

            # ループ回数に応じて循環させる
            strategy_id = i % 3
            if self.is_eval: #評価時は戦略としてNoneを返す
                strategy_id = None
                #print("[Eval Mode] Using strategy_id = None for evaluation.")

            # data_utils に ID を渡して指示パートを取得
            # (data_utils側で strategy_id に対応した戦略テキストが挿入される)
            instruction_prompt = get_action_prompt(observation, available_actions, strategy_id=strategy_id)
            
            # Villagerの夜フェーズなど、アクションが不要な場合は None が返る
            if instruction_prompt is None: 
                 return {"phase": phase}

            # 完全なプロンプトを構築してリストに追加
            full_prompt = context_prompt + "\n" + instruction_prompt #+ constraint_prompt
            batch_prompts.append(full_prompt)

        # B. バッチ推論の実行 (高速化)
        timestamp = int(time.time())
        round_num = observation.get('round', 0)
        filename_prefix = f"{phase}_round{round_num}_p{self.player_id}_ts{timestamp}"
        
        # ここで3つのプロンプトを並列に処理する
        responses = self._query_llm_batch(
            batch_prompts, 
            save_dir=self.debug_log_dir, 
            filename_prefix=filename_prefix
        )
        
        # C. 候補リストの構築
        candidates = []
        for response_json in responses:
            if not response_json: continue
            
            if phase == "day_discussion":
                val = response_json.get("statement", "I will pass my turn to think.")
                candidates.append(val)
            elif phase in ["night", "day_voting"]:
                candidates.append(response_json)
        
        # 候補が一つも生成できなかった場合のフォールバック
        if not candidates:
            if phase == "day_discussion":
                return {"phase": phase, "statement": "I will pass my turn to think."}
            return {"phase": phase} # 棄権

        # ------------------------------------------------------------------
        # 選択ロジック (既存ロジック維持)
        # ------------------------------------------------------------------

        # === A. 議論フェーズ (Day Discussion) ===
        if phase == "day_discussion":
            # 1. 評価モード (CFR + KMeans で最適解を選ぶ)
            if self.is_eval and self.my_cfr_net and self.my_kmeans:
                return self._select_discussion_via_cfr(observation, phase, candidates)
            
            # 2. 学習モード (ランダム選択)
            # 生成された3つの戦略(Aggressive, Defensive, Strategic)から1つをランダムに選んで実行する
            # これにより、自己対戦データに多様な戦略が蓄積される
            selected_statement = random.choice(candidates)
            selected_cluster_id = -1
            if self.my_kmeans:
                try:
                    emb = get_embeddings([selected_statement])
                    if emb:
                        selected_cluster_id = self.my_kmeans.predict(np.array(emb))[0]
                except Exception: pass

            return {
                "phase": phase,
                "statement_candidates": candidates,
                "statement": selected_statement,
                "cluster_id": int(selected_cluster_id)
            }

        # === B. 夜・投票フェーズ (学習時: ランダム選択) ===
        else:
            # 学習時は生成された候補からランダムに選ぶことで探索を行う
            selected_json = random.choice(candidates)
            return self._parse_discrete_action(phase, selected_json, available_actions)

            

    def _get_action_via_cfr_direct(self, observation, phase, available_actions):
        """
        評価時専用: LLMを使わず、CFRネットワークのRegret値に従って直接行動を選択する
        """
        try:
            obs_vector = format_obs_to_vector(observation)
            net_device = next(self.my_cfr_net.parameters()).device
            state_tensor = torch.from_numpy(obs_vector).float().to(net_device).unsqueeze(0)
            
            with torch.no_grad():
                predicted_regrets = self.my_cfr_net(state_tensor).cpu().numpy().flatten()
            
            # available_actions (List[int]) に含まれるアクションの中で、Regretが最大のものを探す
            best_action = None
            best_regret = -float('inf')
            
            valid_actions = [a for a in available_actions if isinstance(a, int) and a >= 0]
            
            # もし有効なアクションがなければ棄権
            if not valid_actions:
                return {"phase": phase, "target": None}

            for action_idx in valid_actions:
                # ネットワークの出力次元範囲内かチェック
                if action_idx < len(predicted_regrets):
                    regret = predicted_regrets[action_idx]
                    if regret > best_regret:
                        best_regret = regret
                        best_action = action_idx
            
            # 何も選べなかった場合 (範囲外など) はランダムフォールバック
            if best_action is None:
                best_action = None #random.choice(valid_actions)
                
            if phase == "night":
                return {"phase": phase, "target": best_action}
            elif phase == "day_voting":
                return {"phase": phase, "target": best_action}
                
        except Exception as e:
            print(f"Error in CFR direct selection: {e}")
            return {"phase": phase,  "target": phase} # エラー時は棄権

    def _select_discussion_via_cfr(self, observation, phase, candidates):
        """
        議論フェーズのCFR選択ロジック
        """
        try:
            embeddings = get_embeddings(candidates)
            if not embeddings: raise ValueError("Embedding failed")
            
            cluster_ids = self.my_kmeans.predict(np.array(embeddings))
            
            obs_vector = format_obs_to_vector(observation)
            net_device = next(self.my_cfr_net.parameters()).device
            state_tensor = torch.from_numpy(obs_vector).float().to(net_device).unsqueeze(0)
            
            with torch.no_grad():
                predicted_regrets = self.my_cfr_net(state_tensor).cpu().numpy().flatten()
            
            best_candidate_idx = -1
            best_regret = -float('inf')
            best_cluster_id = -1
            
            for idx, cluster_id in enumerate(cluster_ids):
                if cluster_id < len(predicted_regrets):
                    regret = predicted_regrets[cluster_id]
                    if regret > best_regret:
                        best_regret = regret
                        best_candidate_idx = idx
                        best_cluster_id = cluster_id
            
            if best_candidate_idx != -1:
                return {
                    "phase": phase,
                    "statement_candidates": candidates,
                    "statement": candidates[best_candidate_idx],
                    "cluster_id": int(best_cluster_id)
                }
        except Exception:
            pass
        
        # 失敗時はランダム (フォールバック)
        selected_idx = random.randrange(len(candidates))
        selected_statement = candidates[selected_idx]
        selected_cluster_id = -1
        # クラスタID再計算は省略(エラー時なので)
        
        return {
            "phase": phase,
            "statement_candidates": candidates,
            "statement": selected_statement,
            "cluster_id": int(selected_cluster_id)
        }

    def _parse_discrete_action(self, phase, json_obj, available_actions):
        """
        LLMのJSON出力からターゲットIDを抽出し、アクション形式で返す
        """
        chosen_action_str = json_obj.get("action")
        if not chosen_action_str: chosen_action_str = json_obj.get("vote")
        if not chosen_action_str: chosen_action_str = json_obj.get("target")
        if not chosen_action_str: chosen_action_str = str(json_obj)

        chosen_action = None
        match = re.search(r'(?:player_?|Player_?|\b)(\d+)', str(chosen_action_str))
        
        if match:
            action_num = int(match.group(1))
            if action_num in available_actions:
                chosen_action = action_num
            else:
                # 無効IDならNone (棄権)
                chosen_action = None 
        else:
            chosen_action = None

        if phase == "night":
            return {"phase": phase, "target": chosen_action}
        elif phase == "day_voting":
            return {"phase": phase, "target": chosen_action}
        return {"phase": phase, "target": None}

    def predict_roles(self, observation, game_idx=0):
        # 既存コードと同じ
        prompt = format_obs_to_prompt(observation)
                # 2. 情報抽出
        my_role = self.role
        my_id = self.player_id
        teammates = observation.get("teammates", [])

        prompt = prompt.replace("--> IT IS VOTING TIME. MAKE YOUR DECISION.", "")
        prompt += "\n[TASK]\nBased on the game history, predict the roles of all other players.\n"
        prompt += "\n[CONSTRAINTS]\n1. Use ONLY these four labels: 'werewolf', 'seer', 'doctor', 'villager'.\n"
        prompt += "2. Do NOT use 'unknown', 'dead', or multiple roles (e.g., 'villager or doctor').\n"
        prompt += "3. You must ssign exactly one role per player, ensuring the total count matches the game setup (2 Werewolves, 1 Seer, 1 Doctor, 3 Villagers).\n"
        # 要件3: 役職内訳の提示
        prompt += "\n[GAME SETUP] \nThere are 7 players total. The role distribution is:\n"
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
        
        response_json = self._query_llm(prompt, max_new_tokens=500)

        # ### MODIFIED ###: ログ保存
        # LSPOAgentの _query_llm は内部で既にログ保存機能を持つ場合がありますが、
        # 予測専用ディレクトリに、より詳細な情報を保存します。
        timestamp = int(time.time())
        round_num = observation.get('round', 0)
        
        filename = f"pred_LSPO_G{game_idx}_R{round_num}_p{self.player_id}_ts{timestamp}.txt"
        log_path = os.path.join(self.predict_log_dir, filename)
        
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"=== LSPO ROLE PREDICTION LOG ===\n")
            f.write(f"Game: {game_idx} | Round: {round_num} | Player: {self.player_id}\n\n")
            f.write(f"--- PROMPT ---\n{prompt}\n\n")
            f.write(f"--- EXTRACTED JSON ---\n{json.dumps(response_json, indent=2, ensure_ascii=False)}\n")


        predictions = {}
        valid_roles = {'werewolf', 'seer', 'doctor', 'villager'}
        
        if isinstance(response_json, dict):
            for key, value in response_json.items():
                pid_match = re.search(r'(\d+)', str(key))
                if pid_match and isinstance(value, str) and value.lower() in valid_roles:
                    predictions[int(pid_match.group(1))] = value.lower()
        return predictions
