#expansion.py
import torch
#from transformers import TrainingArguments
from trl import DPOConfig, DPOTrainer
from datasets import Dataset
from utils.data_utils import format_obs_to_prompt, format_obs_to_vector
from lspo.api_utils import get_embeddings
import numpy as np
# 【修正】PeftModel をインポートに追加
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

def prepare_dpo_dataset(discussion_data, role_cfr_nets, kmeans_models, device):
    preference_data = []
    print("Preparing DPO dataset...")

    for role, data in discussion_data.items():
        if role not in role_cfr_nets or role not in kmeans_models or kmeans_models[role] is None:
            continue

        cfr_net = role_cfr_nets[role]
        kmeans = kmeans_models[role]
        
        for item in data:
            obs_vector = format_obs_to_vector(item['observation'])
            if obs_vector is None: continue
            state_tensor = torch.from_numpy(obs_vector).float().to(device).unsqueeze(0)

            with torch.no_grad():
                regrets = cfr_net(state_tensor).cpu().numpy().flatten()

            candidates = item['candidates']
            if len(candidates) < 2:
                continue

            embeddings = get_embeddings(candidates)
            if not embeddings:
                continue
            
            latent_strategies = kmeans.predict(embeddings)
            
            candidate_regrets = {}
            for cand, ls_id in zip(candidates, latent_strategies):
                if ls_id < len(regrets):
                    if cand not in candidate_regrets or regrets[ls_id] < candidate_regrets[cand]:
                        candidate_regrets[cand] = regrets[ls_id]

            if len(candidate_regrets) < 2:
                continue

            sorted_candidates = sorted(candidate_regrets.items(), key=lambda x: x[1], reverse=True)
            # Regretが高いほど「良い行動」であるため、降順（大きい順）にソート # reverse=True を追加
            chosen_statement = sorted_candidates[0][0]
            rejected_statement = sorted_candidates[-1][0]

            if chosen_statement == rejected_statement:
                continue

            prompt = format_obs_to_prompt(item['observation'])
            if prompt is None: continue
            
            preference_data.append({
                "prompt": prompt,
                "chosen": chosen_statement,
                "rejected": rejected_statement
            })

    if not preference_data:
        print("Warning: No preference data could be generated for DPO.")
        return None

    print(f"DPO dataset prepared with {len(preference_data)} samples.")
    return Dataset.from_list(preference_data)


def fine_tune_with_dpo(base_model, tokenizer, dpo_dataset, config):
    if dpo_dataset is None:
        print("Skipping DPO fine-tuning as no data is available.")
        return base_model

    print("Starting DPO fine-tuning...")

    # --- 【修正】アダプタ重複回避のための分岐ロジック ---
    peft_config = None

    # モデルが既に PeftModel (アダプタ付き) かどうかをチェック
    if isinstance(base_model, PeftModel):
        print("=== DETECTED EXISTING ADAPTER: Continuing training on existing adapter ===")
        # 既存のアダプタを継続学習モードにする
        base_model.gradient_checkpointing_enable()
        base_model = prepare_model_for_kbit_training(base_model)
        
        # 既存のLoRAパラメータを学習対象(requires_grad=True)にする
        for name, param in base_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        
        # peft_config は None にする (DPOTrainerに新しいアダプタを作らせないため)
        peft_config = None

    else:
        # 初回 (Iter 0) または純粋なベースモデルの場合
        print("=== NO ADAPTER DETECTED: Initializing new LoRA adapter ===")
        # 1. 4bitモデルをPEFT (QLoRA) でトレーニング可能にするための準備
        base_model.gradient_checkpointing_enable()
        base_model = prepare_model_for_kbit_training(base_model)

        # 2. LoRA (PEFT) の設定を定義 (新規作成)
        peft_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            target_modules=config.LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )
    # ----------------------------------------------------
    
    dpo_config = DPOConfig(
        per_device_train_batch_size=config.DPO_BATCH_SIZE,
        num_train_epochs=config.DPO_EPOCHS,
        learning_rate=config.DPO_LEARNING_RATE,
        output_dir=f"{config.MODEL_SAVE_DIR}/dpo_iter",
        logging_steps=10,
        remove_unused_columns=False,
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        bf16=True, 
        beta=config.DPO_BETA, 
        max_prompt_length=1024,
        max_length=1536,
    )

    dpo_trainer = DPOTrainer(
        model=base_model,
        ref_model=None, 
        args= dpo_config,
        train_dataset=dpo_dataset,
        #tokenizer=tokenizer,
        peft_config=peft_config, # ここがNoneなら既存アダプタを使用、あれば新規作成
        #config=dpo_config, 
    )
    dpo_trainer.tokenizer = tokenizer

    
    print("DPOTrainer initialized. Starting training...")
    dpo_trainer.train()
    print("DPO fine-tuning finished.")
    
    return dpo_trainer.model
