import time
import torch
import config
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# api_utils の場所に合わせてインポート (lspoフォルダ内にあると想定)
try:
    from lspo.api_utils import get_embeddings
except ImportError:
    # もし直下にある場合
    from api_utils import get_embeddings

def benchmark_api(num_trials=5):
    print("\n" + "="*50)
    print(" [1] OpenAI API Latency Test (text-embedding-3-small)")
    print("="*50)
    
    test_texts = ["This is a test sentence for measuring embedding latency.", "Another short sentence."]
    
    times = []
    print(f"Testing {num_trials} times...")
    
    for i in range(num_trials):
        start = time.time()
        _ = get_embeddings(test_texts)
        end = time.time()
        duration = end - start
        times.append(duration)
        print(f"  Trial {i+1}: {duration:.4f} sec")
        time.sleep(1) # レート制限回避のため少し待機

    avg_time = sum(times) / len(times)
    print(f"\nResult: Average API Latency = {avg_time:.4f} sec / request")
    
    if avg_time > 2.0:
        print("WARNING: API response is slow. This will significantly delay latent space construction.")

def benchmark_local_llm():
    print("\n" + "="*50)
    print(" [2] Local LLM Load & Inference Test (Qwen2.5-32B)")
    print("="*50)

    print(f"Model Path: {config.BASE_LLM_MODEL}")
    print("Loading model options (4-bit quantization)...")

    # Trainer.py と同じ設定でロード時間を計測
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # --- Load Time Measurement ---
    start_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_LLM_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16
    )
    end_load = time.time()
    load_time = end_load - start_load
    print(f"\nResult: Model Loading Time = {load_time:.2f} sec")
    
    # --- Inference Speed Measurement ---
    print("\nStarting Inference Speed Test (Generating 50 tokens)...")
    prompt = "Explain the strategy of the Werewolf game in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt").to(config.DEVICE)
    
    # Warmup (初回は遅いことがあるため)
    print("  Warming up...")
    _ = model.generate(**inputs, max_new_tokens=10)

    # Actual Test
    start_inf = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    end_inf = time.time()
    
    inf_time = end_inf - start_inf
    print(f"  Generated Text: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    print(f"\nResult: Inference Time (50 tokens) = {inf_time:.4f} sec")
    
    if load_time > 300: # 5分以上
        print("WARNING: Model loading is extremely slow. Check disk I/O.")
    if inf_time > 10.0:
        print("WARNING: Inference is slow. This will delay Self-Play and Evaluation.")

if __name__ == "__main__":
    print(f"Running benchmark on: {config.DEVICE}")
    
    # 1. API計測
    try:
        benchmark_api()
    except Exception as e:
        print(f"API Test Failed: {e}")

    # 2. ローカルLLM計測
    try:
        benchmark_local_llm()
    except Exception as e:
        print(f"Local LLM Test Failed: {e}")
