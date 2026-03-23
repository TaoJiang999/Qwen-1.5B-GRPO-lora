import time
import torch
from dotenv import load_dotenv
load_dotenv()
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams



from trl.rewards import (
    accuracy_reward,
    think_format_reward,
    reasoning_accuracy_reward,
)


MODEL_PATH = "/home/waas/LLM/cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"  # 替换为你要评测的模型路径
DATASET_PATH = "/home/waas/LLM/cache/datasets--DEAR-Tao--gsm8k-test/snapshots/b53d22284e3244a0816cadbff17471869ee90b60"
SPLIT = "test"  
TENSOR_PARALLEL_SIZE = 1  

def main():
    print(f"🚀 开始加载模型进行评测: {MODEL_PATH}")
    
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    dataset = load_dataset(DATASET_PATH, split=SPLIT)
    
    
    raw_prompts = dataset["prompt"]
    solutions = dataset["solution"]
    
    print(f"📊 成功加载测试集，共 {len(dataset)} 条样本。")

    formatted_prompts = []
    for p in raw_prompts:
        prompt_str = tokenizer.apply_chat_template(
            p, 
            tokenize=False, 
            add_generation_prompt=True
        )
        formatted_prompts.append(prompt_str)

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=8192
    )
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8192)
    
    print(f"⚡ 开始 vLLM 批量推理...")
    start_time = time.time()
    outputs = llm.generate(formatted_prompts, sampling_params)
    print(f"✅ 推理完成，耗时: {time.time() - start_time:.2f} 秒")


    wrapped_completions = []
    
    for output in outputs:
        generated_text = output.outputs[0].text
        wrapped_completions.append([
            {"role": "assistant", "content": generated_text}
        ])

    print("⚖️ 开始计算 TRL 奖励分数...")
    
    fmt_scores = think_format_reward( 
        completions=wrapped_completions
    )
    
    reasoning_scores = reasoning_accuracy_reward(
        completions=wrapped_completions, 
        solution=solutions,
        reasoning_delimiters = ["</think>"]
    )
    
    acc_scores = accuracy_reward(
        completions=wrapped_completions, 
        solution=solutions
    )

    fmt_scores = [x if x is not None else 0.0 for x in fmt_scores]
    reasoning_scores = [x if x is not None else 0.0 for x in reasoning_scores]
    acc_scores = [x if x is not None else 0.0 for x in acc_scores]


    mean_fmt = sum(fmt_scores) / len(fmt_scores) * 100
    mean_reason = sum(reasoning_scores) / len(reasoning_scores) * 100
    mean_acc = sum(acc_scores) / len(acc_scores) * 100

    print("\n" + "=" * 50)
    print("📈 最终定量评测报告 (基于 TRL 奖励函数)")
    print(f"模型路径: {MODEL_PATH}")
    print(f"测试集大小: {len(dataset)}")
    print("-" * 50)
    print(f"🔹 格式遵循度 (Format Score)     : {mean_fmt:.2f} 分")
    print(f"🔹 纯答案准确性 (Accuracy Score) : {mean_acc:.2f} 分")
    print(f"🔹 推理链准确性 (Reasoning Score): {mean_reason:.2f} 分")
    print("=" * 50)

if __name__ == "__main__":
    main()