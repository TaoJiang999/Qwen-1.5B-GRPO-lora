"""
环境:  torch 2.8.0 / vLLM 0.10.2 / TRL 0.29.1 / transformers 4.56.2
硬件:  双卡 RTX 4090 48G
"""

import os, time, logging, requests
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, TrainerCallback, TrainerState, TrainerControl
import concurrent.futures
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import (
    accuracy_reward,
    think_format_reward,
    reasoning_accuracy_reward,
    get_soft_overlong_punishment,
)

#路径配置
POLICY_MODEL = "/home/waas/LLM/cache/models--DEAR-Tao--Qwen2.5-1.5B-Instruct-SFT-lora/snapshots/0752c79cb2e93b05a7998cbe49491e5418d51f90"
REWARD_MODEL = "/home/waas/LLM/cache/models--Skywork--Skywork-Reward-Llama-3.1-8B-v0.2/snapshots/d4117fbfd81b72f41b96341238baa1e3e90a4ce1"
TRAIN_DATA   = "/home/waas/LLM/cache/datasets--DEAR-Tao--DeepMath-55k/snapshots/c266e59aef9895488df9a4ace14b07319ed8b474"
EVAL_DATA    = TRAIN_DATA
OUTPUT_DIR   = "./grpo_output"


#  服务端口
REWARD_VLLM_PORT = 8001
REWARD_VLLM_URL  = f"http://127.0.0.1:{REWARD_VLLM_PORT}"

#  数据集字段与 split
PROMPT_FIELD   = "prompt"
SOLUTION_FIELD = "solution"
TRAIN_SPLIT    = "train"
EVAL_SPLIT     = "test"
# EVAL_SAMPLES   = 10   # 每次评估的样本数，-1 = 全量


# 训练超参
LEARNING_RATE = 1e-5
BATCH_SIZE    = 2     # per_device_train_batch_size
GRAD_ACCUM    = 8     # 等效全局 batch = 16
EPOCHS        = 1
WARMUP_RATIO  = 0.05
BF16          = True


# GRPO 超参
NUM_GENERATIONS = 4
MAX_NEW_TOKENS  = 4096
MAX_PROMPT_LEN  = 2048
TEMPERATURE     = 1
TOP_P           = 0.95
KL_COEFF        = 0.04   #

# vLLM 策略生成：GRPOTrainer 内部 fork，分配 GPU0 的一部分显存
# 48G 显存：~55% 给训练（1.5B模型+梯度+Adam），~20% 给内部 vLLM 生成
VLLM_GPU_MEM_UTIL = 0.2


# LoRA 配置

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


#  奖励权重（总和 = 1.0）
W_FORMAT   = 0.15   # 稍微提高，初期诱导它输出 <think>
W_REASON   = 0.40   # 依然是主力，但给惩罚项腾出空间
W_ACC      = 0.10   # 辅助得分
W_OVERLONG = 0.20   # 核心修改：重拳出击，严惩无限啰嗦！
W_RM       = 0.15   # 降低一点权重，防止模型靠谄媚 RM 骗分

OVERLONG_SOFT = 2048
OVERLONG_HARD = 4096   # == MAX_NEW_TOKENS


# 保存 / 日志
SAVE_STEPS = 10
LOG_STEPS  = 5
# EVAL_STEPS = 1
SEED       = 42
REPORT_TO  = "none"   # "wandb" / "tensorboard" / "none"

# ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("grpo")



# 奖励模型打分（调用外部 vLLM /v1/score）
reward_tokrnizer = AutoTokenizer.from_pretrained(REWARD_MODEL)

def call_reward_model(prompts: list[list[dict[str, str]]],responses: list[list[dict[str, str]]]) -> list[float]:
    """
    调用 Skywork 奖励模型服务（运行在 GPU1，端口 8001）。
    /v1/score 返回分类头原始 logit，sigmoid → [0,1] 概率分。
    兜底 0.5 = sigmoid(0)，语义为"中立/不确定"。
    """
    test_list = [ (p[-1].get('content'),c[-1].get('content')) for p,c in zip(prompts,responses)]
    reward_list = []
    for p,c in test_list:
        try:
            resp = requests.post(
                f"{REWARD_VLLM_URL}/v1/score",
                json={"model": REWARD_MODEL, "text_1": p, "text_2": c},
                timeout=15,
            )
            resp.raise_for_status()
            logit = resp.json()["data"][0]["score"]
            reward_list.append(float(torch.sigmoid(torch.tensor(logit)).item()))
        except Exception as e:
            log.warning(f"[RM] 打分失败: {e}，本批次置 0.0")
            reward_list.append(0.0)

    return reward_list



# 组合奖励函数
soft_overlong_punishment = get_soft_overlong_punishment(
    max_completion_len=OVERLONG_HARD, 
    soft_punish_cache=OVERLONG_SOFT
)
def combined_reward(
    prompts:     list[list[dict[str, str]]],
    completions: list[list[dict[str, str]]],
    solution:    list[str],   # GRPOTrainer 从数据集 SOLUTION_FIELD 自动注入
    completion_ids: list[list[int]],
    **kwargs,
) -> list[float]:
    fmt      = think_format_reward(prompts=prompts, completions=completions)
    reason   = reasoning_accuracy_reward(prompts=prompts, completions=completions, solution=solution)
    acc      = accuracy_reward(prompts=prompts, completions=completions, solution=solution)
    overlong = soft_overlong_punishment(completion_ids)
    rm       = call_reward_model(prompts, completions)

    fmt    = [x if x is not None else 0.0 for x in fmt]
    reason = [x if x is not None else 0.0 for x in reason]
    acc    = [x if x is not None else 0.0 for x in acc]
    overlong = [x if x is not None else 0.0 for x in overlong]
    rm     = [x if x is not None else 0.0 for x in rm]

    rewards = [
        W_FORMAT * fmt[i] + W_REASON * reason[i] + W_ACC * acc[i]
        + W_OVERLONG * overlong[i] + W_RM * rm[i]
        for i in range(len(completions))
    ]
    log.info(
        f"  fmt={sum(fmt)/len(fmt):.2f}  reason={sum(reason)/len(reason):.2f}  "
        f"acc={sum(acc)/len(acc):.2f}  overlong={sum(overlong)/len(overlong):.2f}  "
        f"rm={sum(rm)/len(rm):.2f}  → mean={sum(rewards)/len(rewards):.3f}"
    )
    return rewards



#  测试集评估回调
class EvalCallback(TrainerCallback):
    """每隔 EVAL_STEPS 步在测试集上计算四项指标并打印样例。"""

    def __init__(self, eval_ds, tokenizer):
        self.eval_ds   = eval_ds
        self.tokenizer = tokenizer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl,
                    model=None, **kwargs):
        if state.global_step == 0 or state.global_step % EVAL_STEPS != 0:
            return
        if model is None:
            return

        samples   = (self.eval_ds.select(range(min(EVAL_SAMPLES, len(self.eval_ds))))
                     if EVAL_SAMPLES > 0 else self.eval_ds)
        prompts   = samples[PROMPT_FIELD]
        solutions = samples[SOLUTION_FIELD]

        model.eval()
        completions = []
        with torch.no_grad():
            for i in range(0, len(prompts), NUM_GENERATIONS):
                batch_raw_prompts = prompts[i: i + NUM_GENERATIONS]
                batch_str_prompts = [
                    self.tokenizer.apply_chat_template(
                        p, 
                        tokenize=False, 
                        add_generation_prompt=True
                    ) for p in batch_raw_prompts
                ]
                enc = self.tokenizer(
                    batch_str_prompts, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True, 
                    max_length=MAX_PROMPT_LEN,
                ).to(model.device)

                ids = model.generate(
                    **enc, 
                    max_new_tokens=MAX_NEW_TOKENS, 
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                for seq in ids:
                    gen = seq[enc["input_ids"].shape[1]:]
                    completions.append(self.tokenizer.decode(gen, skip_special_tokens=True))

        model.train()

        fmt_s    = think_format_reward(prompts=prompts, completions=completions)
        acc_s    = accuracy_reward(prompts=prompts, completions=completions, solution=solutions)
        reason_s = reasoning_accuracy_reward(prompts=prompts, completions=completions, solution=solutions)

        fmt_s = [x if x is not None else 0.0 for x in fmt_s]
        acc_s = [x if x is not None else 0.0 for x in acc_s]
        reason_s = [x if x is not None else 0.0 for x in reason_s]

        avg_len  = sum(len(self.tokenizer.encode(c)) for c in completions) / len(completions)

        log.info(f"\n{'─'*54}  [Eval] step={state.global_step}")
        log.info(f"  answer_acc    {sum(acc_s)/len(acc_s):.4f}  ({sum(acc_s)/len(acc_s)*100:.1f}%)")
        log.info(f"  format_acc    {sum(fmt_s)/len(fmt_s):.4f}  ({sum(fmt_s)/len(fmt_s)*100:.1f}%)")
        log.info(f"  reasoning_acc {sum(reason_s)/len(reason_s):.4f}  ({sum(reason_s)/len(reason_s)*100:.1f}%)")
        log.info(f"  avg_length    {avg_len:.1f} tokens  (样本数={len(completions)})")

        import random
        for idx in random.sample(range(len(completions)), min(2, len(completions))):
            log.info(f"\n  [样例 {idx}]  solution: {solutions[idx][:80]}")
            log.info(f"  generated:  {completions[idx][:200]}")
        log.info(f"{'─'*54}\n")

        state.log_history.append({
            "step": state.global_step,
            "eval/answer_acc":    sum(acc_s)    / len(acc_s),
            "eval/format_acc":    sum(fmt_s)    / len(fmt_s),
            "eval/reasoning_acc": sum(reason_s) / len(reason_s),
            "eval/avg_length":    avg_len,
        })



#  主函数

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 确认奖励模型服务已就绪（需在 GPU1 手动启动）
    log.info("═" * 54)
    log.info("1/3  检查奖励模型服务")
    log.info(f"     期望地址: {REWARD_VLLM_URL}  (GPU1 手动启动)")
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if requests.get(f"{REWARD_VLLM_URL}/health", timeout=2).status_code == 200:
                log.info("奖励模型服务已就绪")
                break
        except Exception:
            pass
        time.sleep(3)
    else:
        raise RuntimeError(
            "奖励模型服务未响应，请先在 GPU1 启动：\n"
            f"  CUDA_VISIBLE_DEVICES=1 vllm serve {REWARD_MODEL} "
            f"--task auto --port {REWARD_VLLM_PORT} --max-model-len 20480"
        )

    # 2. 加载数据集与 tokenizer
    log.info("═" * 54)
    log.info("2/3  加载数据集")

    def load(path, split):
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            return load_dataset("json" if ext in (".json", ".jsonl") else "csv",
                                data_files=path, split=split)
        return load_dataset(path, split=split)

    train_ds = load(TRAIN_DATA, TRAIN_SPLIT)
    # eval_ds  = load(EVAL_DATA,  EVAL_SPLIT)
    # log.info(f"  训练: {len(train_ds)} 条  |  测试: {len(eval_ds)} 条")
    log.info(f"  训练: {len(train_ds)} 条")

    tokenizer = AutoTokenizer.from_pretrained(
        POLICY_MODEL, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 配置并启动训练
    log.info("═" * 54)
    log.info("3/3  开始 GRPO 训练")
    log.info(f"  策略模型  : {POLICY_MODEL}")
    log.info(f"  奖励模型  : {REWARD_MODEL}")
    log.info(f"  batch     : {BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}  G={NUM_GENERATIONS}")
    log.info(f"  LoRA      : r={LORA_R}  α={LORA_ALPHA}")
    log.info(f"  奖励权重  : fmt={W_FORMAT} reason={W_REASON} acc={W_ACC} overlong={W_OVERLONG} rm={W_RM}")
    log.info("═" * 54)

    grpo_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        # 训练
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        bf16=BF16,
        seed=SEED,
        # GRPO
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_NEW_TOKENS,
        # max_prompt_length=MAX_PROMPT_LEN,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        beta=KL_COEFF,
        # vLLM 内部管理（GRPOTrainer 自动 fork，不需要外部手动启动）
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=VLLM_GPU_MEM_UTIL,
        # 速度优化
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # 保存 & 日志
        save_steps=SAVE_STEPS,
        save_total_limit=10,
        logging_steps=LOG_STEPS,
        report_to=REPORT_TO,
    )

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none", task_type=TaskType.CAUSAL_LM,
    )

    trainer = GRPOTrainer(
        model=POLICY_MODEL,
        args=grpo_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        reward_funcs=[combined_reward],
        peft_config=lora_cfg,
    )
    # trainer.add_callback(EvalCallback(eval_ds, tokenizer))

    log.info("GRPOTrainer 初始化完成，开始训练")
    trainer.train()

    # 保存 LoRA 适配器
    trainer.save_model(f"{OUTPUT_DIR}/final_lora")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_lora")
    log.info(f"LoRA 已保存: {OUTPUT_DIR}/final_lora")

    # 合并为完整权重（方便后续部署）
    trainer.model.merge_and_unload().save_pretrained(
        f"{OUTPUT_DIR}/merged", safe_serialization=True
    )
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged")
    log.info(f"合并模型已保存: {OUTPUT_DIR}/merged")


if __name__ == "__main__":
    main()