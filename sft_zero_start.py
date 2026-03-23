#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT + LoRA 微调脚本 —— Qwen/Qwen2.5-1.5B-Instruct
支持：单卡 / 单机多卡 / FSDP ZeRO-2

依赖:
    pip install trl peft transformers datasets accelerate bitsandbytes
"""
from dotenv import load_dotenv
load_dotenv()

import os
import logging

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ──────────────────────────────────────────────
# 日志
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)



# 1. 模型 & Tokenizer 参数
MODEL_NAME_OR_PATH = "/home/waas/LLM/cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TRUST_REMOTE_CODE   = True
ATTN_IMPLEMENTATION = "flash_attention_2"   # 改为 "eager" 可关闭 FA2
DTYPE               = "bfloat16"            # bf16 / fp16 / float32

# 是否使用 QLoRA（4-bit 量化）
USE_QLORA           = False
LOAD_IN_4BIT        = False                  # QLoRA 才生效



# 2. LoRA 参数
LORA_R              = 32
LORA_ALPHA          = 64
LORA_DROPOUT        = 0.05
LORA_BIAS           = "none"
# Qwen3 的注意力 / MLP 投影名称
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_TASK_TYPE      = TaskType.CAUSAL_LM



# 3. 数据集参数

DATASET_NAME        = "/home/waas/LLM/cache/datasets--DEAR-Tao--ultrachat-bespoke-mixed/snapshots/f7c2a2e5045c44a60bbabc72f45d55be91bca3a3"    # 替换为你的数据集
DATASET_CONFIG      = '10k'
DATASET_TRAIN_SPLIT = "train"
DATASET_TEST_SPLIT  = "test"
DATASET_STREAMING   = False
# 如果数据集字段名与默认不同，可在此指定
# DATASET_TEXT_FIELD  = "text"



# 4. 训练参数（SFTConfig 继承 TrainingArguments）
OUTPUT_DIR                  = "./output/Qwen2.5-1.5B-SFT-LoRA"
NUM_TRAIN_EPOCHS            = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE  = 1
GRADIENT_ACCUMULATION_STEPS = 16            # 等效全局 batch = GPU数 × 2 × 8
LEARNING_RATE               = 2e-4
LR_SCHEDULER_TYPE           = "cosine"
WARMUP_RATIO                = 0.05
WEIGHT_DECAY                = 0.01
MAX_GRAD_NORM               = 1.0

# 序列长度
MAX_SEQ_LENGTH              = 16384
PACKING                     = False         # 序列打包，提升吞吐

# 评估 & 保存
EVAL_STRATEGY               = "steps"
EVAL_STEPS                  = 50
SAVE_STRATEGY               = "steps"
SAVE_STEPS                  = 50
SAVE_TOTAL_LIMIT            = 10
LOAD_BEST_MODEL_AT_END      = True

# 精度
BF16                        = True         # A100/H100 推荐 bf16
FP16                        = False

# 日志
LOGGING_STEPS               = 10
REPORT_TO                   = "wandb"   # 改为 "wandb" 可用 W&B

# Hub 推送（可选）
PUSH_TO_HUB                 = False
HUB_MODEL_ID                = None          # e.g. "your-org/Qwen3-4B-SFT-LoRA"

# Dataloader
DATALOADER_NUM_WORKERS      = 1
DATALOADER_PREFETCH_FACTOR  = 2


# 辅助函数
def get_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16":     torch.bfloat16,
        "float16":  torch.float16,
        "fp16":     torch.float16,
        "float32":  torch.float32,
        "fp32":     torch.float32,
    }
    return mapping.get(dtype_str.lower(), torch.bfloat16)


def build_bnb_config() -> BitsAndBytesConfig | None:
    """QLoRA 量化配置，仅在 USE_QLORA=True 时返回非 None。"""
    if not USE_QLORA:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        target_modules=LORA_TARGET_MODULES,
        task_type=LORA_TASK_TYPE,
        # 让 LoRA 权重以 bf16 保存，节省显存
        inference_mode=False,
    )


def build_sft_config() -> SFTConfig:
    return SFTConfig(
        # ── 输出 ──
        output_dir=OUTPUT_DIR,
        # ── 轮次 & batch ──
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=500,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        # ── 优化器 ──
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        optim="adamw_torch_fused",   # 使用 fused AdamW，更快
        # ── 序列 ──
        max_length=MAX_SEQ_LENGTH,
        packing=PACKING,
        # ── 精度 ──
        bf16=BF16,
        fp16=FP16,
        # ── 评估 & 保存 ──
        eval_strategy=EVAL_STRATEGY,
        eval_steps=EVAL_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # ── 日志 ──
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        report_to=REPORT_TO,
        # ── Hub ──
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=HUB_MODEL_ID,
        # ── Dataloader ──
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_prefetch_factor=DATALOADER_PREFETCH_FACTOR,
        # ── 其他 ──
        remove_unused_columns=True,   # 保留所有数据列，防止 chat template 报错
        gradient_checkpointing=True,   # 节省显存
        gradient_checkpointing_kwargs={"use_reentrant": False},
        padding_free=False,
        # assistant_only_loss=True
    )



# 主流程
def main():
    logger.info("=" * 60)
    logger.info("SFT LoRA 微调 —— %s", MODEL_NAME_OR_PATH)
    logger.info("=" * 60)

    # ── Tokenizer ──
    logger.info("加载 tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        trust_remote_code=TRUST_REMOTE_CODE,
        padding_side="right",   # 训练时右填充
    )
    # Qwen3 的 pad token 通常与 eos token 相同，保险起见显式设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 模型 ──
    logger.info("加载模型 …")
    torch_dtype = get_torch_dtype(DTYPE)
    bnb_config  = build_bnb_config()

    model_kwargs = dict(
        pretrained_model_name_or_path=MODEL_NAME_OR_PATH,
        trust_remote_code=TRUST_REMOTE_CODE,
        attn_implementation=ATTN_IMPLEMENTATION,
        torch_dtype=torch_dtype,
    )
    if bnb_config is not None:
        # QLoRA：量化加载，device_map 交给 accelerate 管理
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", 0))}

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.config.use_cache = False          # 训练时关闭 KV cache
    logger.info("模型参数量：%.2fB", sum(p.numel() for p in model.parameters()) / 1e9)

    # ── 数据集 ──
    logger.info("加载数据集 %s …", DATASET_NAME)
    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        streaming=DATASET_STREAMING,
    )
    train_dataset = dataset[DATASET_TRAIN_SPLIT]
    eval_dataset  = (
        dataset[DATASET_TEST_SPLIT]
        if EVAL_STRATEGY != "no" and DATASET_TEST_SPLIT in dataset
        else None
    )
    logger.info("训练集大小：%d", len(train_dataset))
    if eval_dataset is not None:
        logger.info("验证集大小：%d", len(eval_dataset))

    # ── 配置 ──
    lora_config = build_lora_config()
    sft_config  = build_sft_config()
    logger.info("LoRA 配置：r=%d, alpha=%d, target_modules=%s",
                LORA_R, LORA_ALPHA, LORA_TARGET_MODULES)

    from trl import DataCollatorForCompletionOnlyLM

    # 在 main() 里
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
    )

    # ── Trainer ──
    logger.info("初始化 SFTTrainer …")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        data_collator=collator,
        
    )

    # ── 训练 ──
    logger.info("开始训练 …")
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    # ── 保存 ──
    logger.info("保存 LoRA 适配器到 %s", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    trainer.save_state()

    # ── 推送 Hub（可选）──
    if PUSH_TO_HUB:
        logger.info("推送到 HuggingFace Hub …")
        trainer.push_to_hub(dataset_name=DATASET_NAME,token=os.getenv("HF_TOKEN"))

    logger.info("训练完成！")


if __name__ == "__main__":
    main()