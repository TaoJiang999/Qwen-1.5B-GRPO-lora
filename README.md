<div align="center">

# 🚀 基于 Qwen2.5 与 GRPO 的模型推理强化学习项目

**本项目基于 Qwen2.5-1.5B 轻量级大语言模型，使用 GRPO 算法进行强化学习微调，旨在通过 SFT 与 RL 阶段的训练，提升模型在复杂数学逻辑和长思考（Think）上的推理能力。**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.8.0-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.9-green?logo=nvidia" alt="CUDA">
  <img src="https://img.shields.io/badge/Flash%20Attention-2.6.3-orange" alt="Flash Attention">
  <img src="https://img.shields.io/badge/Transformers-4.56.2-blueviolet" alt="Transformers">
  <img src="https://img.shields.io/badge/TRL-0.29.1-success" alt="TRL">
  <img src="https://img.shields.io/badge/vLLM-0.10.2-8A2BE2" alt="vLLM">
  <img src="https://img.shields.io/badge/Model-Qwen2.5-purple?logo=alibabacloud" alt="Qwen">
</p>

</div>

## 📖 项目简述 (Project Overview)

本项目通过 **监督微调（SFT）冷启动** 和 **强化学习（GRPO）策略优化** 两阶段训练，旨在提升基础大模型在数学推理任务上的表现。工作流涵盖了高质量开源数据的引入与配置、多卡 DDP 环境部署、基于 vLLM 独立提供奖励反馈的 RL 训练，以及最终的定量评估与模型合并。

**项目核心亮点**：
- **完整的强化学习微调闭环**：设计了定制化复合奖励函数（包含格式约束、推理链准确率判定以及针对过长生成的软性惩罚机制），引导模型以 `<think>` 的结构化格式输出推理过程。
- **工程训练架构的整合**：采用 `vLLM` 后端独立运行奖励模型以提高打分的并发效率，在主进程中利用 `TRL` 库进行 GRPO 策略计算，有效缓解了 RLHF 训练中普遍存在的显存与计算阻塞问题。
- **客观的指标提升验证**：基于 GSM8K 测试集的量化评估显示，Base 模型的推理准确度从 2.35% 被有效提升至 42.0%，且最终的格式遵循度达到 50.8%，直观验证了完整数据与训练流水线的设计成效。

---

## 🔗 模型与数据引流 (Models & Datasets)

项目中涉及的所有开源数据与模型版本引用如下，感谢开源社区的贡献：

### 📦 模型资源 
- **Base 模型**: [HuggingFace: Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Reward 模型**: [HuggingFace: Skywork/Skywork-Reward-Llama-3.1-8B-v0.2](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2)
- **产出 SFT 模型**: [HuggingFace: DEAR-Tao/Qwen2.5-1.5B-Instruct-SFT-lora](https://huggingface.co/DEAR-Tao/Qwen2.5-1.5B-Instruct-SFT-lora) (学习基础对话与初步思考框架)
- **产出 GRPO 模型**: [HuggingFace: DEAR-Tao/Qwen2.5-1.5B-Instruct-GRPO-think-lora](https://huggingface.co/DEAR-Tao/Qwen2.5-1.5B-Instruct-GRPO-think-lora) (强化学习硬核推理能力)

### 📊 数据集
- **SFT 冷启动数据**: [HuggingFace: DEAR-Tao/ultrachat-bespoke-mixed](https://huggingface.co/datasets/DEAR-Tao/ultrachat-bespoke-mixed) *(混合 `ultrachat_200k` 30% + `Bespoke-Stratos-17k` 70%)*
- **GRPO 强化推理数据**: [HuggingFace: DEAR-Tao/DeepMath-55k](https://huggingface.co/datasets/DEAR-Tao/DeepMath-55k) *(筛选自 `trl-lib/DeepMath-103K`)*
- **定量测试数据集**: [HuggingFace: DEAR-Tao/gsm8k-test](https://huggingface.co/datasets/DEAR-Tao/gsm8k-test) *(源自 OpenAI 原版测试集)*

---

## 🚀 性能结果评估 (Evaluation Results)

项目使用 vLLM 以极高吞吐完成了全量 GSM8K 评测，各项核心奖励分数指标的变化显著，完美证明了 GRPO 的算法有效性：

| 训练阶段 | 格式遵循度 (Format Score) | 纯答案准确性 (Accuracy Score) | 推理链准确性 (Reasoning Score) |
| :--- | :---: | :---: | :---: |
| 🪨 **Base 原生模型** (1.5B Instruct) | 3.26 分 | 30.40 分 | 2.35 分 |
| 🔨 **SFT 冷启动微调后** | 39.81 分 | 35.86 分 | 35.86 分 |
| 🧠 **GRPO 强化学习后** *(最终产出)* | **50.80 分** | **42.00 分** | **42.00 分** |

*评估条件：通过定量 TRL Custom Reward 评测函数，检测 `<think>` 标签的结构完整性 与 最终输出答案的准确性*

---

## 📂 项目结构 (Repository Structure)

```text
GRPO/
 ├── .env                      # 环境变量配置文件 (存放 HF_TOKEN 等)
 ├── .gitignore                # Git 忽略规则
 ├── requirements.txt          # Python 依赖包清单
 ├── multi_gpu_ddp.yaml        # SFT 多卡分布式环境配置文件（Accelerate 驱动）
 ├── sft_zero_start.py         # 阶段一：SFT 冷启动、格式灌输与能力打底核心训练脚本
 ├── grpo_train.py             # 阶段二：GRPO 强化学习核心脚本 (主入口 + 复合奖励矩阵构建)
 ├── eval_script.py            # 推理阶段：通过 vLLM 对测试集进行自动化定量指标对齐和测试
 └── merge_weight.py           # 推送阶段：将训练得到的 LoRA Adapter 合并到主模型并推向 Hub
```

---

## 🛠️ 快速复现指南 (Quick Start)

### 1. 克隆项目与依赖安装

建议在您的 Ubuntu 22.04 + 双卡 4090 (48G) 机器上执行：

```bash
# 克隆仓库
git clone https://github.com/TaoJiang999/Qwen-1.5B-GRPO-lora.git
cd Qwen-1.5B-GRPO-lora

# 创建并激活 Python 3.12 虚拟环境 (推荐使用 conda)
conda create -n grpo_env python=3.12 -y
conda activate grpo_env

# 安装依赖
pip install -r requirements.txt
```

### 2. 核心凭证配置
使用前，您需要将 HuggingFace 访问令牌配置在本地环境变量中。打开或新建 `.env` 文件，写入并替换下方的 `your_token`：

```env
# .env 配置文件
HF_TOKEN=your_token
```

### 3. Stage 1：SFT 冷启动训练
使用 Accelerate 结合 `multi_gpu_ddp.yaml` 分布式配置全面开启监督微调：

```bash
nohup accelerate launch --config_file multi_gpu_ddp.yaml sft_zero_start.py > sft.log 2>&1 &
```

### 4. Stage 2：GRPO 强化学习
大模型的强化学习被拆分为**奖励服务分发网络**以及**主策略进化网络**：

- **第一步：启动奖励判定模型 (GPU 1)**  
  让 Skywork 奖励模型挂载进 vLLM 后端常驻，处理打分请求。
  ```bash
  CUDA_VISIBLE_DEVICES=1 nohup vllm serve <your_reward_model_path> \
      --task auto \
      --port 8001 \
      --max-model-len 20480 \
      --gpu-memory-utilization 0.9 > reward.log 2>&1 &
  ```

- **第二步：启动 GRPO 策略进化训练 (GPU 0)**  
  等待上方服务端口映射成功后，正式开启 Actor 角色迭代强化！
  ```bash
  CUDA_VISIBLE_DEVICES=0 nohup python grpo_train.py > policy.log 2>&1 &
  ```

### 5. 效果测评与模型部署
验证最终成果：
```bash
python eval_script.py
```

将提纯的思维能力（LoRA Checkpoint）融入骨架（Base Model）中：
```bash
python merge_weight.py
```

---

*By [DEAR-Tao] ✨*
