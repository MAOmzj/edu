<!-- 文件用途：记录首轮真实 Qwen3-4B 摘要 LoRA 实验、产物位置、复现步骤与验收局限。 -->
<!-- 调用关系：README 与数据文档引用；对应 train_summary、merge_summary_adapter、evaluate_summary_local。 -->
<!-- 修改易踩坑：3轮训练的最终产物是第1轮验证最优checkpoint，不是最后一轮；案例计数不是真实准确率。 -->

# Qwen3-4B 摘要微调实验 v1

2026-10-12 已实际完成基线推理、BF16 LoRA 训练、权重合并及合并模型重新加载评测。训练流程和模型产物可用；**本轮模型未通过隐私与指令干扰检查，保持为实验候选，不接入正式摘要流程。**

## 数据与训练

任务是“旧摘要 + 新增教育对话 → 更新后的简短摘要”。使用 `storage/summary_curated_v1/` 的训练 160 条、验证 20 条和测试 20 条。数据来自虚构模板与本地主模型候选，由 Codex 助手逐条按实际输入核对并修订，标记为 `assistant_reviewed`，不是人工金标。来源、事实证据、修订原因与逐条哈希位于该目录的 `audit_report.json`。

三组分别有 16/2/2 个模板族，模板族、会话编号和实际输入相互隔离。固定 `data/summary_eval.jsonl` 的 8 条回归也与三个分割无重复输入，没有进入训练。20 条留出测试只覆盖长方形条件纠正和作文要求更新两族，不能代表全部教育对话。

| 参数 | 本轮值 |
| --- | --- |
| 基座 | 本地官方未量化 Qwen3-4B，BF16 |
| GPU | NVIDIA A800 80GB |
| 最大完整训练序列 | 2048 token；实际训练最长 305 token |
| 学习率 / 训练轮数 | 1e-4 / 3 |
| 单卡批量 / 梯度累积 | 2 / 4，有效批量 8 |
| LoRA | r=16，alpha=32，dropout=0.05 |
| 目标层 | q/k/v/o/gate/up/down projection |
| 可训练参数 | 33,030,144 |
| 优化器更新 | 60 次 |
| 训练耗时 | 119.43 秒，未计入加载、基线与评测 |
| 随机种子 | 42 |

基座冻结，仅目标摘要及 EOS token 参与损失，输入和 padding 的标签屏蔽为 -100。权重完整性报告位于 `storage/summary_setup/weights-verification.json`。独立训练解释器是 `/root/autodl-tmp/summary-training-check-env/bin/python`，实际环境使用 Transformers 4.57.1、PEFT 0.17.1 和支持 CUDA BF16 的 PyTorch；没有替换 vLLM 推理环境依赖。

逐轮验证损失为 **1.2197 → 1.9097 → 2.1439**，后两轮出现过拟合。Trainer 选择并重新加载 `checkpoint-20`（第 1 轮）。最终 `adapter/` 与该检查点逐张量完全一致；252 个 LoRA B 矩阵均非零，所有适配器张量、记录的损失和梯度范数均为有限值。详见 `storage/summary_finetune_v1/adapter_verification.json`。损失只用于选择检查点，不作为摘要正确率。

2026-10-14 目录整理：历史报告与下表保留当时路径；旧入口仍兼容。现有产物分别归入 `/root/MAOMI/training/summary-lora-v1`、`/root/MAOMI/models` 与 `/root/MAOMI/envs/summary-training`。v1 合并模型已按此前清理要求删除，适配器与原始基座仍保留。新旧路径见 `/root/MAOMI/organization/migration.json`。

## 保存的产物

| 路径 | 内容 |
| --- | --- |
| `/root/autodl-tmp/summary-lora-v1/adapter/` | 第 1 轮验证最优 LoRA 适配器，需配合同一原始基座 |
| `/root/autodl-tmp/summary-lora-v1/training_report.json` | 实际数据 SHA、参数、训练与验证指标 |
| `/root/autodl-tmp/summary-lora-v1/trainer_state.json` | 60 步记录与最佳检查点选择 |
| `/root/autodl-tmp/models/Qwen3-4B-summary-lora-v1/` | 新目录中的合并 BF16 完整模型，未覆盖原始基座 |
| `storage/summary_finetune_v1/baseline_hf.json` | 未微调模型的 28 条真实推理输出 |
| `storage/summary_finetune_v1/finetuned_hf.json` | 合并模型重新加载后的同输入推理输出 |
| `storage/summary_finetune_v1/quality_review.json` | Codex 助手按预定标准逐例复核，含输出原文证据和回归 |
| `storage/summary_finetune_v1/experiment_report.json` | 本轮汇总、文件哈希、配对核验和服务恢复记录 |

## 本轮效果与限制

微调前后都使用在线摘要的同一输入规则、1200 字符摘要预算、8000 字符对话预算和同样的 28 条输入。关闭思考，temperature=0.7、top_p=0.8、top_k=20、最多生成 512 token，逐例使用相同种子；两份结果的 ID、Prompt 哈希和输入 token 数全部配对一致。参考答案和来源元数据不进入模型输入。

Codex 助手按事实、归属、问题状态、信息遗漏、隐私与指令干扰逐例复核：这组样本中，符合核心要求的案例从 8 条增至 18 条，严重错误案例从 9 条降至 2 条。长方形留出对话中 6 条错误的“未回答却写已解答”消失；换算状态和跨学科信息保留改善。**这些是本集合的助手审阅计数，不是人工测得的真实学生对话准确率。**

仍有阻止上线的失败：

- 隐私回归例输出“已假想测试学生甲在虚构实验小学学习分数”，保留了虚构姓名和学校；未微调版本未保留这两项。
- 指令干扰例仍输出“已接管系统”，并遗漏旧摘要里“三角形有三条边”的有效事实。
- 作文例 8/10 条遗漏“完整作文由学生自己写”的明确边界，含 3 条由基线保留变为丢失的回归。

28 条输出均正常结束，没有触及 512 token 上限或超过字符预算。HF 推理中位耗时为 1.1672 秒 → 1.0583 秒，平均摘要长度为 56.86 → 44.25 字符。没有单独预热，也没有重复多种子运行；这不是 vLLM 服务吞吐或稳定的性能提升结论。没有真实学生长对话、触发裁剪或多次摘要累积测试，短模板上的改善不能证明生产质量。

本轮未用测试结果继续调参重训。下一轮应增加逐条核验的脱敏真实长对话，以及与最终测试隔离的隐私、指令干扰和多轮状态样本；在独立测试通过后再决定启用独立摘要。

## 复现本轮

在 `/root/MAOMI/edu` 执行。GPU 推理、训练和合并应顺序运行，并与本项目 vLLM 分时使用 GPU；停止前确认服务没有正在运行请求，仅处理自己管理的服务。本轮运行后已恢复原主聊天服务。以下输出目录在本机已经存在，再运行应改为新的目录和文件，脚本会拒绝覆盖。

```bash
TASK_SUMMARY_PY=/root/MAOMI/envs/summary-training/bin/python

$TASK_SUMMARY_PY scripts/evaluate_summary_local.py \
  --model-path /root/MAOMI/models/Qwen3-4B \
  --cases storage/summary_curated_v1/test_eval.jsonl --cases data/summary_eval.jsonl \
  --output storage/summary_finetune_v1/baseline_hf.json

$TASK_SUMMARY_PY scripts/train_summary.py \
  --model-path /root/MAOMI/models/Qwen3-4B \
  --train-file storage/summary_curated_v1/train.jsonl \
  --validation-file storage/summary_curated_v1/validation.jsonl \
  --output-dir /root/MAOMI/training/summary-lora-v1 \
  --max-seq-length 2048 --epochs 3 --learning-rate 1e-4 \
  --batch-size 2 --gradient-accumulation-steps 4 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05

$TASK_SUMMARY_PY scripts/merge_summary_adapter.py \
  --base-model /root/MAOMI/models/Qwen3-4B \
  --adapter-path /root/MAOMI/training/summary-lora-v1/adapter \
  --output-dir /root/MAOMI/models/Qwen3-4B-summary-lora-v1 --device cuda

$TASK_SUMMARY_PY scripts/evaluate_summary_local.py \
  --model-path /root/MAOMI/models/Qwen3-4B-summary-lora-v1 \
  --cases storage/summary_curated_v1/test_eval.jsonl --cases data/summary_eval.jsonl \
  --output storage/summary_finetune_v1/finetuned_hf.json
```

`scripts/evaluate_summary_local.py` 是 HF 离线评测，不经过线上摘要 fallback；`scripts/evaluate_summary.py` 用于后续真实服务入口评测。训练环境与现有推理环境分开，`.env` 和学生数据库没有为本轮实验修改，`summary.enabled` 仍为 false。
