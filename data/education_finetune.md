<!-- 文件用途：说明小学多学科教学回答的数据格式、检查、训练与验收步骤。 -->
<!-- 调用关系：README 引用；按本文运行 run_education_finetune 与 train_education。 -->
<!-- 修改易踩坑：完整27B位于models/Qwen3.8-27B，需按配置加载；教学合成候选非人工金标，已有训练输出拒绝覆盖。 -->
# Qwen3.8-27B 小学教学回答微调

本入口训练语文、数学、英语与科学的教学回答风格：适合儿童的表达、分步讲解、条件缺失时追问、温和纠错，以及不捏造教材、工具操作和检索来源。它与4B对话摘要训练分开，摘要数据不能直接代替教学回答数据。

配置位于 `config/education_finetune.yml`。当前完整官方未量化基座的配置、分词器和18个权重分片已统一整理到 `/root/MAOMI/models/Qwen3.8-27B/`，`model_path` 已指向该目录；无需重复下载。独立解释器为 `/root/MAOMI/envs/education-training/bin/python`。现有 `Qwen3.8-27B-AWQ-INT4` 是compressed-tensors推理量化权重，此训练入口拒绝接受。Qwen3.8使用qwen3_5架构，但不能用Qwen3.5模型权重当作同一个基座；配置结构检查本身不替代官方来源和下载完整性核验。

现有训练环境已安装并验证：Transformers 5.17.0、PEFT 0.21.2、Accelerate 1.15.0、bitsandbytes 0.50.2、PyTorch 2.13.0+cu130。为节省磁盘，环境通过 `.pth` 读取 MAOMI 的 vLLM 环境内现有 CUDA PyTorch，新增训练包只装在教育环境中；不会修改推理环境。环境布局与重新建立方法见 `../../envs/README.md`。

早期使用随机初始化的极小64层 qwen3_5 模型完成2步CPU训练、适配器保存与重新加载对比、基座冻结核验，并通过小模型CUDA NF4反向传播测试；该报告为 `/root/MAOMI/organization/education-environment-smoke.json`。之后另用真实27B完成单tensor的一次LoRA更新，见文末入口及 `/root/MAOMI/organization/education-tensor-test.json`；小模型报告与单批结果都不能代表完整数据训练速度或教学质量。当前使用Gated DeltaNet的PyTorch参考实现，正式27B训练可能较慢。

官方未量化权重总计55,563,006,776字节（约51.7 GiB）。2026-10-17检查确认18个本地分片大小与固定版本记录相符，数据盘已扩容，检查时约有28 GiB可用。`--check` 与 `--dry-run` 均已通过，准备检查与权重核验见 `/root/MAOMI/organization/education-training-readiness.json`；当前没有开始真实27B训练。官方来源见 [Qwen3.8-27B模型页](https://huggingface.co/Qwen/Qwen3.8-27B)，NF4训练流程参照 [PEFT量化文档](https://huggingface.co/docs/peft/main/developer_guides/quantization)。

先检查目前缺少哪些文件，不会下载、训练或加载GPU：

```bash
cd /root/MAOMI/edu
.venv/bin/python run_education_finetune.py --check
```

当前教学数据版本为 `/root/MAOMI/training/education-dataset-v2/`：训练2500条（语文、数学、英语、科学各625条），验证100条、测试100条。原v1包含1000条训练，新增批次`education-addition-v1`含1500条训练；合并版本保留原验证和测试文件。生成、复核与修订方法见[教学数据扩充](education_generation.md)，各目录的`dataset_report.json`记录数量和来源校验。

这些是本地27B教师生成并经AI抽查修订的合成教学候选，`review_status`保留`pending`；这不等于全批人工审核，也不代表真实课堂正确率。每条JSONL至少包含非空 `id`、`group_id` 与 `messages`；末条必须为assistant最终教学回答，前面的user/assistant交替，可在首条添加system。没有system时自动加入配置中的小学教学指令。`subject`、`source`、`review_status`可选；复核状态只统计，不作为审批门槛。

`group_id`用于完整会话或题目族分组，训练/验证不能共享group、id或相同输入；同题只换数字也应归同组。另留一份测试集评估最终效果。数据可来自真实且已适当处理的教学对话、人工编写或教师模型生成，建议抽检事实、年级适合度、纠错与追问行为，避免泄漏学生隐私。示例仅有6条，用于说明格式，不是正式训练数据，不代表足够训练质量。

检查演示格式，不需要模型或训练环境：

```bash
.venv/bin/python scripts/train_education.py --validate-data-only \
  --train-file data/education_train.example.jsonl \
  --validation-file data/education_validation.example.jsonl
```

演示文件的`.example.jsonl`命名与`dataset_kind: example`会阻止正式训练。自行复制示例并改标签也不能使其变成充分的正式数据。

准备完整基座、正式数据及独立环境后，检查实际token与loss监督边界：

```bash
.venv/bin/python run_education_finetune.py --dry-run
```

dry-run加载本地tokenizer，并检查已安装Transformers的训练参数签名，不构造GPU训练参数对象、不加载GPU模型；超长样本拒绝截断。提示、历史assistant回答和padding的labels全部为-100，只有最后的assistant回答与EOS参与loss。训练使用`enable_thinking=False`，不训练内部推理段，也不训练工具调用或审核JSON。

项目配置仍使用`warmup_ratio: 0.05`；`build_training_arguments`将它传为Transformers 5.17支持的`warmup_steps=0.05`，浮点值表示前5%的更新步用于预热。正式训练也在加载27B前构造并校验训练参数，避免接口错误等到权重加载后才暴露。

正式训练前停止项目GPU服务，确认没有其它任务占用GPU，再运行：

```bash
.venv/bin/python scripts/serve.py stop all
.venv/bin/python run_education_finetune.py
```

启动器不自动停止或恢复其它服务。默认NF4 QLoRA在加载完整未量化底座时量化，BF16计算，r16/alpha32，batch1、梯度累积8、序列2048、学习率5e-5、1轮；参数是起点，需根据验证集调整。`method: lora`改用BF16 LoRA，显存占用明显增加。两者仅训练64个语言层的注意力与MLP适配器，视觉和MTP冻结。GPU可用性、库兼容性和峰值显存必须在正式训练时验证；文件检查通过不表示训练已验证。

输出 `/root/MAOMI/training/education-lora-v1/adapter` 和 `training_report.json`；已有输出目录拒绝覆盖。训练恢复验证loss最低的epoch检查点，保留原始底座，不自动合并、不切换线上部署。训练结束后可自行重新启动 `scripts/serve.py all -d`。

验证loss仅用于选择检查点，不等于教学正确率。与未微调基线用同一组未训练问题比较事实正确性、教学风格、缺条件追问与安全边界；主模型被四个子Agent共用，部署前还需要评估工具调用、审核JSON与记忆提取是否退化。普通教学SFT不能保证这些能力保持不变。

## 用一条tensor测试微调

入口为 `scripts/test_education_tensor.py`。先只查看张量，不加载27B：

```bash
cd /root/MAOMI/edu
/root/MAOMI/envs/education-training/bin/python scripts/test_education_tensor.py --prepare-only
```

默认样本是“2加3等于几？”与参考回答“2加3等于5。”。`encode_record`采用正式训练的同一套指令和监督边界，再用`torch.tensor([values], dtype=torch.long)`得到一批一条样本的`input_ids`、`attention_mask`、`labels`；提问和指令的labels为-100，参考回答及EOS参与loss。

运行下面命令会加载真实27B并完成一次前向、反向传播及LoRA参数更新：

```bash
/root/MAOMI/envs/education-training/bin/python scripts/test_education_tensor.py
```

调用链为 `main → load_trainable_model → update_once → model(**batch) → loss.backward() → AdamW.step()`。模型加载与正式微调共享同一个函数，前向使用BF16混合精度。单步测试直接使用非零学习率，不做预热；程序会验证loss和梯度有限，以及LoRA参数确实变化。结果和测试适配器保存在自动创建的`training/education-tensor-test-时间戳/`，不使用正式训练输出目录。

自定义样本可加 `--question "问题" --answer "参考回答"`。这是单批训练链路检查，不能用这一条样本的loss评估教学质量；2500条正式训练仍通过`run_education_finetune.py`单独启动。

默认样本已在真实27B上跑通：输入形状`[1, 212]`，loss约`0.331576`，完成1次优化器更新且LoRA参数最大变化约`5e-5`，峰值已分配显存约`26.7 GiB`。本次测试报告和适配器位于`training/education-tensor-test-v1/`；正式数据训练输出保持独立。
