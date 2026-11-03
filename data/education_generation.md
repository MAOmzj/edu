<!-- 文件用途：说明教学数据新增1500条、复核修订、保留留出集并合成2500条训练集的流程。 -->
<!-- 调用关系：prepare_education_expansion生成输入，generate/revise调用本地教师，combine输出新数据版本。 -->
<!-- 修改易踩坑：新增只用原训练任务族；reasoning和复核反馈不进入训练messages；已有数据不能覆盖。 -->
# 教学数据扩充

本次目标是在原1000条训练数据上新增1500条，得到2500条训练、100条验证和100条测试。四科新增各375条；合并后训练每科625条。输入是原创合成题与教学对话，最终回答由本地`qwen3.8-27b`生成，服务实际使用`Qwen3.8-27B-AWQ-INT4`推理权重。

新增数据只扩展原训练任务族，原验证/测试任务族不参与扩充。原题与变体属于同一个`group_id`；2500条包含同一教学任务的不同情境，不代表2500个独立知识点。原始数据、生成候选、复核意见与修订台账分别保留。

准备新增输入，不调用模型：

```bash
cd /root/MAOMI/edu
/root/MAOMI/envs/education-training/bin/python scripts/prepare_education_expansion.py
```

先启动聊天教师服务，等`serve.py status`显示可用；本任务不需要向量、摘要或网页服务：

```bash
.venv/bin/python scripts/serve.py chat -d
.venv/bin/python scripts/serve.py status
```

生成新增候选，先不发布训练文件：

```bash
/root/MAOMI/envs/education-training/bin/python scripts/generate_education_data.py \
  --inputs-file /root/MAOMI/training/education-expansion-inputs-v1.jsonl \
  --output-dir /root/MAOMI/training/education-addition-v1 \
  --train-count 1500 --validation-count 0 --test-count 0 \
  --teacher-thinking --temperature 0.3 --max-tokens 2048 \
  --defer-publication
```

教师可以思考，vLLM的reasoning parser分离思考和最终回答。数据目标只取`message.content`，保存`reasoning_content_present`布尔标记，不把reasoning内容放入训练文件。训练分词仍使用`enable_thinking=False`，只监督最后assistant教学回答及EOS，历史消息、提示与padding不计算loss。

中断续跑需保留同一输入和参数，在相同命令上加`--resume`。服务失败、截断、空回答、工具调用、控制标签、超长训练序列都不作为已成功目标；错误记录在`failed_attempts.jsonl`。候选与输入哈希不符会停止，不能靠换ID或跳过坏行凑数量。

复核反馈文件是`{"样本id":"明确的事实与教学修正要求"}`形式的JSON对象。需要修订时调用：

```bash
/root/MAOMI/envs/education-training/bin/python scripts/revise_education_data.py \
  --output-dir /root/MAOMI/training/education-addition-v1 \
  --feedback /绝对路径/feedback.json --teacher-thinking --regenerate \
  --concurrency 12
```

`--regenerate`将核验提示加入教师系统消息，让教师依据原题重新作答；不提供上一版错误回答。省略此项则采用上一版回答加校对反馈的修订对话。两种方式都会再次调用真实教师模型，保留完整原候选、实际请求、response ID、参数与新旧哈希。反馈和校对上下文仅保存在来源元数据中；最终训练`messages`仍为原学生对话加新回答，不能把校对对话作为学生上下文。

复核修订完成后，在生成命令上加`--resume`并去掉`--defer-publication`，发布新增`train.jsonl`。然后合并已发布的两批：

```bash
/root/MAOMI/envs/education-training/bin/python scripts/combine_education_data.py
```

合并输出为`/root/MAOMI/training/education-dataset-v2/`，再次检查逐条来源、ID/实际输入去重、任务组隔离、真实token长度与EOS监督，并要求验证/测试文件与原版逐字节一致。`dataset_report.json`保存数量和文件哈希。合并脚本不会改原文件或自动启动训练。

`review_status: pending`表示合成候选未成为人工金标，不是审批门槛。结构、AI复核和修订不能保证全批教学正确率；留出集仍是合成分布，应另用真实教学问题比较未微调与微调模型。正式微调还需准备完整未量化27B基座，详见[教育微调说明](education_finetune.md)。
