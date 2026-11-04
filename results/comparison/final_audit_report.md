# COMP9444 Project 013 — B+C Final Verification & Quality Audit

## 1. Overall Assessment

**Passed with minor issues. B+C 可以交付给 A、D、E。**
本次使用现有 9444/.venv；未下载数据/模型，未重训任何实验，未对 Test 推理或修改其标注。四个检查点均在完整 Dev 上重新推理。已确认问题均作最小修复；保留单种子、重复 Train 记录和逐步 AMP 日志不足等局限。

## 2. Critical Checks

| Check | Status | Evidence |
|---|---|---|
| Dataset integrity | PASS | ZIP SHA-256 匹配 provenance；解压文本逐字节匹配；593/100/100 文档记录；跨划分 PMID/完整文档重叠均为 0。 |
| BIO conversion | PASS | 全部 6,892 实体独立恢复；两处 Train 规范化边界例外明确保留原始值；25 对相邻实体亦通过。 |
| Tokenization | PASS | 两个保存的 Tokenizer 分别验证全部 184,550 原始词元；first subword、special/subword/padding -100 逐项检查。 |
| Chunk reconstruction | PASS | 每个 Token 恰好一次；每个 Tokenizer 的 9 个确定性用例通过，包含跨 Chunk 实体；真实语料跨 Chunk 实体数为 0。 |
| Entity-level metrics | PASS | 7 个已知用例；独立严格跨度计数、直接 seqeval Strict IOB2 与 src.ner.entity_metrics 三方一致。 |
| Checkpoint reload | PASS | 四组完整 Dev 推理；每组 882 句、23,969 Token、787 gold 实体；与旧导出预测的不同 Token 数均为 0。 |
| Training comparison | PASS | BERT/BioBERT Initial 全部核心参数及数据哈希相同；TrainingArguments 仅输出路径不同。 |

重新计算的数据规模：

| Split | Documents / unique PMID | Sentences | Tokens | Entities |
|---|---:|---:|---:|---:|
| train | 593 / 592 | 5129 | 136086 | 5145 |
| dev | 100 / 100 | 882 | 23969 | 787 |
| test | 100 / 100 | 885 | 24495 | 960 |

重复 PMID `8528200` 的两条官方 Train 记录文本/标注完全相同，每条含 385 个 Token、11 个实体。为保留官方记录，未删除或合并；该摘要相对其他摘要被加权两次，增加 385 Token 和 11 个实体实例。Train/Dev 仅共享无实体标题 `METHODS.`、`RESULTS.`；共享疾病名称不是文档或标注实例泄漏。

边界复核：PMID `10802668` 的 `[105,131)` → `[105,132)`；PMID `2792129` 的 `[195,240)` → `[195,241)`。原文末尾确实缺少 `r`/`t`。完整末词取得 I-Disease；原始区间在当前词元体系中不能无损表示，原始值始终保留。所有其他原始边界不变。PMID `10923035` 的 `[711,761)` 有引号/空格表面差异，但偏移有效，未改动。

## 3. Recomputed Metrics

以下是新推理结果；评估划分均为 Dev，非 Test。

| Experiment | Precision | Recall | F1 | TP / predicted / gold |
|---|---:|---:|---:|---:|
| BERT Baseline | 0.806913 | 0.860229 | 0.832718 | 677 / 839 / 787 |
| BioBERT Initial | 0.848596 | 0.883100 | 0.865504 | 695 / 819 / 787 |
| BioBERT LR Candidate | 0.847666 | 0.876747 | 0.861961 | 690 / 814 / 787 |
| BioBERT Selected (extended5) | 0.863579 | 0.876747 | 0.870113 | 690 / 799 / 787 |

每组预测与旧文件完全一致；新验证 loss 与旧值的最大绝对差小于 4e-9。检查点与 selection.json、配置和 SHA-256 对应；没有权重覆盖或模型间文件混用。

## 4. Issues Found

### Critical

未发现影响现有标签对齐、指标、权重选择或实验公平性的 Critical 问题。

### Important — 已修复

- **src/data.py / acquire**：原实现每次调用都重写已有原始文本及 provenance，可能掩盖缓存差异。已改为比较既有 archive 哈希和原始文本字节；一致则复用，不一致则报错。真实原始文件哈希/mtime 未改变；临时损坏副本被正确拒绝。实验数据和指标不变。

### Minor — 已修复

- **src/ner.py / smoke_test**：`any` 有限梯度断言不能证明所有可训练参数梯度都有限。已改为逐参数要求梯度存在且有限；两个模型的新 Kernel smoke test 均通过。
- **Notebook**：原始样例按字符截断导致最后一行不完整，存在 IProgress 无关警告，BIO 图例遮挡数据，重复记录影响/规范化边界例外/调度数值说明不充分。已修复并重放；只过滤明确的无关警告，未隐藏模型诊断。
- 审计运行器的 nbclient 索引写回问题已修正，末尾 Markdown 和参考文献已恢复；最终 Notebook 保持 55 Cell、26 代码 Cell、9 图和完整正文。该临时运行器将清理。

### 保留局限与证据边界

- 仅 seed 42；不能声称统计显著或普遍优越。保留官方 Train 重复记录，可能造成少量重复加权。
- 保存的训练/验证 loss、epoch-end gradient norm 和最终权重均有限；gradient norm 是裁剪前的值。没有逐步 AMP scaler 状态，不能证明每步均无溢出。日志 LR 与此前两次未推进 scheduler 的更新相符，可能是正常 FP16 scaler 跳步，无法定位事件；这不影响已经重现的最终 Dev 结果。
- 三个 3-Epoch 实验计划并完成 966 global steps，warmup 97；5-Epoch 实验完成 1,610 steps，warmup 161。全部最佳点为 Epoch 3 / checkpoint-966。Initial 与 Extended 在 Epoch 3 的 LR 分别约 6.90e-8 / 8.93e-6；两条优化轨迹已在前三个 Epoch 不同，因此不能把差异仅归因于多训练两个 Epoch。
- 所有 BioBERT 实验的预训练源文件/commit/hash 相同，代码每次重设 seed 并重新 from_pretrained，未传入上一轮训练后的对象。训练日志未保存初始化参数快照，因此不声称有历史逐步权重快照。

## 5. Marking Criteria Coverage

依据项目本地官方 COMP9444_Group_Project_Marking_Criteria.pdf 的 Notebook 部分逐项验收。

| Criterion | B+C coverage | Remaining limitations / group work |
|---|---|---|
| Data Sources | PASS：官方来源、规模、文本类型、类别、格式、划分和异常记录完整。 | 规范化只用于 Train；已明确与原始跨度的两处差异。 |
| EDA | PASS：统计、4 类核心分布图、密度、样例和模型影响分析完整；新 Kernel 从数据重绘。 | 重复 Train 记录会影响计数，已明确记录。 |
| Models / Methods | PASS：模型依据、分类头、fine-tuning、alignment、chunking、训练配置和引用完整。 | 不是自研架构；第三方方法与本项目工作已区分。 |
| Results | PASS：4 组真实实验、重推理、严格实体指标、loss/F1 曲线及公平对照。 | 单种子；Published Test 结果单列；最终 Test 和完整错误分析交 E。 |
| Writing | PASS：唯一 Notebook、英文正文、清晰标题、最大代码 Cell 30 行、9 张可读图、无执行错误。 | A 仍负责最终整合与全文引用一致性。 |

本次不承诺具体分数。A 的 Introduction/Literature Review、D 的 Domain Adaptation、E 的完整 Discussion/Conclusion 和 held-out Test 评估属于后续整合范围。

## 6. Files Modified

- src/data.py：原始缓存复用及完整性检查。
- src/ner.py：smoke test 的全参数有限梯度检查。
- notebooks/project013.ipynb：最小展示修复、已核实解释及新 Kernel 输出。
- results/eda/label_distribution.png：将 BIO 图例移出数据区。
- results/comparison/quality_assurance.json：刷新受保护重放证据。
- README.md：补充审计入口及重复记录影响。
- 新增 results/comparison/final_audit.json、final_audit_report.md：本次可核查证据及报告。
其他图表/统计通过 Notebook 重新生成；模型权重、原始数据、Test 标注、训练日志、历史预测和实验指标内容保持不变。

## 7. Final Recommendation

**可以正式交付 B+C 给 A、D、E；没有阻塞交付的问题。**
最终受保护重放使用 project013 Kernel，代码执行编号 1–26，无错误；训练和下载调用被禁止。Checkpoint 可重新加载，全部 Dev 预测可完整复现。无需重训现有实验。

机器可读证据：final_audit.json。主交付：../../notebooks/project013.ipynb。
