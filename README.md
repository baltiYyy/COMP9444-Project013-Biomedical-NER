# COMP9444 Project 013 项目概览

当前已完成 NCBI Disease NER 的四个 Baseline，以及两组 BioBERT 调参实验。

## 项目结构

```text
project/
├── data/                         # raw/：官方原始数据；processed/：预处理导出
├── models/                       # 预训练权重、微调 Checkpoint 与 Tokenizer
│   ├── bert_base/
│   ├── biobert/                  # 另有 tuning/lr3e5/ 与 tuning/extended5/
│   ├── pubmedbert/
│   └── scibert/
├── notebooks/project013.ipynb     # 主运行入口，包含完整共享 NER pipeline
├── results/                      # 配置、日志、指标、预测及图表
│   ├── bert_base/baseline/
│   ├── biobert/                  # baseline/、tuning/、tuning_results.csv
│   ├── pubmedbert/baseline/
│   ├── scibert/baseline/
│   ├── comparison/               # 四模型公平对比、独立 BioBERT 调参与验证记录
│   └── eda/                      # 数据统计与 EDA 图表
├── src/                          # 历史参考模块，Notebook 不依赖其导入
├── README.md                     # 详细运行、目录及复现说明
└── requirements.txt              # 依赖清单
```

四个模型均使用 `models/<model>/pretrained/` 和 `models/<model>/baseline/best/`；结果对应 `results/<model>/baseline/`。
BioBERT 的 Initial 就是公平 Baseline；额外实验在两棵目录的 `biobert/tuning/{lr3e5,extended5}/` 下，Checkpoint 再进入 `best/`。

## 训练情况

四个 Baseline 均完成 3 epochs，LR `2e-5`、batch size `8`、gradient accumulation `2`（effective batch `16`）、max length `128`。
共同采用 AdamW、linear scheduler、10% warmup、weight decay `0.01`、seed `42` 和 FP16；各模型使用自己的 Tokenizer，PubMedBERT 为 uncased，其余为 cased。
BioBERT LR Candidate 使用 LR `3e-5`、3 epochs；Extended5 使用 LR `2e-5`、5 epochs，并重新缩放学习率调度。

## Dev 实验结果

| Baseline | Precision | Recall | Entity F1 |
|---|---:|---:|---:|
| BERT-base | 0.806913 | 0.860229 | 0.832718 |
| BioBERT Initial | 0.848596 | 0.883100 | 0.865504 |
| PubMedBERT | 0.855377 | 0.879288 | 0.867168 |
| SciBERT | 0.847584 | 0.869123 | 0.858218 |

BioBERT LR Candidate F1 为 **0.861961**；最佳调参结果 Selected / Extended5 为 **0.870113**，最佳 Epoch 为 3，权重位于 `models/biobert/tuning/extended5/best/`。
全部按 Dev strict IOB2 entity-level F1 选 Checkpoint；四个 Baseline 也均选 Epoch 3。Selected 不属于同预算 Baseline 对照；单一 seed、词表及大小写差异限制了结论。

## 运行方法

打开 `notebooks/project013.ipynb`，选择现有 `9444/.venv` 对应的 **COMP9444 Project 013 (.venv)** Kernel，再顺序运行全部 Cell。
默认 `RETRAIN=False`、`TRAIN_NEW_MODELS=False`：自动检查原始数据并预处理，加载六组 Checkpoint，重新计算 Dev 指标并更新对比图表，不训练。
`RETRAIN=True`：从各自预训练权重重新训练四个 Baseline 及 BioBERT 调参实验，结果写入独立的时间戳目录，保留既有实验；普通查看和复现无需开启。

## 后续工作

- **D：Domain Adaptation**：复用 Notebook 的数据处理、alignment、训练和评估 pipeline，以及 `pretrained/` 权重；保持 Train/Dev 划分，新 Checkpoint 和结果另存，避免覆盖 Baseline。
- **E：最终评估**：冻结模型选择后，用对应 `best/` 权重在官方 Test 上统一评估并导出 Test 预测；现有 `predictions.jsonl` 均为 Dev，可用于前期错误分析，不能作为 Test 结果。

## 获取模型与运行补充

本仓库的 10 个模型权重使用 **Git LFS** 保存（合计约 4.05 GiB），其余代码、数据、记录和图表使用普通 Git。
安装 Git LFS 后，建议克隆到 `9444/project/`，并使用已有的 `9444/.venv`：

```bash
git lfs install
git clone https://github.com/baltiYyy/COMP9444-Project013-Biomedical-NER.git project
git -C project lfs pull
```

打开 `project/notebooks/project013.ipynb`，选择 `COMP9444 Project 013 (.venv)` Kernel，保持 `RETRAIN=False`、`TRAIN_NEW_MODELS=False`，再顺序运行。
保留原有 CUDA PyTorch；其余依赖见 `requirements.txt`。环境检查失败时按提示选择正确的 Kernel，不会自动切换 Python。
缺少原始数据时 Notebook 会自动获取；默认复现需要完整下载 Checkpoint 及匹配的结果记录。
详细环境、运行、目录和实验说明见 [RUNNING.md](RUNNING.md)。

## 公开版本说明

发布副本仅对 Notebook 输出和 JSON 运行元数据中的本机个人路径做脱敏，训练代码、指标、数据和模型权重保持不变。
缓存、临时文件与含本机路径的 `training_args.bin` 未公开；完整训练参数仍保存在各实验的 `training_config.json` 中。
历史审计文件描述发布前的验证，其文件摘要可能对应原始元数据；公开文件的实际摘要和权重清单见 [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json)。
