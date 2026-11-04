# Project 013: disease named entity recognition

**Run only `notebooks/project013.ipynb`.** All project functions are defined inside
its eleven sections: environment; acquisition/loading; preprocessing/BIO validation;
EDA; shared NER pipeline; BERT; BioBERT; hyperparameter experiments; PubMedBERT;
SciBERT; comparison.
`src/` contains historical references and is not imported or executed.
No external Python script or `%run` is required.

## Environment

Select **COMP9444 Project 013 (.venv)** using the existing `9444/.venv`.
The notebook locates the project through relative directories, checks `sys.prefix`
and dependencies, and reports a clear error for an incorrect kernel. It never
switches to system Python or creates an environment. The original environment was
Python 3.10.11, PyTorch 2.14.0+cu130, Transformers 4.57.6 and an RTX 3070 Laptop GPU.
Current runtime information is displayed directly, without requiring environment.json.

`requirements.txt` is an optional installation convenience. From `9444`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r project\requirements.txt
```

Keep the existing CUDA-enabled PyTorch installation.

## Reproduction mode (default)

Leave `RETRAIN = False` and `TRAIN_NEW_MODELS = False`, then run all cells sequentially.
The notebook:

1. Verifies existing raw text without repeated downloads, even if ZIP files are
   absent. Missing data is downloaded from official NCBI archives and checked
   against pinned archive/text SHA-256 values and ZIP CRCs.
2. Rebuilds every processed sentence, label and entity from raw annotations.
   Existing processed exports must agree; absent exports are created. Processed
   data is not an input requirement.
3. Generates and displays EDA from the reconstructed data.
4. Reloads all six fine-tuned checkpoints with their saved tokenizer/config,
   performs fresh Dev inference, and recomputes strict IOB2 entity metrics.
5. Verifies fresh predictions against the original records and displays saved
   histories, loss/F1 curves and comparisons.

No training or pretrained-model download is needed when the supplied experiments
are available. Missing checkpoints or experiment records produce an actionable
error; there is no automatic training fallback or fabricated metric. Saved
experiment records and original figures are preserved. Derived comparison tables
and figures are refreshed from the saved records on each run.

## Training mode

Set `RETRAIN = True`, restart and run all cells. Each experiment resets seed 42
and loads the pinned original pretrained encoder. Missing pretrained assets may
be downloaded from Hugging Face.

New files use timestamped sibling folders such as
`models/biobert/baseline_<RUN_TAG>/best/` and
`results/biobert/baseline_<RUN_TAG>/`. BioBERT tuning runs remain under
`models/biobert/tuning/` and `results/biobert/tuning/`, with matching tagged run names. Tuning/comparison outputs also
receive a tag. Existing experiment folders and audit records remain unchanged.
A new selected checkpoint is reported by its `best_checkpoint` path.
Restart before another training attempt to obtain a new tag.

With `RETRAIN = False`, `TRAIN_NEW_MODELS = True` trains only missing PubMedBERT
and SciBERT baselines; completed baselines are reloaded. Their first runs use
`models/{pubmedbert,scibert}/baseline/best/` and matching result directories.
All supplied runs are complete, so both flags are off by default.

The initial comparison uses LR 2e-5, 3 epochs, effective batch 16 (8 x 2), max
length 128, weight decay 0.01, AdamW, linear scheduling with 10% warmup and FP16.
The LR candidate changes only LR to 3e-5. The 5-epoch experiment uses the better
3-epoch LR with a newly scaled schedule; its selected epoch is 3. Its first three
epochs already follow a different LR trajectory. Test is never used for selection.

Each model uses its own tokenizer: BERT and BioBERT are cased, PubMedBERT is
uncased, and SciBERT uses cased SciVocab. Whole-token chunk counts differ slightly:
BERT/BioBERT use 966 optimizer steps over three epochs, and PubMedBERT/SciBERT use
963. All use 97 warmup steps. The detailed configuration comparison is in
`results/comparison/four_model_training_settings.csv`.

## Dev results

The matched three-epoch experiments use the same seed and training configuration.
All four select epoch 3 by strict IOB2 entity F1.

| Model | Precision | Recall | Entity F1 |
|---|---:|---:|---:|
| BERT-base | 0.806913 | 0.860229 | 0.832718 |
| BioBERT Initial | 0.848596 | 0.883100 | 0.865504 |
| PubMedBERT | 0.855377 | 0.879288 | 0.867168 |
| SciBERT | 0.847584 | 0.869123 | 0.858218 |

BioBERT Selected retains F1 0.870113 from the separate five-epoch-budget experiment.
It is excluded from the matched-budget table. Results use a single seed, and
vocabulary, casing and pretraining differences prevent isolating a corpus effect.

The new models start from
[`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract)
and [`allenai/scibert_scivocab_cased`](https://huggingface.co/allenai/scibert_scivocab_cased).
Their pinned revisions and source hashes are recorded with each run.

## Comparison artifacts

`results/comparison/` separates the four matched-budget baselines from BioBERT tuning:

- `model_comparison.csv` and `baseline_metrics_summary.json`: four-baseline Dev
  precision, recall, entity F1, configuration and source-file references.
- `model_comparison.png`: four-baseline entity F1.
- `precision_recall_f1_comparison.png`: grouped precision, recall and F1.
- `validation_loss_comparison.png` and `baseline_validation_history.csv`: all four
  three-epoch Dev loss histories. Loss depends on tokenization and is not used for selection.
- `biobert_tuning_comparison.csv` and `.png`: Initial, LR Candidate and
  Selected / Extended5, separate from the baseline comparison. The CSV also updates
  `results/biobert/tuning_results.csv`.
- `four_model_training_settings.csv` and `four_model_tokenization_audit.csv`:
  configuration and chunk-count checks. The EDA tokenization audit covers all four tokenizers.

`four_model_comparison.csv` and `four_model_f1.png` are synchronized aliases of
`model_comparison.csv` and `.png` for existing references. Notebook comparison
images display the exported PNG files directly. Original per-experiment curves,
dataset statistics and EDA figures are retained.

## Experiment layout

```text
models/
├── bert_base/
│   ├── pretrained/
│   └── baseline/best/
├── biobert/
│   ├── pretrained/
│   ├── baseline/best/          # BioBERT Initial: matched-budget baseline
│   └── tuning/
│       ├── lr3e5/best/
│       ├── extended5/best/     # BioBERT Selected
│       └── selection.json
├── pubmedbert/
│   ├── pretrained/
│   └── baseline/best/
└── scibert/
    ├── pretrained/
    └── baseline/best/
```

The duplicate root-level BERT/BioBERT `best/` exports were byte-identical copies
and have been removed. `selection.json` points to the retained canonical checkpoint;
BioBERT's selected tuning run is separate from its initial baseline.

The results directories mirror the checkpoint runs:

```text
results/
├── bert_base/baseline/
├── biobert/
│   ├── baseline/              # BioBERT Initial
│   ├── tuning/
│   │   ├── lr3e5/
│   │   └── extended5/         # BioBERT Selected
│   └── tuning_results.csv
├── pubmedbert/
│   ├── baseline/
│   └── smoke_test.json
├── scibert/
│   ├── baseline/
│   └── smoke_test.json
├── eda/
├── comparison/
├── environment.json
└── smoke_test.json
```

Each of the six experiment directories contains `training_config.json`,
`training_history.csv`, `trainer_log.json`, `validation_metrics.json`,
`predictions.jsonl`, `training_curves.png` and `tokenization_audit.json`.
Metrics and comparison tables identify the matching `best_checkpoint` and
`result_directory`; `tuning_results.csv` remains at the BioBERT root.

BioBERT's historical experiment ID `initial` is retained in signatures and
predictions, and resolves to `baseline/` in both directory trees. Historical
audit snapshots and serialized training arguments retain their original paths;
current configuration JSON and summary tables use the relocated paths.

## Files to deliver

- **Required code:** `notebooks/project013.ipynb` only.
- **Required for default reproduction:** `models/{bert_base,biobert,pubmedbert,scibert}/baseline/best/`
  and `models/biobert/tuning/{lr3e5,extended5}/best/`, including model/config/tokenizer
  assets; matching `results/{model}/baseline/` and
  `results/biobert/tuning/{lr3e5,extended5}/` configuration, history, metrics and
  predictions.
- **Keep with the project:** results figures/comparisons/audits and original course
  PDFs/member-allocation image.
- **For offline use:** `data/raw/`. Otherwise the notebook acquires it automatically.
- **Rebuildable:** `data/processed/`. Keep the supplied exports for handoff, although
  they are not needed to initialize the notebook.
- **Optional reference:** `src/`, `requirements.txt`, cached pretrained assets and
  `results/environment.json`. Pretrained assets are needed only for new training
  and may be downloaded automatically.

## Data conventions and handoff

The official 593/100/100 records contain 592/100/100 unique PMIDs. PMID 8528200
appears twice with identical text/annotations, weighting that Train abstract twice
(385 extra tokens and 11 extra entity instances). Both official records are kept.
Two documented Train end offsets are extended by one character; original offsets
and surfaces are retained. Dev/Test annotations are unchanged.

Disease recognition uses O/B-Disease/I-Disease. First-subword supervision,
whole-token chunking, full-sentence reconstruction and strict seqeval IOB2 scoring
are unchanged. The reusable dataset/training/inference/metric functions are now
in the notebook. A handles final integration, D adds domain adaptation, and E
performs held-out Test evaluation and full error analysis after selection is frozen.

## Verification evidence

The current [comparison verification](results/comparison/comparison_verification.json)
records a sequential offline replay with training and local-module imports blocked:
63 code cells, 14 figures and no errors. Plot data, CSV/JSON summaries and all six
saved experiments agree. The four comparison images embedded in the notebook match
the exported PNG bytes. Model assets, experiment records, original curves and data
are unchanged. The current `quality_assurance.json` covers this replay and retains
its earlier verification snapshot.

The earlier [results-layout verification](results/comparison/results_layout_verification.json)
records a fresh sequential notebook run with training, downloads and local-module
imports blocked. All six checkpoints reproduce the saved Dev metrics and original
predictions. Every result file is retained; only path references and directory
locations change. Histories, trainer logs, predictions, figures, data and all model
artifacts remain byte-identical. No model was retrained.

The earlier [checkpoint-layout verification](models/checkpoint_layout_verification.json)
records the directory moves, duplicate hashes and a fresh sequential notebook run
with training and downloads blocked. All six checkpoints reproduce their historical
Dev metrics and predictions; data and results files were unchanged at that stage.
Every original model asset is retained byte-for-byte at its canonical path, apart
from the updated selected-checkpoint pointer. No model was retrained.

The earlier [four-model verification](results/comparison/four_model_verification.json)
records a fresh-kernel, offline run with training, downloads and `src` imports
blocked: 56 code cells, 12 figures and no execution errors. All six checkpoints
were reloaded and their Dev metrics and predictions matched saved records. Both new
tokenizers passed forced cross-chunk entity reconstruction checks. The new runs
completed three epochs with finite losses, matching legacy training arguments.
All 131 pre-existing protected artifacts retained their hashes, including old
weights, experiment records, EDA and audit files. Only the two new models were trained.

The [four-model table](results/comparison/model_comparison.csv),
[configuration comparison](results/comparison/four_model_training_settings.csv) and
[F1 figure](results/comparison/model_comparison.png) accompany the notebook outputs.
The separate [BioBERT tuning table](results/comparison/biobert_tuning_comparison.csv)
and [tuning figure](results/comparison/biobert_tuning_comparison.png) retain its extra experiments.

[Self-containment verification](results/comparison/self_containment_verification.json)
records two earlier successful fresh-kernel runs: Test A with processed data isolated and
rebuilt byte-identically, and Test B with `src/` physically absent and its imports
blocked. Both executed 49 code cells and displayed 9 figures. All four Dev F1 values
matched: 0.832718, 0.865504, 0.861961 and 0.870113. Training/download calls were
blocked. All 121 pre-existing protected files retained their hashes; isolated
originals were restored.

Additional safe temporary-directory tests verified actual official downloads and
the raw-text-only cache branch with downloads blocked. Original audited functions,
TrainingArguments and Trainer constructors were compared structurally (AST).

The earlier [audit report](results/comparison/final_audit_report.md) and
[audit evidence](results/comparison/final_audit.json) are retained as historical
records. The new extension preserves the original BERT/BioBERT experiment cells and
the core data, alignment, reconstruction, evaluation and checkpoint-selection
functions. No BERT/BioBERT training or held-out Test inference was performed.
