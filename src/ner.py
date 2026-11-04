"""One Hugging Face token-classification pipeline shared by BERT and BioBERT."""
from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
import transformers
from huggingface_hub import HfApi, snapshot_download
from seqeval.metrics import precision_score, recall_score, f1_score
from seqeval.scheme import IOB2
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
    DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed)

from .data import LABELS, label2id, id2label, write_json, write_jsonl

MODEL_CONFIGS = {"bert_base": "bert-base-cased", "biobert": "dmis-lab/biobert-base-cased-v1.1"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_pretrained(root, model_key):
    directory = Path(root) / "models" / model_key / "pretrained"
    marker = directory / "source_revision.json"
    if marker.exists():
        return directory, json.loads(marker.read_text())
    repository = MODEL_CONFIGS[model_key]
    info = HfApi().model_info(repository)
    available = {file.rfilename for file in info.siblings}
    weight = "model.safetensors" if "model.safetensors" in available else "pytorch_model.bin"
    assert weight in available
    patterns = [weight, "config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json"]
    snapshot_download(repository, revision=info.sha, local_dir=directory,
                      allow_patterns=patterns, max_workers=2)
    source_config = json.loads((directory / "config.json").read_text())
    assert not any("TokenClassification" in name for name in source_config.get("architectures", []))
    record = {"requested_repository": repository, "resolved_repository": info.id,
        "revision": info.sha, "weight_file": weight, "weight_sha256": sha256(directory / weight),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(), "source_config": source_config,
        "purpose": "pretrained encoder; not an NCBI Disease fine-tuned checkpoint"}
    write_json(marker, record)
    return directory, record


def encode_samples(samples, tokenizer, max_length=128):
    """Chunk at whole-word boundaries; every original token is supervised once."""
    features, mappings, full_lengths = [], [], []
    cut_inside_entity = 0
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    for sample_index, sample in enumerate(samples):
        raw = tokenizer(sample["tokens"], is_split_into_words=True, add_special_tokens=False,
                        truncation=False, return_attention_mask=False, return_token_type_ids=False)
        words = raw.word_ids()
        sizes = np.bincount(words, minlength=len(sample["tokens"]))
        assert len(sizes) == len(sample["tokens"]) and np.all(sizes > 0)
        assert max(sizes) <= budget, "A single token exceeds the subword budget."
        full_lengths.append(len(raw["input_ids"]) + tokenizer.num_special_tokens_to_add(False))
        start = 0
        while start < len(sizes):
            end, used = start, 0
            while end < len(sizes) and used + sizes[end] <= budget:
                used += sizes[end]
                end += 1
            encoded = tokenizer(sample["tokens"][start:end], is_split_into_words=True,
                                truncation=False)
            assert len(encoded["input_ids"]) <= max_length
            previous = None
            labels, positions, word_indices = [], [], []
            for position, word in enumerate(encoded.word_ids()):
                if word is not None and word != previous:
                    labels.append(label2id[sample["ner_tags"][start + word]])
                    positions.append(position)
                    word_indices.append(start + word)
                else:
                    labels.append(-100)
                previous = word
            assert word_indices == list(range(start, end))
            features.append({**dict(encoded), "labels": labels})
            mappings.append({"sample_index": sample_index, "word_indices": word_indices,
                             "positions": positions})
            if end < len(sizes) and sample["ner_tags"][end] == "I-Disease":
                cut_inside_entity += 1
            start = end
    assert sum(len(m["word_indices"]) for m in mappings) == sum(len(s["tokens"]) for s in samples)
    audit = {"sentence_samples": len(samples), "encoded_chunks": len(features),
        "sentences_over_max_length": sum(n > max_length for n in full_lengths),
        "max_sentence_subwords_with_specials": max(full_lengths),
        "mean_sentence_subwords_with_specials": float(np.mean(full_lengths)),
        "entity_crossing_chunk_boundaries": cut_inside_entity,
        "original_tokens": sum(len(s["tokens"]) for s in samples), "dropped_tokens": 0}
    return features, mappings, audit


def decode_predictions(logits, mappings, samples):
    predictions = [[None] * len(sample["tokens"]) for sample in samples]
    ids = np.argmax(logits, axis=-1)
    assert len(ids) == len(mappings)
    for row, mapping in zip(ids, mappings):
        for position, word in zip(mapping["positions"], mapping["word_indices"]):
            target = predictions[mapping["sample_index"]]
            assert target[word] is None
            target[word] = id2label[int(row[position])]
    assert all(all(tag is not None for tag in row) for row in predictions)
    return predictions


def entity_metrics(gold, predicted):
    assert len(gold) == len(predicted) and all(len(a) == len(b) for a, b in zip(gold, predicted))
    kwargs = {"mode": "strict", "scheme": IOB2, "zero_division": 0}
    return {"precision": float(precision_score(gold, predicted, **kwargs)),
            "recall": float(recall_score(gold, predicted, **kwargs)),
            "f1": float(f1_score(gold, predicted, **kwargs))}


def metric_function(mappings, samples):
    gold = [s["ner_tags"] for s in samples]
    def compute(evaluation):
        predicted = decode_predictions(evaluation.predictions, mappings, samples)
        # Padding, special tokens, and non-first subwords are never decoded.
        return entity_metrics(gold, predicted)
    return compute


def load_model(directory):
    return AutoModelForTokenClassification.from_pretrained(directory, num_labels=len(LABELS),
        id2label=id2label, label2id=label2id, attn_implementation="sdpa")


def smoke_test(directory, tokenizer, samples, max_length=128):
    set_seed(42)
    model = load_model(directory).to("cuda" if torch.cuda.is_available() else "cpu")
    features, mapping, audit = encode_samples(samples, tokenizer, max_length)
    batch = DataCollatorForTokenClassification(tokenizer)(features)
    batch = {k:v.to(model.device) for k,v in batch.items()}
    output = model(**batch)
    assert output.logits.shape[:2] == batch["labels"].shape
    assert output.logits.shape[-1] == len(LABELS) and torch.isfinite(output.loss)
    output.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    predicted = decode_predictions(output.logits.detach().float().cpu().numpy(), mapping, samples)
    # A boundary error must lower entity F1; exact gold recovery must score one.
    fixture = [["B-Disease", "I-Disease", "O", "B-Disease"]]
    partial = [["B-Disease", "O", "O", "B-Disease"]]
    assert entity_metrics(fixture, fixture)["f1"] == 1.0
    assert entity_metrics(fixture, partial)["f1"] == 0.5
    assert entity_metrics([["O", "B-Disease"]], [["O", "I-Disease"]])["f1"] == 0.0
    # Force chunking and verify that reconstruction recovers every gold token.
    tiny_features, tiny_map, _ = encode_samples(samples, tokenizer, max_length=16)
    mock = np.zeros((len(tiny_features), 16, len(LABELS)), dtype=np.float32)
    for i, feature in enumerate(tiny_features):
        for j, label in enumerate(feature["labels"]):
            if label != -100:
                mock[i,j,label] = 1
    assert decode_predictions(mock, tiny_map, samples) == [s["ner_tags"] for s in samples]
    result = {"status": "passed", "loss": output.loss.item(), "batch_shape": list(batch['input_ids'].shape),
              "logits_shape": list(output.logits.shape), "backward_pass": True,
              "prediction_metric": entity_metrics([s["ner_tags"] for s in samples], predicted),
              "forced_chunk_gold_roundtrip": True, **audit}
    del model, output, batch
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_experiment(root, model_key, run_name, config, datasets, pretrained, source, retrain=False):
    """Train on Train; select the maximum Dev entity-F1 epoch; export Dev predictions."""
    root = Path(root)
    model_output = root / "models" / model_key / run_name
    if model_key == 'biobert':
        model_output = (root/'models'/model_key/'baseline' if run_name == 'initial'
                        else root/'models'/model_key/'tuning'/run_name)
    output = root/'results'/model_output.relative_to(root/'models')
    output.mkdir(parents=True, exist_ok=True)
    model_output.mkdir(parents=True, exist_ok=True)
    fingerprint = {s:sha256(root / 'data/processed' / f'{s}.jsonl') for s in ['train','dev']}
    signature = {"model_key":model_key, "run_name":run_name, "hyperparameters":config,
                 "data_sha256":fingerprint, "pretrained_revision":source['revision']}
    metric_path = output / "validation_metrics.json"
    if not retrain and metric_path.exists():
        recorded = json.loads((output / "training_config.json").read_text())
        assert recorded['signature'] == signature, "Configuration changed; set RETRAIN=True."
        metrics = json.loads(metric_path.read_text())
        assert (model_output / 'best/model.safetensors').exists()
        assert sha256(model_output / 'best/model.safetensors') == metrics['checkpoint_sha256']
        metrics['best_checkpoint'] = (model_output / 'best').relative_to(root).as_posix()
        metrics['result_directory'] = output.relative_to(root).as_posix()
        print(f"Reusing verified completed run: {model_key}/{run_name}", flush=True)
        return metrics, pd.read_csv(output / 'training_history.csv')
    set_seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True, do_lower_case=False)
    encoded = {s:encode_samples(datasets[s], tokenizer, config['max_length']) for s in ['train','dev']}
    write_json(output / "tokenization_audit.json", {s:parts[2] for s,parts in encoded.items()})
    model = load_model(pretrained)
    args = TrainingArguments(output_dir=str(model_output / 'checkpoints'),
        learning_rate=config['learning_rate'], num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['batch_size'], per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        weight_decay=config['weight_decay'], warmup_ratio=config['warmup_ratio'],
        lr_scheduler_type='linear', optim='adamw_torch', max_grad_norm=1.0,
        eval_strategy='epoch', save_strategy='epoch', logging_strategy='epoch',
        metric_for_best_model='f1', greater_is_better=True, load_best_model_at_end=True,
        save_total_limit=1, save_only_model=True, seed=config['seed'], data_seed=config['seed'],
        fp16=config['fp16'], report_to=[], disable_tqdm=True, dataloader_num_workers=0,
        dataloader_pin_memory=True, full_determinism=False)
    recorded = {"signature":signature, "model_source":source,
        "python_executable": __import__('sys').executable,
        "torch":torch.__version__, "transformers":transformers.__version__,
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        "started_utc":datetime.now(timezone.utc).isoformat(),
        "optimizer":"torch AdamW", "scheduler":"linear with warmup",
        "selection":"maximum Dev strict IOB2 entity F1; earliest epoch on ties",
        "subword_policy":"first subword only; remaining/special/padding labels=-100",
        "long_sentence_policy":"non-overlapping whole-word chunks; full-sentence reconstruction",
        "test_policy":"not evaluated or used for selection", "training_arguments":args.to_dict()}
    write_json(output / 'training_config.json', recorded)
    trainer = Trainer(model=model, args=args, train_dataset=encoded['train'][0],
        eval_dataset=encoded['dev'][0], processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8),
        compute_metrics=metric_function(encoded['dev'][1], datasets['dev']))
    start = time.perf_counter()
    trainer.train()
    duration = time.perf_counter() - start
    write_json(output / 'trainer_log.json', trainer.state.log_history)
    epoch_rows = []
    for log in trainer.state.log_history:
        if 'eval_f1' in log:
            training_log = next(x for x in trainer.state.log_history
                                if 'loss' in x and abs(x['epoch']-log['epoch']) < 1e-5)
            epoch_rows.append({'epoch':log['epoch'], 'training_loss':training_log['loss'],
                'validation_loss':log['eval_loss'], 'precision':log['eval_precision'],
                'recall':log['eval_recall'], 'f1':log['eval_f1']})
    history = pd.DataFrame(epoch_rows)
    history.to_csv(output / 'training_history.csv', index=False)
    best_row = history.loc[history['f1'].idxmax()].to_dict()
    selected_checkpoint = Path(trainer.state.best_model_checkpoint).name
    trainer.save_model(str(model_output / 'best'))
    tokenizer.save_pretrained(model_output / 'best')
    prediction = trainer.predict(encoded['dev'][0])
    predicted = decode_predictions(prediction.predictions, encoded['dev'][1], datasets['dev'])
    metrics = {**entity_metrics([r['ner_tags'] for r in datasets['dev']], predicted),
        'validation_loss':float(prediction.metrics['test_loss']), 'split':'dev',
        'model':MODEL_CONFIGS[model_key], 'run_name':run_name, 'best_epoch':best_row['epoch'],
        'selected_trainer_checkpoint':selected_checkpoint,
        'best_checkpoint':(model_output / 'best').relative_to(root).as_posix(),
        'checkpoint_sha256':sha256(model_output / 'best/model.safetensors'),
        'training_seconds':duration, 'number_of_sentences':len(datasets['dev']),
        'number_of_gold_entities':sum(len(r['entities']) for r in datasets['dev']),
        'completed_utc':datetime.now(timezone.utc).isoformat()}
    assert abs(metrics['f1']-best_row['f1']) < 1e-12
    write_jsonl(output / 'predictions.jsonl', [{
        'sample_id':sample['id'], 'document_id':sample['document_id'], 'pmid':sample['pmid'],
        'split':'dev', 'tokens':sample['tokens'], 'gold_labels':sample['ner_tags'],
        'predicted_labels':tags, 'token_offsets':sample['token_offsets'],
        'gold_entities':sample['entities'], 'model':metrics['model'], 'run_name':run_name,
        'pretrained_revision':source['revision'], 'checkpoint_sha256':metrics['checkpoint_sha256']}
        for sample,tags in zip(datasets['dev'], predicted)])
    # Reload the saved checkpoint and require identical logits on a real batch.
    restored = AutoModelForTokenClassification.from_pretrained(model_output / 'best',
                    attn_implementation='sdpa').to(trainer.model.device).eval()
    batch = trainer.data_collator(encoded['dev'][0][:2])
    batch = {k:v.to(trainer.model.device) for k,v in batch.items() if k != 'labels'}
    trainer.model.eval()
    # Accelerate wraps Trainer.forward in FP16 autocast; use the same context
    # for the independently restored model instead of comparing FP16 to FP32.
    assert all(torch.equal(value, restored.state_dict()[name])
               for name,value in trainer.model.state_dict().items())
    with torch.no_grad(), torch.autocast(device_type=trainer.model.device.type,
            dtype=torch.float16, enabled=config['fp16']):
        before, after = trainer.model(**batch).logits, restored(**batch).logits
    assert torch.allclose(before.float(), after.float(), atol=1e-6, rtol=1e-5)
    metrics['checkpoint_reload_verified'] = True
    write_json(metric_path, metrics)
    del restored, before, after, prediction, trainer, model, batch
    gc.collect()
    torch.cuda.empty_cache()
    # Only completed Trainer-generated checkpoints inside this run are removed.
    temporary = (model_output / 'checkpoints').resolve()
    assert temporary.is_relative_to((root / 'models').resolve())
    if temporary.exists():
        shutil.rmtree(temporary)
    return metrics, history
