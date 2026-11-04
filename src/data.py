"""NCBI PubTator parsing with auditable character offsets and model-free tokens."""
from collections import Counter
from pathlib import Path
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone

import requests
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

LABELS = ["O", "B-Disease", "I-Disease"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = dict(enumerate(LABELS))
SOURCE = "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/"
FILES = {"train": "NCBItrainset_corpus", "dev": "NCBIdevelopset_corpus", "test": "NCBItestset_corpus"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def acquire(root):
    raw = Path(root) / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    old = raw / "provenance.json"
    recorded = json.loads(old.read_text(encoding="utf-8")) if old.exists() else None
    manifest = {"source": SOURCE, "retrieved_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
    for split, stem in FILES.items():
        archive = raw / (stem + ".zip")
        if not archive.exists():
            response = requests.get(SOURCE + archive.name, timeout=90)
            response.raise_for_status()
            archive.write_bytes(response.content)
        payload = archive.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if recorded is not None:
            assert digest == recorded["files"][split]["sha256"], f"Cached archive changed: {archive}"
        with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
            original = zipped.read(stem + ".txt")
        text_path = raw / (stem + ".txt")
        if text_path.exists():
            assert text_path.read_bytes() == original, f"Raw text differs from archive: {text_path}"
        else:
            text_path.write_bytes(original)
        manifest["files"][split] = {"url": SOURCE + archive.name, "archive": archive.name,
            "sha256": digest, "extracted": [stem + ".txt"]}
    # Preserve the date of the original download on subsequent local executions.
    if recorded is not None:
        manifest["retrieved_utc"] = recorded["retrieved_utc"]
        assert manifest == recorded, "Cached dataset provenance changed."
    else:
        write_json(old, manifest)
    return manifest


def parse_pubtator(path, split):
    documents, occurrences = [], Counter()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if "|t|" in line:
            pmid, title = line.split("|t|", 1)
            occurrences[pmid] += 1
            documents.append({"id": f"{split}:{pmid}:{occurrences[pmid]}", "pmid": pmid,
                              "split": split, "title": title, "annotations": []})
        elif "|a|" in line:
            pmid, abstract = line.split("|a|", 1)
            assert documents[-1]["pmid"] == pmid
            documents[-1].update(abstract=abstract, text=documents[-1]["title"] + " " + abstract)
        elif line.strip():
            pmid, start, end, surface, category, concept = line.split("\t")
            assert documents[-1]["pmid"] == pmid
            documents[-1]["annotations"].append({"start": int(start), "end": int(end),
                "source_surface": surface, "source_category": category, "concept_id": concept})
    return documents


def sentence_spans(document):
    params = PunktParameters()
    params.abbrev_types = {"al", "et", "fig", "figs", "e.g", "i.e", "vs", "dr", "mr",
                          "mrs", "prof", "no", "nos", "ref", "refs", "approx", "inc", "e", "c", "ec"}
    segmenter = PunktSentenceTokenizer(params)
    offset = len(document["title"]) + 1
    spans = [(0, offset - 1)]
    for start, end in segmenter.span_tokenize(document["abstract"]):
        start, end = start + offset, end + offset
        # PubTator contains spaced decimals and EC numbers: "1. 1. 1. 49".
        # This rule uses text only, never entity labels.
        if len(spans) > 1 and document["text"][start].isdigit():
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return spans


def prepare(root):
    root = Path(root)
    acquire(root)
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    datasets, all_documents, issues = {}, {}, []
    for split, stem in FILES.items():
        documents = parse_pubtator(root / "data/raw" / (stem + ".txt"), split)
        samples = []
        for doc in documents:
            text = doc["text"]
            for ann in doc["annotations"]:
                start, end = ann["start"], ann["end"]
                assert 0 <= start < end <= len(text)
                if text[start:end] != ann["source_surface"]:
                    issues.append({"document_id": doc["id"], "kind": "surface_mismatch",
                                   **ann, "offset_surface": text[start:end], "action": "offsets authoritative"})
                ann["model_start"], ann["model_end"] = start, end
                # Two train annotations end one letter before a complete word.
                # Repair the labels, not the tokenizer; retain original spans.
                if end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
                    assert split == "train" and (doc["pmid"], start, end) in {
                        ("10802668", 105, 131), ("2792129", 195, 240)}
                    while end < len(text) and text[end].isalnum():
                        end += 1
                    ann["model_end"] = end
                    issues.append({"document_id": doc["id"], "kind": "partial_word_end", **ann,
                                   "action": "extend train label to complete word", "corrected_surface": text[start:end]})
            spans = sentence_spans(doc)
            assert all(sum(a <= x["model_start"] < x["model_end"] <= b for a, b in spans) == 1
                       for x in doc["annotations"]), doc["id"]
            first_sample = len(samples)
            for sentence_index, (start, end) in enumerate(spans):
                offsets = [(m.start() + start, m.end() + start)
                           for m in re.finditer(r"\w+|[^\w\s]", text[start:end])]
                if not offsets:
                    continue
                tokens = [text[a:b] for a, b in offsets]
                tags = ["O"] * len(tokens)
                entities = []
                for ann in doc["annotations"]:
                    a, b = ann["model_start"], ann["model_end"]
                    if start <= a < b <= end:
                        indices = [i for i, (x, y) in enumerate(offsets) if a <= x < y <= b]
                        assert indices and offsets[indices[0]][0] == a and offsets[indices[-1]][1] == b
                        assert all(tags[i] == "O" for i in indices)
                        for j, i in enumerate(indices):
                            tags[i] = "B-Disease" if j == 0 else "I-Disease"
                        entities.append({**ann, "token_start": indices[0], "token_end": indices[-1] + 1,
                                         "text": text[a:b], "label": "Disease"})
                assert len(tokens) == len(tags)
                assert all(tag != "I-Disease" or (i > 0 and tags[i-1] in LABELS[1:])
                           for i, tag in enumerate(tags))
                assert tags.count("B-Disease") == len(entities)
                # Check every BIO entity reconstructs the canonical character span.
                for entity in entities:
                    a, b = entity["token_start"], entity["token_end"]
                    assert text[offsets[a][0]:offsets[b-1][1]] == entity["text"]
                samples.append({"id": f"{doc['id']}:s{sentence_index}", "document_id": doc["id"],
                    "pmid": doc["pmid"], "split": split, "sentence_index": sentence_index,
                    "sentence_start": start, "sentence_end": end, "text": text[start:end],
                    "tokens": tokens, "ner_tags": tags, "token_offsets": offsets, "entities": entities})
            assert [offset for sample in samples[first_sample:] for offset in sample["token_offsets"]] == [
                (m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]
        assert sum(len(x["entities"]) for x in samples) == sum(len(x["annotations"]) for x in documents)
        datasets[split], all_documents[split] = samples, documents
        write_jsonl(processed / f"{split}.jsonl", samples)
        write_jsonl(processed / f"{split}_documents.jsonl", documents)
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        assert not ({d["pmid"] for d in all_documents[a]} & {d["pmid"] for d in all_documents[b]})
        assert not ({d["text"] for d in all_documents[a]} & {d["text"] for d in all_documents[b]})
    report = {"status": "passed", "label2id": label2id, "annotation_issues": issues,
        "split_overlap_pmids": 0, "split_overlap_texts": 0, "dropped_annotations": 0,
        "all_bio_and_character_roundtrips_passed": True,
        "full_document_token_coverage": True,
        "document_records": {s: len(d) for s, d in all_documents.items()},
        "unique_pmids": {s: len({x['pmid'] for x in d}) for s, d in all_documents.items()},
        "duplicate_policy": "Retain official train records; repeated PMID 8528200 has distinct record IDs."}
    write_json(processed / "validation_report.json", report)
    write_json(processed / "label_mapping.json", {"label2id": label2id, "id2label": id2label})
    return datasets, all_documents, report
