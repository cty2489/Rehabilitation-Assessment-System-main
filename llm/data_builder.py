"""Build ChatML JSONL train/val/test files from patient_rehab_suggestions_*.json.

Each line is `{"messages": [system, user, assistant]}` ready for TRL's
SFTTrainer with `apply_chat_template`.

If the split file does not exist, a 3-fold subject-level split is created
in-place under splits/ and persisted so subsequent runs are deterministic.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from .prompts import build_chat_messages


_COURSE_RE = re.compile(r"病程[^，。]*?(?=[，。])")


def normalize_rehab_text(item: Dict[str, object]) -> str:
    """Rewrite rehab_text so 病程 is always rendered as `病程{days_post}天`.

    The raw data mixes 「X个月」、「N天」 and other forms; aligning every
    sample to `病程{days_post}天` lets the model learn one句法骨架 and
    boosts BLEU/ROUGE against the same normalized reference.
    """
    text = str(item.get("rehab_text", ""))
    days_post = int(item["demographics"]["days_post"])
    return _COURSE_RE.sub(f"病程{days_post}天", text, count=1)


REAL_SUBJECT_IDS: Tuple[str, ...] = ("1", "2", "3", "4", "5")


def _load_suggestions(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在，请先运行 `python simulate_data.py` 生成。"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["subjects"]


def make_split(
    suggestions: Dict[str, Dict[str, object]],
    n_folds: int = 3,
    seed: int = 1024,
) -> Dict[str, object]:
    """Build a subject-level k-fold split.

    Real subjects (S1..S5) are round-robined across folds so each fold's
    val_test_subjects contains 1-2 real patients.
    """
    real_ids = [sid for sid in REAL_SUBJECT_IDS if sid in suggestions]
    synth_ids = sorted(
        (sid for sid in suggestions if sid not in REAL_SUBJECT_IDS),
        key=lambda s: int(s),
    )

    rng = random.Random(seed)
    rng.shuffle(synth_ids)

    fold_val_test: List[List[str]] = [[] for _ in range(n_folds)]
    for i, sid in enumerate(real_ids):
        fold_val_test[i % n_folds].append(sid)
    for i, sid in enumerate(synth_ids):
        fold_val_test[i % n_folds].append(sid)

    all_ids = set(real_ids) | set(synth_ids)
    folds = []
    for k, val_test in enumerate(fold_val_test, start=1):
        train_subjects = sorted(all_ids - set(val_test), key=lambda s: int(s))
        val_test_sorted = sorted(val_test, key=lambda s: int(s))
        folds.append({
            "fold": k,
            "train_subjects": train_subjects,
            "val_test_subjects": val_test_sorted,
        })

    return {
        "schema_version": 1,
        "split_unit": "subject_id",
        "split_mode": "subject",
        "n_splits": n_folds,
        "seed": seed,
        "strategy": "real_roundrobin_then_synthetic_shuffle",
        "folds": folds,
    }


def _split_val_test(val_test_subjects: List[str]) -> Tuple[List[str], List[str]]:
    """Split val_test_subjects: even-indexed (sorted) -> val, odd-indexed -> test."""
    sorted_ids = sorted(val_test_subjects, key=lambda s: int(s))
    val = [sid for i, sid in enumerate(sorted_ids) if i % 2 == 0]
    test = [sid for i, sid in enumerate(sorted_ids) if i % 2 == 1]
    return val, test


def _get_fold(split_payload: Dict[str, object], fold: int) -> Dict[str, object]:
    for f in split_payload["folds"]:
        if int(f["fold"]) == int(fold):
            return f
    raise KeyError(f"Fold {fold} not found in split file")


def _write_jsonl(records: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def build_records(
    suggestions: Dict[str, Dict[str, object]],
    subject_ids: List[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for sid in subject_ids:
        item = suggestions.get(sid)
        if item is None:
            continue
        messages = build_chat_messages(
            subject_id=sid,
            demographics=item["demographics"],
            labels=item["labels"],
            rehab_text=normalize_rehab_text(item),
        )
        rows.append({"subject_id": sid, "messages": messages, "source": item.get("source", "")})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ChatML JSONL from rehab suggestions.")
    ap.add_argument("--suggestions", type=Path,
                    default=Path("patient_rehab_suggestions_100subjects.json"),
                    help="Path to patient_rehab_suggestions_*.json")
    ap.add_argument("--split", type=Path,
                    default=Path("splits/3fold_patient_split_llm_100subjects.json"),
                    help="Path to (or destination for) the split JSON.")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--n-folds", type=int, default=3,
                    help="Used only when --split does not exist yet.")
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory, will contain train/val/test.jsonl")
    args = ap.parse_args()

    suggestions = _load_suggestions(args.suggestions)

    if not args.split.exists():
        print(f"[data_builder] split 文件不存在，生成新的 3-fold split → {args.split}")
        split_payload = make_split(suggestions, n_folds=args.n_folds, seed=args.seed)
        args.split.parent.mkdir(parents=True, exist_ok=True)
        args.split.write_text(
            json.dumps(split_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        split_payload = json.loads(args.split.read_text(encoding="utf-8"))

    fold = _get_fold(split_payload, args.fold)
    train_subjects: List[str] = list(map(str, fold["train_subjects"]))
    val_subjects, test_subjects = _split_val_test(list(map(str, fold["val_test_subjects"])))

    train_records = build_records(suggestions, train_subjects)
    val_records = build_records(suggestions, val_subjects)
    test_records = build_records(suggestions, test_subjects)

    _write_jsonl(train_records, args.out / "train.jsonl")
    _write_jsonl(val_records, args.out / "val.jsonl")
    _write_jsonl(test_records, args.out / "test.jsonl")

    print(
        f"[data_builder] fold={args.fold} | "
        f"train={len(train_records)}  val={len(val_records)}  test={len(test_records)}"
    )
    print(f"  train subjects: {train_subjects}")
    print(f"  val subjects:   {val_subjects}")
    print(f"  test subjects:  {test_subjects}")
    print(f"  → {args.out}/{{train,val,test}}.jsonl")


if __name__ == "__main__":
    main()
