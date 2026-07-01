"""CLI：为病例（真实 S1–S5 + 模拟 S6–S15）提取临床可解释生物标志物。

用法（仓库规范的直接路径调用形式）：
    python analysis/02_biomarkers/extract_biomarkers.py --subject S1
    python analysis/02_biomarkers/extract_biomarkers.py --subject S6
    python analysis/02_biomarkers/extract_biomarkers.py --subject S1 --trial 2_1
    python analysis/02_biomarkers/extract_biomarkers.py --subject S1 --all-trials
    python analysis/02_biomarkers/extract_biomarkers.py --cohort

输出（默认 analysis/02_biomarkers/out/）：
    json/S1.json            机器可读
    reports/S1.txt          中文人类可读报告（同时打印到屏幕）
    csv/cohort_biomarkers.csv  跨受试者汇总（--cohort 时写出）

支持全部受试者 S1–S15：真实 S1–S5 的 EEG 为 .bdf，模拟 S6–S15 的 EEG 为 .csv，
均由 manifest 枚举、io 层按格式分流加载。标签统一取自
patient_rehab_suggestions_15subjects.json。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# 本目录名 "02_biomarkers" 不是合法 Python 包名，故把目录直接加入 path，
# 以普通模块方式导入同级的 biomarkers / reasonableness。
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysis.common.manifest import (  # noqa: E402
    LABELS, find_trial, list_trials_for, subjects as _manifest_subjects,
)
from analysis.common.io import load_trial_raw  # noqa: E402
import biomarkers as bm  # noqa: E402

def ALL_SUBJECTS() -> tuple:
    """全部磁盘上有 trial 的受试者（S1–S5 真实 + S6–S15 模拟），按编号排序。"""
    return tuple(_manifest_subjects())


DEFAULT_OUT = _REPO_ROOT / "analysis" / "02_biomarkers" / "out"

_LABEL_KEYS = ("FMA_UE", "BI", "hand_tone", "hand_function")


# --- 计算 -------------------------------------------------------------------- #
def _subject_labels(subject_id: str) -> Dict[str, object]:
    lab = LABELS[subject_id.lstrip("S")]
    return {k: lab[k] for k in _LABEL_KEYS}


def compute_subject(subject_id: str) -> dict:
    """计算一个受试者的 trial 级 + 聚合级生物标志物。"""
    demo = bm.load_demographics(subject_id)
    affected = demo["affected_side"]
    trials = list_trials_for(subject_id)
    if not trials:
        raise SystemExit(f"未找到 {subject_id} 的 trial（需 BJH/EEG_new/*.bdf 或 *.csv）。")

    per_trial = []
    for ref in trials:
        raw = load_trial_raw(ref)
        vals = bm.compute_trial_biomarkers(raw, affected)
        per_trial.append({"trial": ref.name, "values": vals})

    agg = bm.aggregate_trials([t["values"] for t in per_trial])
    return {
        "subject": subject_id,
        "affected_side": affected,
        "demographics": demo,
        "labels": _subject_labels(subject_id),
        "n_trials": len(trials),
        "per_trial": per_trial,
        "aggregate": agg,
    }


def build_cohort(subject_ids=None) -> Dict[str, dict]:
    """计算全部受试者（S1–S15）的聚合生物标志物。"""
    if subject_ids is None:
        subject_ids = ALL_SUBJECTS()
    cohort = {}
    for sid in subject_ids:
        try:
            cohort[sid] = compute_subject(sid)
        except SystemExit:
            continue
    return cohort


# --- 输出 -------------------------------------------------------------------- #
def build_case_json(rec: dict, cohort_ids: List[str]) -> dict:
    return {
        "subject": rec["subject"],
        "n_trials": rec["n_trials"],
        "demographics": rec["demographics"],
        "labels": rec["labels"],
        "cohort": {"n_subjects": len(cohort_ids), "subjects": cohort_ids},
        "biomarkers": rec["aggregate"],
    }


def _fmt_val(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, float):
        if abs(v) < 1e-3 and v != 0:
            return f"{v:.2e}"
        return f"{v:.4g}"
    return str(v)


def render_report(case: dict) -> str:
    lab = case["labels"]
    demo = case["demographics"]
    lines = []
    lines.append("=" * 78)
    lines.append(f"病例生物标志物报告  {case['subject']}")
    lines.append(
        f"{case['subject']} · FMA={lab['FMA_UE']} · BI={lab['BI']} · "
        f"肌张力(MAS)={lab['hand_tone']} · 手功能(Brunnstrom)={lab['hand_function']} · "
        f"患侧={demo['affected_side']} · 年龄={demo.get('age','?')} · "
        f"病种={demo.get('disease','?')} · 病程={demo.get('days_post','?')}天 · "
        f"trial 数={case['n_trials']}"
    )
    lines.append("-" * 78)
    header = f"{'生物标志物':<34}{'值':>14}  {'有效trial数':>10}"
    lines.append(header)
    lines.append("-" * 78)
    for name, info in case["biomarkers"].items():
        lines.append(
            f"{name:<34}{_fmt_val(info['value']):>14}  {info['n_valid']:>10}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def write_cohort_csv(cohort: Dict[str, dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(bm.BIOMARKER_NAMES)
    header = (["subject", "n_trials"] + list(_LABEL_KEYS) + ["affected_side"]
              + [f"{n}__value" for n in names])
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for sid, rec in cohort.items():
            agg = rec["aggregate"]
            row = ([sid, rec["n_trials"]]
                   + [rec["labels"][k] for k in _LABEL_KEYS]
                   + [rec["affected_side"]]
                   + [_fmt_val(agg[n]["value"]) for n in names])
            w.writerow(row)


# --- 主流程 ------------------------------------------------------------------ #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="BJH 三模态临床生物标志物提取（S1–S15）")
    p.add_argument("--subject", help="受试者 ID，如 S1 或 S6（S1–S15）")
    p.add_argument("--trial", help="仅评估单个 trial，如 2_1")
    p.add_argument("--all-trials", action="store_true", help="额外输出每个 trial 的值")
    p.add_argument("--cohort", action="store_true",
                   help="计算全部 S1–S15 并写出 cohort_biomarkers.csv")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)

    all_subjects = ALL_SUBJECTS()
    if args.subject and args.subject not in all_subjects:
        print(f"错误：{args.subject} 在磁盘上无 trial。可用受试者：{', '.join(all_subjects)}。",
              file=sys.stderr)
        return 2

    # 建立队列（作为 JSON 中 cohort 字段的参考集）
    print("正在计算队列（S1–S15）…", file=sys.stderr)
    cohort = build_cohort()
    cohort_ids = list(cohort.keys())

    if args.cohort:
        csv_path = out_dir / "csv" / "cohort_biomarkers.csv"
        write_cohort_csv(cohort, csv_path)
        print(f"已写出队列汇总：{csv_path}", file=sys.stderr)
        if not args.subject:
            return 0

    if not args.subject:
        p.error("需指定 --subject（或仅用 --cohort 生成队列汇总）")

    sid = args.subject
    rec = cohort[sid]

    # 单 trial 模式：用该 trial 的值替换 aggregate
    if args.trial:
        ref = find_trial(sid, args.trial)
        raw = load_trial_raw(ref)
        vals = bm.compute_trial_biomarkers(raw, rec["affected_side"])
        single = {n: {"value": vals.get(n, float("nan")), "n_valid": 1}
                  for n in bm.BIOMARKER_NAMES}
        rec = {**rec, "n_trials": 1, "aggregate": single}

    case = build_case_json(rec, cohort_ids)
    if args.all_trials:
        case["per_trial"] = rec.get("per_trial", [])

    # 写 JSON
    json_path = out_dir / "json" / f"{sid}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(case, fh, ensure_ascii=False, indent=2, default=_json_default)

    # 写 + 打印报告
    report = render_report(case)
    report_path = out_dir / "reports" / f"{sid}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\n[已保存] JSON: {json_path}\n[已保存] 报告: {report_path}", file=sys.stderr)
    return 0


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(f"不可序列化：{type(o)}")


if __name__ == "__main__":
    raise SystemExit(main())
