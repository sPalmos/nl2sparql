"""Revision analyses that read stored runs only (no model calls).

Run from the revision worktree (nl2sparql-revision); the paper's runs are read
from --paper-root (default ..\\NL-to-SPARQL-V4). Prints numbers, labels and
question IDs only, never question text, queries or answers.

    python ..\\revision_tools\\revision_analyses.py b1    # Claude Opus 5.5 vs qwen2.5-coder:32b and Gemini
    python ..\\revision_tools\\revision_analyses.py b4a   # temperature 0.6 vs 0 (qwen2.5-coder:32b, 6 ontologies)
    python ..\\revision_tools\\revision_analyses.py b5    # held-out eNOVATION set vs the original 40
    python ..\\revision_tools\\revision_analyses.py t1    # empty-gold questions
    python ..\\revision_tools\\revision_analyses.py t2    # deepseek-r1:32b vs qwen2.5-coder:32b on eNOVATION and CK25
    python ..\\revision_tools\\revision_analyses.py t3 [--model qwen3:8b]   # reasoning output: length cut-offs, no query, repetition
    python ..\\revision_tools\\revision_analyses.py t4    # onboarding times from the package logs

Run selection mirrors the paper (analysis/build_headline_tables.py): the newest
reference-configuration run per (ontology, model). Scoring uses the paper's code.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "analysis"))
import build_headline_tables as bht  # noqa: E402
from difficulty_levels import level, operations, triple_patterns  # noqa: E402
from error_analysis import outcome  # noqa: E402

ABLATION_ONTOLOGIES = ["bestiary", "ck25", "enovation", "spider-dog-kennels",
                       "spider-student-transcripts-tracking", "spider-world-1"]
GEMINI = "api:gemini-3.1-flash-lite"
HELDOUT_DATASETS = {"eNOVATION_heldout_dataset"}
PAPER_ROOT = Path("..") / "NL-to-SPARQL-V4"
LEVEL_ARGS = SimpleNamespace(simple_p=2, simple_o=1, complex_p=5, complex_o=3)
OUTCOME_NAMES = {"E1": "invalid query", "E2": "endpoint error", "E3": "empty result",
                 "E4": "wrong rows", "E5": "too few rows", "E6": "too many rows", "E7": "partial overlap"}


# ---------------------------------------------------------------- helpers

def read(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return data if data.get("results") else None


def all_runs(packages: Path):
    for path in sorted(packages.rglob("results.json")):
        data = read(path)
        if data is not None:
            yield path, data


def overrides(path: Path) -> dict:
    cfg = path.parent / "run_config.json"
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get("llm_overrides") or {}
    except Exception:  # noqa: BLE001
        return {}


def main_cells(packages: Path) -> dict[tuple[str, str], tuple[Path, dict]]:
    """Newest reference-configuration run per (ontology, model), as in the paper."""
    chosen: dict = {}
    for path, data in all_runs(packages):
        if not bht.matches_reference(data.get("pipeline_config") or {}):
            continue
        if data.get("dataset_name", "") in HELDOUT_DATASETS:
            continue
        key = (bht.ontology_key(data.get("package_dir", "")), data.get("model_name", ""))
        stamp = data.get("timestamp", "")
        if key not in chosen or stamp > chosen[key][0]:
            chosen[key] = (stamp, path, data)
    return {k: (p, d) for k, (_, p, d) in chosen.items()}


def per_question(data: dict) -> dict[str, tuple[bool, bool]]:
    """question_id -> (lenient, strict) for scored (non-empty gold) questions."""
    out = {}
    aliases = prefixes = None
    for q in data["results"]:
        if not q.get("is_scored", True) or not q.get("gold_answers"):
            continue
        lenient = bool((q.get("comparison") or {}).get("exact_match"))
        strict = False
        if lenient and q.get("final_answers") is not None:
            if aliases is None:
                aliases, prefixes = bht.package_context(data["package_dir"])
            strict = (bht.normalize_result_set(q["gold_answers"], prefixes, aliases)
                      == bht.normalize_result_set(q["final_answers"], prefixes, aliases))
        out[q["question_id"]] = (lenient, strict)
    return out


def ea(pq: dict, rule: int, ids=None) -> float:
    ids = list(pq) if ids is None else ids
    return 100.0 * sum(pq[i][rule] for i in ids) / len(ids) if ids else float("nan")


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def mean_latency(data: dict) -> float:
    lat = [q.get("total_latency_ms", 0.0) for q in data["results"] if q.get("gold_answers")]
    return statistics.mean(lat) if lat else float("nan")


def empty_triggered(q: dict) -> bool:
    """An attempt that executed without error but was not the last one can only
    have been followed by a correction because its result was empty."""
    its = q.get("iterations") or []
    return any(it.get("execution_status") == "EXECUTION_OK" for it in its[:-1])


def first_attempt_empty(q: dict) -> bool:
    its = q.get("iterations") or []
    if not its or its[0].get("execution_status") != "EXECUTION_OK":
        return False
    return len(its) > 1 or q.get("final_answers") == []


# ---------------------------------------------------------------- B4a

def b4a(rev: Path, paper: Path, model: str = "qwen2.5-coder:32b", temp: str = "0.6") -> None:
    ref = main_cells(paper)
    hot: dict[str, list] = defaultdict(list)
    for path, data in all_runs(rev):
        if data.get("model_name") != model or not bht.matches_reference(data.get("pipeline_config") or {}):
            continue
        if overrides(path).get("NL2SPARQL_TEMPERATURE") != temp or data.get("dataset_name", "") in HELDOUT_DATASETS:
            continue
        hot[bht.ontology_key(data["package_dir"])].append((data.get("timestamp", ""), path, data))

    print(f"B4a: {model}, T = {temp} vs the paper's T = 0 run (main-table cell)\n")
    print(f"{'ontology':38s} {'n':>4s}  {'T=0 L/S':>11s}  " + "  ".join(f"{'run'+str(i+1)+' L/S':>11s}" for i in range(2))
          + "  changed vs T=0 (run1, run2, either)  run1 vs run2")
    rows = []
    pooled = Counter()
    for ont in ABLATION_ONTOLOGIES:
        runs = sorted(hot.get(ont, []), key=lambda x: x[0])
        if (ont, model) not in ref or len(runs) < 2:
            print(f"{ont:38s} missing (reference: {(ont, model) in ref}, T={temp} runs: {len(runs)})")
            continue
        r0 = per_question(ref[(ont, model)][1])
        rs = [per_question(d) for _, _, d in runs[-2:]]
        ids = sorted(set(r0) & set(rs[0]) & set(rs[1]))
        ch = [[i for i in ids if rs[k][i][0] != r0[i][0]] for k in range(2)]
        either = {i for k in range(2) for i in ch[k]}
        flips = [i for i in ids if rs[0][i][0] != rs[1][i][0]]
        cells = [(ea(r0, 0, ids), ea(r0, 1, ids))] + [(ea(r, 0, ids), ea(r, 1, ids)) for r in rs]
        rows.append((ont, len(ids), cells))
        print(f"{ont:38s} {len(ids):4d}  " + "  ".join(f"{l:5.1f}/{s:5.1f}" for l, s in cells)
              + f"  {len(ch[0]):3d} {len(ch[1]):3d} {len(either):3d}            {len(flips):3d}")
        for k in range(2):
            pooled[f"b{k}"] += sum(1 for i in ids if r0[i][0] and not rs[k][i][0])
            pooled[f"c{k}"] += sum(1 for i in ids if rs[k][i][0] and not r0[i][0])
        pooled["n"] += len(ids)
        pooled["either"] += len(either)
        pooled["flips"] += len(flips)
        pooled["lat0"] += mean_latency(ref[(ont, model)][1]) * len(ids)
        for k, (_, _, d) in enumerate(runs[-2:]):
            pooled[f"lat{k+1}"] += mean_latency(d) * len(ids)
    if not rows:
        return
    macro = [[statistics.mean(r[2][j][rule] for r in rows) for rule in (0, 1)] for j in range(3)]
    print(f"\nmacro mean over {len(rows)} ontologies (lenient / strict): T=0 {macro[0][0]:.1f} / {macro[0][1]:.1f}; "
          f"run1 {macro[1][0]:.1f} / {macro[1][1]:.1f}; run2 {macro[2][0]:.1f} / {macro[2][1]:.1f}; "
          f"mean of runs {(macro[1][0] + macro[2][0]) / 2:.1f} / {(macro[1][1] + macro[2][1]) / 2:.1f}")
    n = pooled["n"]
    for k in range(2):
        b, c = pooled[f"b{k}"], pooled[f"c{k}"]
        print(f"run{k+1} vs T=0 (lenient, pooled {n}): correct only at T=0 {b}, only at T={temp} {c}, "
              f"changed {b + c} ({100 * (b + c) / n:.1f}%), exact McNemar p = {mcnemar(b, c):.3g}")
    print(f"questions that changed outcome in at least one run: {pooled['either']} ({100 * pooled['either'] / n:.1f}%)")
    print(f"run1 vs run2: {pooled['flips']} questions differ ({100 * pooled['flips'] / n:.1f}%)")
    print(f"mean latency (question-weighted): T=0 {pooled['lat0'] / n:,.0f} ms, run1 {pooled['lat1'] / n:,.0f} ms, run2 {pooled['lat2'] / n:,.0f} ms")
    for k, (stamp, path, _) in enumerate(sorted(hot[ABLATION_ONTOLOGIES[0]], key=lambda x: x[0])[-2:]):
        print(f"(run{k+1} e.g. {path.parent.name}, overrides {overrides(path)})")


# ---------------------------------------------------------------- T1

def t1(paper: Path) -> None:
    cells = main_cells(paper)
    gemini: dict[str, list] = defaultdict(list)
    for path, data in all_runs(paper):
        if data.get("model_name") == GEMINI and bht.matches_reference(data.get("pipeline_config") or {}):
            gemini[bht.ontology_key(data["package_dir"])].append((data.get("timestamp", ""), data))

    def tally(runs: list[dict]) -> Counter:
        c = Counter()
        for data in runs:
            for q in data["results"]:
                if q.get("gold_answers"):
                    continue
                c["questions"] += 1
                fa = q.get("final_answers")
                c["final empty (correct)" if fa == [] else "final non-empty" if fa else "no answer (failed)"] += 1
                c["first attempt empty"] += first_attempt_empty(q)
                c["empty-result correction triggered"] += empty_triggered(q)
                c["first attempt empty -> final non-empty"] += first_attempt_empty(q) and bool(fa)
        return c

    keys = ["questions", "final empty (correct)", "final non-empty", "no answer (failed)",
            "first attempt empty", "empty-result correction triggered", "first attempt empty -> final non-empty"]
    print("T1: questions with an empty gold answer (excluded from scoring), main-table runs\n")
    print(f"{'model':28s} " + " | ".join(keys))
    per_ont = Counter()
    for model in bht.MODEL_ORDER:
        runs = [d for (o, m), (_, d) in cells.items() if m == model and o not in bht.EXCLUDED_ONTOLOGIES]
        c = tally(runs)
        print(f"{model:28s} " + " | ".join(str(c[k]) for k in keys))
        if model == "qwen2.5-coder:32b":
            for d in runs:
                per_ont[bht.ontology_key(d["package_dir"])] += sum(1 for q in d["results"] if not q.get("gold_answers"))
    for k_run in (0, 1):
        runs = [sorted(v, key=lambda x: x[0])[-2:][k_run][1] for o, v in gemini.items()
                if o not in bht.EXCLUDED_ONTOLOGIES and len(v) > k_run]
        c = tally(runs)
        print(f"{'Gemini run ' + str(k_run + 1):28s} " + " | ".join(str(c[k]) for k in keys))
    print("\nempty-gold questions per ontology:", dict(sorted((o, n) for o, n in per_ont.items() if n)))
    # cost on the scored questions, for context
    print("\nscored questions: correction triggered by an empty result, and outcome (lenient)")
    for model in bht.MODEL_ORDER:
        runs = [d for (o, m), (_, d) in cells.items() if m == model and o not in bht.EXCLUDED_ONTOLOGIES]
        trig = fixed = 0
        for d in runs:
            for q in d["results"]:
                if q.get("gold_answers") and empty_triggered(q):
                    trig += 1
                    fixed += bool((q.get("comparison") or {}).get("exact_match"))
        print(f"  {model:28s} triggered {trig}, ended correct {fixed}")


# ---------------------------------------------------------------- T2

def prompt_tokens(q: dict) -> int | None:
    trace = q.get("readable_trace_path")
    if not trace:
        return None
    p = Path(trace)
    if not p.exists() and not p.is_absolute():
        p = PAPER_ROOT / trace          # relative paths are relative to the paper repository
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    start, end = text.find("GENERATION PROMPT"), text.find("RETRIEVED ABOX CONTEXT")
    if start < 0:
        return None
    return round(len(text[start + len("GENERATION PROMPT"): end if end > start else None].strip()) / 4)


def t2(paper: Path) -> None:
    cells = main_cells(paper)
    a_model, b_model = "deepseek-r1:32b", "qwen2.5-coder:32b"
    print(f"T2: {a_model} (D) vs {b_model} (Q), main-table runs; prompt tokens = first-attempt prompt chars / 4\n")
    for ont in ("enovation", "ck25"):
        if (ont, a_model) not in cells or (ont, b_model) not in cells:
            print(f"{ont}: missing run")
            continue
        da, db = cells[(ont, a_model)][1], cells[(ont, b_model)][1]
        pa, pb = per_question(da), per_question(db)
        qa = {q["question_id"]: q for q in da["results"]}
        qb = {q["question_id"]: q for q in db["results"]}
        ids = sorted(set(pa) & set(pb))
        print(f"== {ont}: n {len(ids)}; EA lenient D {ea(pa, 0, ids):.1f} / Q {ea(pb, 0, ids):.1f}; "
              f"strict D {ea(pa, 1, ids):.1f} / Q {ea(pb, 1, ids):.1f}")
        for rule, name in ((0, "lenient"), (1, "strict")):
            d_only = [i for i in ids if pa[i][rule] and not pb[i][rule]]
            q_only = [i for i in ids if pb[i][rule] and not pa[i][rule]]
            print(f"   {name}: only D correct {len(d_only)}, only Q correct {len(q_only)}, "
                  f"exact McNemar p = {mcnemar(len(d_only), len(q_only)):.3g}")
        print(f"   {'question':9s} {'correct':8s} {'wrong model outcome':24s} {'level':8s} {'P':>3s} {'O':>3s} {'prompt tok':>10s}  attempts D/Q")
        for i in ids:
            if pa[i][0] == pb[i][0]:
                continue
            winner = "D" if pa[i][0] else "Q"
            loser = qb[i] if winner == "D" else qa[i]
            cat = outcome(loser)
            gold = qa[i].get("gold_sparql") or ""
            try:
                p = triple_patterns(gold)
                o = len(operations(gold))
                lv = level(p, o, LEVEL_ARGS)
            except Exception:  # noqa: BLE001
                p, o, lv = -1, -1, "unparsed"
            tok = prompt_tokens(qb[i]) or prompt_tokens(qa[i])
            print(f"   {i:9s} {winner:8s} {cat + ' ' + OUTCOME_NAMES[cat]:24s} {lv:8s} {p:3d} {o:3d} {str(tok):>10s}  "
                  f"{qa[i].get('total_iterations')}/{qb[i].get('total_iterations')}")
        sample = next((q.get("readable_trace_path") for q in db["results"] if q.get("readable_trace_path")), None)
        print(f"   (trace path form, for checking: {sample})")
        print()


# ---------------------------------------------------------------- T3

def _repetitive(text: str) -> bool:
    """Signs of a generation loop: many identical non-trivial lines, or a
    200-character block from the end that occurs at least 3 times."""
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) >= 20]
    if len(lines) >= 10:
        top = Counter(lines).most_common(1)[0][1]
        if top >= 5 and top / len(lines) >= 0.2:
            return True
    tail = text[-1200:-1000] if len(text) > 1200 else ""
    return bool(tail) and text.count(tail) >= 3


def t3(rev: Path, paper: Path, model: str, temperature: str | None = None) -> None:
    from app.domain.runtime import query_generation as qg  # noqa: PLC0415

    def recoverable(text: str) -> bool:
        stripped = qg._strip_think_blocks(text or "")
        return bool(qg._extract_fenced_block(stripped) or qg._QUERY_START.search(stripped))

    import calendar  # noqa: PLC0415
    # every run of this model in the revision folder marks a time boundary;
    # run timestamps are UTC minutes, the log has epoch seconds
    every = []
    for path, data in all_runs(rev):
        if data.get("model_name") != model:
            continue
        start = calendar.timegm(datetime.strptime(data["timestamp"], "%Y%m%d-%H%M").timetuple())
        selected = (bht.matches_reference(data.get("pipeline_config") or {})
                    and (overrides(path).get("NL2SPARQL_TEMPERATURE") or "") == (temperature or "")
                    and data.get("dataset_name", "") not in HELDOUT_DATASETS)
        every.append((start, path, data, selected))
    every.sort(key=lambda r: r[0])
    runs = [(start, path, data) for start, path, data, sel in every if sel]
    if not runs:
        print(f"no matching {model} runs in the revision folder")
        return
    calls = [json.loads(l) for l in open(Path("logs") / "ollama_usage.jsonl", encoding="utf-8")]
    want_t = float(temperature) if temperature else 0.0
    calls = [c for c in calls if c.get("model") == model
             and abs(float((c.get("options") or {}).get("temperature", 0.0)) - want_t) < 1e-9]
    per_run: dict[int, list] = defaultdict(list)
    index_of = {id(d): i for i, (_, _, d) in enumerate(runs)}
    for c in calls:
        owner = None
        for start, _, data, sel in every:
            if c["t"] >= start - 60:
                owner = (data, sel)
        if owner and owner[1]:
            per_run[index_of[id(owner[0])]].append(c)
    paper_cells = main_cells(paper)

    setting = f"T = {temperature} (override)" if temperature else "paper settings: T = 0"
    print(f"T3: {model}, runs with logging ({setting}, num_predict 8192, num_ctx 32768)\n")
    print(f"{'ontology':38s} {'calls':>5s} {'len-cut':>7s} {'%':>5s} {'noquery':>7s} {'%':>5s} {'repeat(cut)':>11s} "
          f"{'q':>4s} {'q-cut':>5s} {'q-noSPARQL':>10s}  {'EA now/paper':>13s} {'changed':>7s}")
    tot = Counter()
    raw_seen = False
    for i, (start, path, data) in enumerate(runs):
        ont = bht.ontology_key(data["package_dir"])
        cs = per_run.get(i, [])
        cut = [c for c in cs if c.get("done_reason") == "length" or (c.get("eval_count") or 0) >= 8192]
        has_raw = [c for c in cs if "response" in c]
        raw_seen = raw_seen or bool(has_raw)
        noq = [c for c in has_raw if not recoverable(c.get("response", ""))]
        rep = [c for c in cut if "response" in c and _repetitive((c.get("thinking") or "") + (c.get("response") or ""))]
        # question level: map calls to questions by the question text at the end of the prompt
        qs = data["results"]
        tails = {q["question_id"]: q.get("nl_question", "") for q in qs}
        cut_q = {qid for c in cut for qid, txt in tails.items() if txt and c.get("prompt_tail", "").rstrip().endswith(txt.strip())}
        scored = [q for q in qs if q.get("gold_answers")]
        no_sparql = sum(1 for q in scored if not (q.get("final_sparql") or "").strip())
        now = per_question(data)
        ref = per_question(paper_cells[(ont, model)][1]) if (ont, model) in paper_cells else {}
        ids = sorted(set(now) & set(ref))
        changed = sum(now[q][0] != ref[q][0] for q in ids)
        ea_now = ea(now, 0, ids) if ids else ea(now, 0)
        ea_ref = ea(ref, 0, ids) if ids else float("nan")
        print(f"{ont:38s} {len(cs):5d} {len(cut):7d} {100 * len(cut) / max(1, len(cs)):5.1f} {len(noq):7d} "
              f"{100 * len(noq) / max(1, len(has_raw)):5.1f} {len(rep):5d}/{len(cut):<5d} {len(scored):4d} {len(cut_q):5d} {no_sparql:10d}  "
              f"{ea_now:5.1f}/{ea_ref:5.1f}   {changed:5d}")
        if ont in bht.EXCLUDED_ONTOLOGIES:
            continue
        for k, v in (("calls", len(cs)), ("cut", len(cut)), ("raw", len(has_raw)), ("noq", len(noq)), ("rep", len(rep)),
                     ("q", len(scored)), ("qcut", len(cut_q)), ("nosparql", no_sparql), ("changed", changed), ("ids", len(ids))):
            tot[k] += v
        tot["ea_now"] += ea_now
        tot["ea_ref"] += ea_ref
        tot["onts"] += 1
    print(f"\nTOTAL over {tot['onts']} reported ontologies (real-estate excluded):")
    print(f"  calls {tot['calls']}; cut at the 8,192-token limit {tot['cut']} ({100 * tot['cut'] / max(1, tot['calls']):.1f}%)")
    if raw_seen:
        print(f"  calls with no recoverable query {tot['noq']} ({100 * tot['noq'] / max(1, tot['raw']):.1f}% of {tot['raw']} logged outputs)")
        print(f"  cut calls with signs of repeated text: {tot['rep']} of {tot['cut']}")
    else:
        print("  raw outputs were not logged (NL2SPARQL_LOG_RAW was not set), so no recoverable-query / repetition check")
    print(f"  scored questions {tot['q']}; with at least one cut call {tot['qcut']} ({100 * tot['qcut'] / max(1, tot['q']):.1f}%); "
          f"final query empty {tot['nosparql']} ({100 * tot['nosparql'] / max(1, tot['q']):.1f}%)")
    print(f"  macro EA lenient: re-run {tot['ea_now'] / max(1, tot['onts']):.1f}, paper run {tot['ea_ref'] / max(1, tot['onts']):.1f}; "
          f"questions with a different outcome {tot['changed']} of {tot['ids']} ({100 * tot['changed'] / max(1, tot['ids']):.1f}%)")
    ec = [c.get("eval_count") or 0 for i in per_run for c in per_run[i]]
    if ec:
        print(f"  output tokens per call: median {int(statistics.median(ec))}, 90th pct {sorted(ec)[int(0.9 * (len(ec) - 1))]}, max {max(ec)}")

# ---------------------------------------------------------------- B5

def _levels(results: list[dict]) -> dict[str, str]:
    out = {}
    for q in results:
        gold = q.get("gold_sparql") or ""
        try:
            out[q["question_id"]] = level(triple_patterns(gold), len(operations(gold)), LEVEL_ARGS)
        except Exception:  # noqa: BLE001
            out[q["question_id"]] = "unparsed"
    return out


def _boot(values: list[int], reps: int = 10000, seed: int = 0) -> tuple[float, float]:
    import random  # noqa: PLC0415
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(reps))
    return 100 * stats[int(0.025 * reps)], 100 * stats[int(0.975 * reps) - 1]


def _boot_diff(a: list[int], b: list[int], reps: int = 10000, seed: int = 0) -> tuple[float, float]:
    import random  # noqa: PLC0415
    rng = random.Random(seed)
    stats = sorted(sum(rng.choice(a) for _ in a) / len(a) - sum(rng.choice(b) for _ in b) / len(b)
                   for _ in range(reps))
    return 100 * stats[int(0.025 * reps)], 100 * stats[int(0.975 * reps) - 1]


def b5(rev: Path, paper: Path) -> None:
    held: dict[str, tuple] = {}
    for path, data in all_runs(rev):
        if data.get("dataset_name", "") not in HELDOUT_DATASETS:
            continue
        if not bht.matches_reference(data.get("pipeline_config") or {}) or overrides(path).get("NL2SPARQL_TEMPERATURE"):
            continue
        m = data.get("model_name", "")
        if m not in held or data.get("timestamp", "") > held[m][0]:
            held[m] = (data.get("timestamp", ""), path, data)
    cells = main_cells(paper)
    print("B5: held-out eNOVATION set (40 questions) vs the original 40 questions (paper's main-table runs)")
    print("EA in %, empty-gold excluded; 95% CI = bootstrap over questions (10,000 resamples, seed 0);")
    print("reweighted = held-out EA per level, weighted by the original set's level mix\n")
    orig_mix = None
    rows = []
    for model in bht.MODEL_ORDER:
        if model not in held or ("enovation", model) not in cells:
            print(f"{model}: missing (held-out run: {model in held}, original run: {('enovation', model) in cells})")
            continue
        hd, od = held[model][2], cells[("enovation", model)][1]
        hp, op = per_question(hd), per_question(od)
        hl, ol = _levels(hd["results"]), _levels(od["results"])
        if orig_mix is None:
            orig_mix = Counter(ol[q] for q in op)
            held_mix = Counter(hl[q] for q in hp)
            print(f"level mix: original {dict(sorted(orig_mix.items()))}, held-out {dict(sorted(held_mix.items()))}\n")
        hlen = [int(v[0]) for v in hp.values()]
        hstr = [int(v[1]) for v in hp.values()]
        olen = [int(v[0]) for v in op.values()]
        ostr = [int(v[1]) for v in op.values()]
        per_level = {}
        for lv in ("simple", "medium", "complex"):
            ids = [q for q in hp if hl[q] == lv]
            per_level[lv] = (100 * sum(hp[q][0] for q in ids) / len(ids)) if ids else float("nan")
        total = sum(orig_mix.values())
        reweighted = sum(per_level[lv] * orig_mix.get(lv, 0) for lv in per_level) / total
        cats = Counter(outcome(q) for q in hd["results"] if q.get("gold_answers")
                       and not (q.get("comparison") or {}).get("exact_match"))
        lo, hi = _boot(hlen)
        dlo, dhi = _boot_diff(hlen, olen)
        rows.append(model)
        print(f"{model}")
        print(f"   held-out: EA lenient {100 * sum(hlen) / len(hlen):5.1f} [{lo:.1f}, {hi:.1f}]  strict {100 * sum(hstr) / len(hstr):5.1f}  "
              f"reweighted {reweighted:5.1f}  by level S/M/C {per_level['simple']:.1f} / {per_level['medium']:.1f} / {per_level['complex']:.1f}")
        print(f"   original: EA lenient {100 * sum(olen) / len(olen):5.1f}  strict {100 * sum(ostr) / len(ostr):5.1f}")
        print(f"   held-out minus original (lenient): {100 * (sum(hlen) / len(hlen) - sum(olen) / len(olen)):+.1f} pp, 95% CI [{dlo:+.1f}, {dhi:+.1f}]")
        print(f"   held-out failures by outcome: " + ", ".join(f"{k} {OUTCOME_NAMES[k]} {cats[k]}" for k in sorted(cats)))
        print(f"   mean latency {mean_latency(hd):,.0f} ms; run {held[model][1].parent.name}")

# ---------------------------------------------------------------- B1

OPUS = "api:claude-opus-5-5"
LOCAL_REF = "qwen2.5-coder:32b"
PRICES = {"claude-opus-5-5": (4.0, 20.0)}          # USD per million input / output tokens


def _system_stats(runs: dict[str, dict]) -> dict:
    """runs: ontology -> results data. Paper conventions: macro over ontologies, pooled over questions."""
    per_ont, pooled = {}, Counter()
    for ont, data in runs.items():
        pq = per_question(data)
        qs = {q["question_id"]: q for q in data["results"]}
        ids = list(pq)
        n = len(ids)
        first = sum(1 for i in ids if pq[i][0] and qs[i].get("total_iterations", 0) <= 1)
        f1 = statistics.mean(float((qs[i].get("comparison") or {}).get("f1", 0.0) or 0.0) for i in ids)
        per_ont[ont] = {"n": n, "len": ea(pq, 0), "str": ea(pq, 1), "first": 100 * first / n, "f1": f1,
                        "lat": mean_latency(data), "pq": pq}
        pooled["n"] += n
        pooled["len"] += sum(pq[i][0] for i in ids)
        pooled["str"] += sum(pq[i][1] for i in ids)
    m = lambda k: statistics.mean(v[k] for v in per_ont.values())  # noqa: E731
    return {"per_ont": per_ont, "macro_len": m("len"), "macro_str": m("str"), "first": m("first"),
            "f1": m("f1"), "lat": m("lat"), "pooled_len": 100 * pooled["len"] / pooled["n"],
            "pooled_str": 100 * pooled["str"] / pooled["n"], "n": pooled["n"]}


def _paired(a: dict, b: dict, rule: int, reps: int = 10000, seed: int = 0):
    """a, b: _system_stats results on the same ontologies. Macro delta a - b, stratified bootstrap CI, McNemar."""
    import random  # noqa: PLC0415
    onts = sorted(set(a["per_ont"]) & set(b["per_ont"]))
    data = {}
    bb = cc = 0
    for o in onts:
        pa, pb = a["per_ont"][o]["pq"], b["per_ont"][o]["pq"]
        ids = sorted(set(pa) & set(pb))
        data[o] = [(int(pa[i][rule]), int(pb[i][rule])) for i in ids]
        bb += sum(1 for x, y in data[o] if x and not y)
        cc += sum(1 for x, y in data[o] if y and not x)
    delta = statistics.mean(100 * statistics.mean(x - y for x, y in data[o]) for o in onts)
    rng = random.Random(seed)
    stats = []
    for _ in range(reps):
        tot = 0.0
        for o in onts:
            rows = data[o]
            tot += 100 * sum(x - y for x, y in (rng.choice(rows) for _ in rows)) / len(rows)
        stats.append(tot / len(onts))
    stats.sort()
    return delta, stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1], bb, cc, mcnemar(bb, cc)


def b1(rev: Path, paper: Path) -> None:
    excluded = set(bht.EXCLUDED_ONTOLOGIES) | {"enovation"}
    opus: dict[str, tuple] = {}
    for path, data in all_runs(rev):
        if data.get("model_name") != OPUS or not bht.matches_reference(data.get("pipeline_config") or {}):
            continue
        o = bht.ontology_key(data["package_dir"])
        if o in excluded:
            continue
        if o not in opus or data.get("timestamp", "") > opus[o][0]:
            opus[o] = (data.get("timestamp", ""), data)
    cells = main_cells(paper)
    gem: dict[str, list] = defaultdict(list)
    for path, data in all_runs(paper):
        if data.get("model_name") == GEMINI and bht.matches_reference(data.get("pipeline_config") or {}):
            o = bht.ontology_key(data["package_dir"])
            if o not in excluded:
                gem[o].append((data.get("timestamp", ""), data))
    onts = sorted(o for o in opus if (o, LOCAL_REF) in cells)
    missing = sorted(set(o for (o, m) in cells if m == LOCAL_REF) - excluded - set(onts))
    print(f"B1: Claude Opus 5.5 ({OPUS}), one run, vs {LOCAL_REF} (main-table runs) and Gemini 3.1 Flash-Lite (paper runs)")
    print(f"ontologies: {len(onts)}" + (f"; MISSING Opus runs for: {missing}" if missing else "") + "\n")
    S_opus = _system_stats({o: opus[o][1] for o in onts})
    S_ref = _system_stats({o: cells[(o, LOCAL_REF)][1] for o in onts})
    gem_runs = [{o: sorted(gem[o], key=lambda x: x[0])[-2:][k][1] for o in onts if len(gem[o]) >= 2} for k in (0, 1)]
    S_gem = [_system_stats(r) for r in gem_runs if len(r) == len(onts)]

    def line(name, S):
        gain = S["macro_len"] - S["first"]
        print(f"{name:34s} {S['macro_len']:6.1f} {S['macro_str']:6.1f} {S['pooled_len']:7.1f} {S['pooled_str']:7.1f} "
              f"{S['f1']:6.3f} {S['first']:7.1f} {gain:6.1f} {S['lat']:9,.0f}")
    print(f"{'system':34s} {'EA-L':>6s} {'EA-S':>6s} {'pool-L':>7s} {'pool-S':>7s} {'F1':>6s} {'1st-att':>7s} {'gain':>6s} {'lat ms':>9s}")
    line("Claude Opus 5.5", S_opus)
    line(LOCAL_REF + " (same ontologies)", S_ref)
    for k, S in enumerate(S_gem):
        line(f"Gemini 3.1 Flash-Lite run {k + 1}", S)
    print(f"(pooled over {S_opus['n']} questions)\n")

    print("paired tests (macro delta, 95% stratified bootstrap CI; McNemar exact on pooled questions):")
    for rule, name in ((0, "lenient"), (1, "strict")):
        d, lo, hi, b_, c_, p_ = _paired(S_opus, S_ref, rule)
        print(f"  Opus - {LOCAL_REF} [{name}]: {d:+.1f} pp [{lo:+.1f}, {hi:+.1f}]; only Opus {b_}, only qwen {c_}; p = {p_:.3g}")
        for k, S in enumerate(S_gem):
            d, lo, hi, b_, c_, p_ = _paired(S_opus, S, rule)
            print(f"  Opus - Gemini run {k + 1} [{name}]: {d:+.1f} pp [{lo:+.1f}, {hi:+.1f}]; only Opus {b_}, only Gemini {c_}; p = {p_:.3g}")

    print("\nper-ontology EA lenient (Opus / qwen2.5-coder:32b / Gemini mean of 2):")
    for o in onts:
        g = statistics.mean(S["per_ont"][o]["len"] for S in S_gem) if S_gem else float("nan")
        print(f"  {o:38s} n {S_opus['per_ont'][o]['n']:4d}  {S_opus['per_ont'][o]['len']:5.1f} / {S_ref['per_ont'][o]['len']:5.1f} / {g:5.1f}")

    spider = [o for o in onts if o.startswith("spider-")]
    def pooled_of(S, subset):
        n = sum(S["per_ont"][o]["n"] for o in subset)
        return 100 * sum(sum(v[0] for v in S["per_ont"][o]["pq"].values()) for o in subset) / n, n
    sp_o, n_sp = pooled_of(S_opus, spider)
    sp_r, _ = pooled_of(S_ref, spider)
    sp_g = [pooled_of(S, spider)[0] for S in S_gem]
    print(f"\nSpider pooled EA ({n_sp} questions, {len(spider)} datasets): Opus {sp_o:.1f}, qwen {sp_r:.1f}, "
          f"Gemini {' / '.join(f'{x:.1f}' for x in sp_g)}")
    if "bestiary" in onts:
        gb = " / ".join("%.1f" % S["per_ont"]["bestiary"]["len"] for S in S_gem)
        print(f"bestiary EA: Opus {S_opus['per_ont']['bestiary']['len']:.1f}, qwen {S_ref['per_ont']['bestiary']['len']:.1f}, Gemini {gb}")

    cats = Counter()
    for o in onts:
        for q in opus[o][1]["results"]:
            if q.get("gold_answers") and not (q.get("comparison") or {}).get("exact_match"):
                cats[outcome(q)] += 1
    tot = sum(cats.values())
    print(f"\nOpus failures {tot}: " + ", ".join(f"{k} {OUTCOME_NAMES[k]} {100 * cats[k] / tot:.1f}%" for k in sorted(cats)))

    # cost from logs/api_usage.jsonl (calls made during the full run)
    import calendar  # noqa: PLC0415
    start = min(calendar.timegm(datetime.strptime(opus[o][0], "%Y%m%d-%H%M").timetuple()) for o in onts) - 60
    log = Path("logs") / "api_usage.jsonl"
    if log.exists():
        calls = [json.loads(l) for l in open(log, encoding="utf-8")]
        calls = [c for c in calls if c.get("model") == OPUS.split(":", 1)[1] and c["t"] >= start]
        tin = sum(c.get("prompt_tokens", 0) or 0 for c in calls)
        tout = sum(c.get("completion_tokens", 0) or 0 for c in calls)
        pin, pout = PRICES[OPUS.split(":", 1)[1]]
        fin = Counter(c.get("finish_reason") for c in calls)
        print(f"\nusage (full run): {len(calls)} calls, {tin / 1e6:.2f} M input, {tout / 1e6:.3f} M output tokens; "
              f"at ${pin}/${pout} per M: ${tin / 1e6 * pin + tout / 1e6 * pout:.2f}; finish_reason {dict(fin)}")

# ---------------------------------------------------------------- T4

def parse_ts(value: str) -> datetime | None:
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%MZ", "%Y-%m-%d %H:%M", "%Y%m%d-%H%M",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None


def t4(paper: Path) -> None:
    cells = main_cells(paper)
    packages = sorted({Path(d["package_dir"]).name for (o, m), (_, d) in cells.items()
                       if not m.startswith("api:")})
    print("T4: onboarding time per package, from logs/onboard.log (timestamps have minute resolution)\n")
    for name in packages:
        log = paper / name / "logs" / "onboard.log"
        if not log.exists():
            print(f"{name:52s} no onboard.log")
            continue
        events = []
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                e = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            events.append((parse_ts(e.get("timestamp")), e.get("event", "")))
        stamps = [t for t, _ in events if t is not None]
        if not stamps:
            print(f"{name:52s} {len(events)} events, unreadable timestamps (first: {events[0] if events else None})")
            continue
        minutes = (max(stamps) - min(stamps)).total_seconds() / 60
        print(f"{name:52s} {min(stamps):%Y-%m-%d %H:%M} -> {max(stamps):%H:%M}  {minutes:5.0f} min  "
              f"events {len(events)}: {events[0][1]} ... {events[-1][1]}")


# ---------------------------------------------------------------- main

ap = argparse.ArgumentParser()
ap.add_argument("task", choices=["b1", "b4a", "b5", "t1", "t2", "t3", "t4"])
ap.add_argument("--model", default=None, help="t3: model (default deepseek-r1:32b); b4a: model (default qwen2.5-coder:32b)")
ap.add_argument("--temperature", default=None, help="t3/b4a: select runs made with this NL2SPARQL_TEMPERATURE")
ap.add_argument("--paper-root", default=r"..\NL-to-SPARQL-V4")
args = ap.parse_args()
PAPER_ROOT = Path(args.paper_root)
paper_packages = PAPER_ROOT / "ontology_packages"
if args.task == "b1":
    b1(Path("ontology_packages"), paper_packages)
elif args.task == "b4a":
    b4a(Path("ontology_packages"), paper_packages, args.model or "qwen2.5-coder:32b", args.temperature or "0.6")
elif args.task == "b5":
    b5(Path("ontology_packages"), paper_packages)
elif args.task == "t1":
    t1(paper_packages)
elif args.task == "t2":
    t2(paper_packages)
elif args.task == "t3":
    t3(Path("ontology_packages"), paper_packages, args.model or "deepseek-r1:32b", args.temperature)
else:
    t4(paper_packages)
