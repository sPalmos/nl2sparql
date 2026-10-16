"""Rebuild the paper's knowledge graphs and evaluation datasets from the public releases.

Sources (clone them into ./downloads, or pass --downloads):
    git clone -c core.autocrlf=false https://github.com/SandroGT/LLMs-for-SPARQL downloads/LLMs-for-SPARQL
    git -C downloads/LLMs-for-SPARQL checkout ddfe9c3      # the commit we used
    git clone -c core.autocrlf=false https://github.com/eccenca/ck25-dataset downloads/ck25-dataset
    git -C downloads/ck25-dataset checkout v1.1.0          # the release we used
(`-c core.autocrlf=false` keeps the files byte-identical to the release on Windows.)

Steps
    python prepare_datasets.py build    graphs + question files (no Fuseki needed)
    powershell -File onboard_all.ps1    build the packages and load them into Fuseki
    python prepare_datasets.py gold     fill gold_answers by running every gold query on Fuseki
    python prepare_datasets.py check    compare every produced file with results/source_sha256.csv

What `build` writes (under --out, default the repository root)
    resources/knowledge_graphs/bestiary_sandro.rdf             copy of the release file
    resources/knowledge_graphs/spider_correct/<db>.rdf          copies of the 18 Spider4SPARQL graphs
    resources/knowledge_graphs/ck25_text2sparql_combined.ttl    union of the three CK25 graph files
    trimmed_sources/ck25.ttl                                    the CK25 graph with trailing
                                                                whitespace removed from literals
    evaluation/datasets/...json                                 questions and gold SPARQL as released;
                                                                complexity_tier / query_shape /
                                                                question_type from
                                                                evaluation/question_labels.csv (our
                                                                labels; the paper does not use them
                                                                except question_type = boolean)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "evaluation" / "question_labels.csv"
FUSEKI = "http://127.0.0.1:3030"

SANDRO = "LLMs-for-SPARQL"
CK25 = "ck25-dataset"
SPIDER_DBS = [
    "battle_death", "car_1", "concert_singer", "course_teach", "cre_Doc_Template_Mgt", "dog_kennels",
    "employee_hire_evaluation", "flight_2", "museum_visit", "network_1", "pets_1", "poker_player",
    "real_estate_properties", "singer", "student_transcripts_tracking", "tvshow", "voter_1", "world_1",
]
SPIDER_DIR = "evaluation/datasets/spider4sparql_sandro"


def spider_dataset(db: str) -> str:
    return f"spider_{db.lower()}_combined_eval_dataset"


def spider_package(db: str) -> str:
    return "spider-" + db.lower().replace("_", "-")


# dataset file (relative to --out) -> package name prefix
DATASETS = {
    "evaluation/datasets/bestiary_sandro_eval_dataset.json": "bestiary",
    "evaluation/datasets/ck25_text2sparql_combined_eval_dataset.json": "ck25",
    **{f"{SPIDER_DIR}/{spider_dataset(db)}.json": spider_package(db) for db in SPIDER_DBS},
}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_labels() -> dict[tuple[str, str], dict]:
    return {(r["dataset"], r["id"]): r for r in read_csv(LABELS)}


def label_fields(labels: dict, dataset: str, qid: str) -> dict:
    row = labels[(dataset, qid)]
    return {k: row[k] for k in ("complexity_tier", "query_shape", "question_type")}


def need(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"Missing source file: {path}\nSee the clone commands at the top of this script.")
    return path


# --------------------------------------------------------------------------- build

def check_line_endings(dl: Path) -> None:
    """The releases use LF line endings; a Windows clone with core.autocrlf=true turns them into CRLF."""
    for rel in (f"{SANDRO}/datasets/bestiary/processed/queries/dev/bestiary.csv", f"{CK25}/graphs/prod-inst.ttl"):
        if b"\r\n" in need(dl / rel).read_bytes():
            sys.exit(f"{dl / rel} has Windows (CRLF) line endings, so the rebuilt files would differ.\n"
                     "Delete the downloads folder and clone again with: git clone -c core.autocrlf=false ...")


def build_graphs(dl: Path, out: Path) -> None:
    from rdflib import Graph, Literal                                    # noqa: PLC0415

    check_line_endings(dl)

    kg = out / "resources" / "knowledge_graphs"
    (kg / "spider_correct").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(need(dl / SANDRO / "datasets/bestiary/processed/graph/dev/bestiary.rdf"),
                    kg / "bestiary_sandro.rdf")
    for db in SPIDER_DBS:
        shutil.copyfile(need(dl / SANDRO / f"datasets/spider4sparql/processed/graph/dev/{db}.rdf"),
                        kg / "spider_correct" / f"{db}.rdf")
    print("  copied bestiary and 18 Spider4SPARQL graphs")

    combined = Graph()
    for name in ("text2sparql.ttl", "prod-inst.ttl", "queries.ttl"):      # this order reproduces our file
        combined.parse(need(dl / CK25 / "graphs" / name))
    combined.serialize(destination=str(kg / "ck25_text2sparql_combined.ttl"), format="turtle")

    trimmed, n = Graph(), 0
    for prefix, namespace in combined.namespaces():
        trimmed.bind(prefix, namespace)
    for s, p, o in combined:
        if isinstance(o, Literal) and str(o) != str(o).rstrip():
            o = Literal(str(o).rstrip(), datatype=o.datatype, lang=o.language)
            n += 1
        trimmed.add((s, p, o))
    (out / "trimmed_sources").mkdir(parents=True, exist_ok=True)
    trimmed.serialize(destination=str(out / "trimmed_sources" / "ck25.ttl"), format="turtle")
    print(f"  CK25: {len(combined)} triples combined, {n} literals trimmed")


def build_questions(dl: Path, out: Path) -> None:
    import yaml                                                          # noqa: PLC0415

    labels = load_labels()
    total = 0

    name = "bestiary_sandro_eval_dataset"
    rows = read_csv(need(dl / SANDRO / "datasets/bestiary/processed/queries/dev/bestiary.csv"))
    questions = []
    for i, r in enumerate(rows, 1):
        qid = f"Q{i:03d}"
        questions.append({"id": qid, "original_id": f"bestiary_{i:03d}", "nl_question": r["nl_question"],
                          "gold_sparql": r["sparql_complete_uri"], "gold_answers": [],
                          **label_fields(labels, name, qid), "sparql_partial_uri": r["sparql_partial_uri"].strip()})
    write_json(out / "evaluation/datasets" / f"{name}.json", {
        "dataset_name": name,
        "description": "Sandro bestiary questions converted from resources/library/sandro_bestiary/bestiary.csv.",
        "source": "custom", "ontology_file": "sandro_bestiary/bestiary.rdf", "endpoint": None,
        "questions": questions})
    total += len(questions)

    name = "ck25_text2sparql_combined_eval_dataset"
    release = yaml.safe_load(need(dl / CK25 / "questions.yml").read_text(encoding="utf-8"))
    questions = []
    for i, q in enumerate(release["questions"], 1):
        qid = f"Q{i:03d}"
        questions.append({"id": qid, "original_id": f"ck25_{q['id']}", "nl_question": q["question"]["en"],
                          "gold_sparql": q["query"]["sparql"].rstrip("\n"), "gold_answers": [],
                          **label_fields(labels, name, qid), "features": q.get("features"),
                          "classes": q.get("classes"), "properties": q.get("properties")})
    meta = release["dataset"]
    write_json(out / "evaluation/datasets" / f"{name}.json", {
        "dataset_name": name,
        "description": "English CK25 Text2SPARQL questions converted from "
                       "resources/library/ontologies/text2sparql/questions.yml. Gold answers are intentionally not populated.",
        "source": "custom", "ontology_file": "ontologies/text2sparql/ck25_combined.ttl", "endpoint": None,
        "dataset_id": meta["id"], "dataset_prefix": meta["prefix"], "default_namespace": meta["defaultNamespace"],
        "questions": questions})
    total += len(questions)

    for db in SPIDER_DBS:
        name = spider_dataset(db)
        rows = read_csv(need(dl / SANDRO / f"datasets/spider4sparql/processed/queries/dev/{db}.csv"))
        questions = []
        for i, r in enumerate(rows, 1):
            qid = f"Q{i:03d}"
            questions.append({"id": qid, "original_id": f"{db}_{i:03d}", "gold_sparql": r["sparql_complete_uri"],
                              "gold_answers": [], **label_fields(labels, name, qid), "notes": None,
                              "nl_question": r["nl_question"]})
        write_json(out / SPIDER_DIR / f"{name}.json", {
            "dataset_name": name,
            "description": f"SandroGT Spider4SPARQL processed dev questions for {db}. "
                           "Gold answers are populated from the RDF graph.",
            "source": "spider4sparql", "ontology_file": f"sandro_spider4sparql_graphs/{db}.rdf", "endpoint": None,
            "questions": questions})
        total += len(questions)
    print(f"  wrote {len(DATASETS)} question files, {total} questions (gold_answers still empty)")


# --------------------------------------------------------------------------- gold

def newest_package(prefix: str) -> Path | None:
    import re                                                            # noqa: PLC0415
    found = [p for p in (ROOT / "ontology_packages").glob(f"{prefix}-*")
             if p.is_dir() and re.fullmatch(rf"{re.escape(prefix)}-\d{{8}}-\d{{4}}(-\d+)?", p.name)]
    return sorted(found)[-1] if found else None


def gold(out: Path) -> None:
    sys.path.insert(0, str(ROOT))
    from evaluation.dataset_builder import execute_sparql, extract_bindings  # noqa: PLC0415

    for rel, prefix in DATASETS.items():
        path = out / rel
        package = newest_package(prefix)
        if package is None:
            sys.exit(f"No package for {prefix}; run onboard_all.ps1 first.")
        act = subprocess.run([sys.executable, "activate.py", "--package", str(package)],
                             cwd=ROOT, capture_output=True, text=True)
        if act.returncode != 0:
            sys.exit(f"Activation of {package.name} failed:\n{act.stderr[-1000:]}")
        endpoint = f"{FUSEKI}/{package.name}/query"
        data = json.loads(path.read_text(encoding="utf-8"))
        failed = 0
        for q in data["questions"]:
            try:
                q["gold_answers"] = extract_bindings(execute_sparql(endpoint, q["gold_sparql"], timeout=60))
            except Exception as exc:                                     # noqa: BLE001
                failed += 1
                print(f"    ! {q['id']}: {str(exc)[:120]}")
        write_json(path, data)
        print(f"  {Path(rel).name:70s} {len(data['questions']):>4} questions  {failed} failed  [{package.name}]")


# --------------------------------------------------------------------------- check

def produced_files(out: Path) -> list[str]:
    files = ["resources/knowledge_graphs/bestiary_sandro.rdf",
             "resources/knowledge_graphs/ck25_text2sparql_combined.ttl",
             "trimmed_sources/ck25.ttl"]
    files += [f"resources/knowledge_graphs/spider_correct/{db}.rdf" for db in SPIDER_DBS]
    return files + list(DATASETS)


def check(out: Path, hashes: Path) -> None:
    with hashes.open(encoding="utf-8") as handle:
        expected = {r["path"].replace("\\", "/"): r["sha256"] for r in csv.DictReader(handle)}
    same = 0
    for rel in produced_files(out):
        path = out / rel
        if rel not in expected:
            print(f"  NOT IN HASH LIST  {rel}")
            continue
        if not path.exists():
            print(f"  MISSING           {rel}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == expected[rel]:
            same += 1
        else:
            print(f"  DIFFERENT         {rel}")
    print(f"\n{same} of {len(produced_files(out))} files are byte-identical to the ones used in the paper.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["build", "gold", "check"])
    parser.add_argument("--downloads", default="downloads", help="folder holding the two cloned repositories")
    parser.add_argument("--out", default=".", help="root folder to write into (default: repository root)")
    parser.add_argument("--hashes", default="results/source_sha256.csv")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if args.step == "build":
        dl = Path(args.downloads).resolve()
        build_graphs(dl, out)
        build_questions(dl, out)
    elif args.step == "gold":
        gold(out)
    else:
        check(out, Path(args.hashes))


if __name__ == "__main__":
    main()