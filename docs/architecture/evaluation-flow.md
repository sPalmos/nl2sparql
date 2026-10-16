# Evaluation Flow

Evaluation runs a dataset of natural-language questions against the active package, compares generated answers to gold answers, aggregates metrics, and writes self-contained evaluation logs.

```mermaid
flowchart TD
    cli[evaluate.py]
    args[parse_args]
    package_arg[--package argument]
    active[ensure_requested_package_is_active\nrequested package == .active_package]
    settings[read package settings.json\nquery_endpoint]
    preflight[preflight_endpoint\nASK WHERE before timed run]
    dataset[load_dataset]
    output[choose output directory\n<package>/evaluation/<dataset>-<minute>]
    config[ExperimentConfig\npackage_dir\nmodel_name\nretrieval top-k\nchunking strategy\ncorrection attempts]
    runner[ExperimentRunner.run_experiment]
    question[for each EvaluationQuestion]
    pipeline[run_query_pipeline\nuses same package\npasses model + top-k + chunking]
    answers[extract_answers_from_sparql_json]
    qresult[QuestionResult\nfinal SPARQL\nanswers\niterations\ntrace paths\nlatency]
    scored{gold_answers present?}
    unscored[mark missing_gold\nno correctness comparison]
    compare[compare_results]
    qmetrics[compute_question_metrics]
    aggregate[aggregate_metrics]
    save[save_experiment]
    readable[write_evaluation_query_logs]
    logs[index.txt\nrun_config.json\nqueries.jsonl\nqueries/Qxxx.txt\nresults.json\nmetrics.json\nreport.txt]

    cli --> args --> package_arg --> active --> settings --> preflight
    preflight --> dataset --> output --> config --> runner
    runner --> question --> pipeline --> answers --> qresult --> scored
    scored -->|yes| compare --> qmetrics
    scored -->|no| unscored --> qmetrics
    qmetrics --> aggregate --> save
    qresult --> save
    save --> readable --> logs
```

## Code Map

| Step | Function / Module |
|---|---|
| Root CLI wrapper | `evaluate.py::main` |
| CLI parsing | `evaluate.py::parse_args` |
| CLI orchestration | `run_from_cli()` in `evaluation/experiment_runner.py` |
| Active package assertion | `ensure_requested_package_is_active()` in `experiment_runner.py` |
| Package endpoint lookup | `read_json_file(settings_path(package_dir))` in `run_from_cli()` |
| Endpoint preflight | `preflight_endpoint()` in `experiment_runner.py` |
| Output directory selection | `default_output_dir()` in `experiment_runner.py` |
| Evaluation config | `ExperimentConfig` in `experiment_runner.py` |
| Dataset schema | `EvaluationDataset`, `EvaluationQuestion` in `evaluation/dataset_schema.py` |
| Per-question execution | `ExperimentRunner._run_single_question()` in `experiment_runner.py` |
| Runtime query pipeline | `run_query_pipeline()` in `app/domain/runtime/pipeline.py` |
| Answer extraction | `extract_answers_from_sparql_json()` in `experiment_runner.py` |
| Answer comparison | `compare_results()` in `evaluation/answer_comparison.py` |
| Question result shape | `QuestionResult` in `evaluation/dataset_schema.py` |
| Metrics | `compute_question_metrics()`, `aggregate_metrics()` in `evaluation/metrics.py` |
| Output writing | `save_experiment()`, `write_evaluation_query_logs()` in `experiment_runner.py` |

## Per-Question Record

Each question stores:

- question id and text
- gold SPARQL and gold answers
- final generated SPARQL
- generated answers extracted from the SPARQL JSON response
- scored or `missing_gold`
- comparison details for scored questions
- iteration summaries from the runtime trace
- trace paths back to package-level query logs
- latency and pipeline config, including retrieval top-k, chunking strategy, and correction attempts

## Metrics Split

```text
correctness metrics:
  scored questions only
  exact match, precision, recall, F1

operational metrics:
  all questions
  latency, execution success, validation success, iterations
```

## Evaluation Outputs

```text
ontology_packages/<package>/evaluation/<run>/
  index.txt
  run_config.json
  report.txt
  metrics.json
  results.json
  queries.jsonl
  queries/
    Q001.txt
    Q002.txt
```

## Invariants

- Evaluation does not activate packages.
- The requested package must already be active.
- Endpoint preflight runs before timed question execution.
- Missing `gold_answers` questions are run but marked `missing_gold` / unscored.
- Unscored questions count toward operational metrics, not correctness metrics.
- `--k` controls retrieval top-k for the underlying query pipeline.
- `--chunking` selects one prebuilt package index for the underlying query pipeline.
- `--corrections` controls the maximum correction loop attempts for each evaluated question.
- `run_config.json` is the concentrated record of the actual runtime settings used for the evaluation run.
