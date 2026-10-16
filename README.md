# V4: ontology-agnostic, retrieval-augmented NL-to-SPARQL

Code, analysis scripts and per-question results for the paper
*Privacy-Preserving Natural-Language Querying of Knowledge Graphs with Local Large Language
Models: An Application to CBRN Preparedness*, S. Palmos et al. (submitted to Expert Systems
with Applications).

The framework turns a natural-language question into a SPARQL query for any OWL/RDF
knowledge graph, using a local open-weight LLM. It combines schema (TBox) retrieval,
instance (ABox) retrieval with reranking and gating, adaptive naming, prompt rules,
four-stage validation and execution-guided correction. One configuration is used
unchanged for every ontology.

## Repository layout

| Path | Contents |
|---|---|
| `app/` | The framework: onboarding, retrieval, prompt rendering, validation, execution, correction |
| `evaluation/` | Dataset schema, evaluation runner, answer comparison and metrics |
| `onboard.py`, `activate.py`, `query.py`, `evaluate.py` | Command-line entry points |
| `prepare_datasets.py`, `evaluation/question_labels.csv` | Rebuild the knowledge graphs and evaluation datasets from the public releases (see Datasets) |
| `run_v4_matrix.py`, `run_ablations.py` | The model matrix and the ablation study of the paper |
| `build_headline_tables.py`, `significance_tests.py`, `api_comparison.py`, `error_analysis.py`, `difficulty_levels.py`, ... | The analysis scripts that produce every table in the paper |
| `results/tables/` | The paper's tables (CSV) |
| `results/paper_outcomes.csv` | Every per-question outcome behind the tables (see below) |
| `results/source_sha256.csv` | SHA-256 of the knowledge-graph sources and converted datasets we used |
| `infra/docker/compose.yml` | Apache Jena Fuseki |
| `docs/architecture/` | Design notes |

## Requirements

- Python 3.11+, then `pip install -r requirements.txt`
- Apache Jena Fuseki 5.1.0 (`infra/docker/compose.yml`)
- Ollama 0.12.3. All models were run as Q4_K_M quantisations with temperature 0 and a
  32,768-token context: qwen2.5-coder:7b, qwen2.5-coder:32b, qwen3:8b, qwen3-coder:30b,
  llama3.1:8b, deepseek-r1:32b. The paper's runs used one NVIDIA L40S (48 GB).
- Optional, for the API comparison: an OpenAI-compatible endpoint, configured with the
  environment variables `NL2SPARQL_API_BASE`, `NL2SPARQL_API_KEY` and
  `NL2SPARQL_API_NO_TEMPERATURE=1`. Never commit a key.

## Datasets

The datasets are not redistributed here. `prepare_datasets.py` rebuilds them from the
public releases, and `results/source_sha256.csv` lets you check that every rebuilt file is
byte-identical to the one used in the paper.

| Dataset | Questions used | Source (version used) |
|---|---|---|
| Bestiary | 92 | Kovriguina et al. (2023), SPARQLGEN, CEUR-WS Vol-3526; files from [SandroGT/LLMs-for-SPARQL](https://github.com/SandroGT/LLMs-for-SPARQL) (commit `ddfe9c3`), `datasets/bestiary/processed/` |
| CK25 (TEXT2SPARQL'25 corporate KG) | 50 | [eccenca/ck25-dataset](https://github.com/eccenca/ck25-dataset), release `v1.1.0` (CC BY 4.0); TEXT2SPARQL'25, [CEUR-WS Vol-4094](https://ceur-ws.org/Vol-4094/) |
| Spider4SPARQL, 17 databases of the Spider dev split | 884 | Kosten et al. (2023), [arXiv:2309.16248](https://arxiv.org/abs/2309.16248); Yu et al. (2018), Spider; processed graphs and queries from [SandroGT/LLMs-for-SPARQL](https://github.com/SandroGT/LLMs-for-SPARQL) (commit `ddfe9c3`), `datasets/spider4sparql/processed/` |
| eNOVATION | 40 | Not public (EU project data). Results on it are reported in the paper only. |

The licence terms of each source apply to the rebuilt files.

### Rebuilding the datasets

```
git clone -c core.autocrlf=false https://github.com/SandroGT/LLMs-for-SPARQL downloads/LLMs-for-SPARQL
git -C downloads/LLMs-for-SPARQL checkout ddfe9c3
git clone -c core.autocrlf=false https://github.com/eccenca/ck25-dataset downloads/ck25-dataset
git -C downloads/ck25-dataset checkout v1.1.0

python prepare_datasets.py build     # knowledge graphs + question files (no Fuseki needed)
powershell -File onboard_all.ps1     # packages, loaded into Fuseki (see Running)
python prepare_datasets.py gold      # gold answers: every gold query run on its graph
python prepare_datasets.py check     # expected: 41 of 41 files byte-identical
```

What `build` does:
- **Bestiary and Spider4SPARQL graphs:** copied unchanged to `resources/knowledge_graphs/`
  (`bestiary_sandro.rdf`, `spider_correct/<db>.rdf`; the 18 databases include
  real_estate_properties, which has 2 questions and is left out of the paper's aggregates).
- **CK25 graph:** the union of `graphs/text2sparql.ttl`, `graphs/prod-inst.ttl` and
  `graphs/queries.ttl` (`resources/knowledge_graphs/ck25_text2sparql_combined.ttl`), then the
  same graph with trailing whitespace removed from literals (`trimmed_sources/ck25.ttl`), which
  is the version onboarded.
- **Question files** (`evaluation/datasets/`): one file per dataset, with the questions and gold
  SPARQL exactly as released (Spider4SPARQL and Bestiary: `sparql_complete_uri`; CK25: the
  English question and the reference query of `questions.yml`). The fields `complexity_tier`,
  `query_shape` and `question_type` come from `evaluation/question_labels.csv`; the paper does
  not use them, except `question_type` to count the yes/no questions.

The two CK25 files are written by rdflib; byte identity was checked with rdflib 7.1.4 and 7.6.0.

## Running

All commands run from the repository root.

1. **Fuseki.** `docker compose -f infra/docker/compose.yml up -d` (UI at `http://127.0.0.1:3030`).
2. **LLM endpoint.** Set `llm_api_url` in `app/core/config.py` to your Ollama endpoint
   (default `http://127.0.0.1:11434/api/generate`) and pull the models, e.g.
   `ollama pull qwen2.5-coder:32b`.
3. **Onboarding.** `powershell -File onboard_all.ps1` onboards every ontology of the paper
   (one: `python onboard.py --ontology <file> --output ontology_packages --name <name>`).
   Each run creates `ontology_packages/<name>-<timestamp>/` with the ontology context, the
   TBox indexes (class-based, property-based, composite) and the ABox index, and loads the
   graph into Fuseki.
4. **Model matrix.** `python run_v4_matrix.py` runs every ontology with the six models in the
   reference configuration (class-based TBox chunks, k = 5; ABox retrieval, k = 5; at most
   3 attempts). Options: `--only <ontologies>`, `--models <models>`, `--dry-run`.
   The API comparison used `--models api:gemini-3.1-flash-lite` with the environment
   variables above.
5. **Ablations.** `python run_ablations.py --list` shows the named configurations (ladder
   S0–S4, leave-one-out, retrieval variants, full-schema baseline, determinism repeats);
   run them with `--family ladder`, `--configs S0 S4 loo_no_abox`, etc. The paper's
   ablation subset also includes eNOVATION, which is not public.
6. **One question.** `python activate.py --package ontology_packages/<package>`, then
   `python query.py --question "..."`.
7. **One dataset.** `python evaluate.py --dataset <dataset.json> --package <package> --model <model> --chunking class_based --k 5 --abox-rag --abox-k 5`.
   Results go to `<package>/evaluation/<run-id>/` (`results.json`, `metrics.json`,
   `run_config.json`, `report.txt`).

## Reproducing the tables

With the packages built and the runs completed, from the repository root:

```
python build_headline_tables.py --root . --out-dir analysis
python significance_tests.py --strict --substitute S4:spider-world-1=det_full_a
python api_comparison.py
python error_analysis.py
python difficulty_levels.py --ea
python ontology_stats.py
python schema_size.py
python prompt_length.py
```

## Per-question outcomes

`results/paper_outcomes.csv` has one row per question and run for every run behind the
paper's tables, eNOVATION excluded. It holds identifiers, the generated SPARQL and the
scores only (no question text and no gold answers).

| Column | Meaning |
|---|---|
| `run_id` | package / run folder |
| `roles` | `main` (model matrix), `api` (the two API runs), `ablation:<config>`; `reference` and `prompt_length:<config>` mark runs used only for side tables |
| `ontology`, `model`, `timestamp`, `question_id` | identifiers |
| `scored` | 1 if the question counts (scored, non-empty gold) |
| `lenient`, `strict` | execution accuracy under the two scoring rules of the paper |
| `first_attempt` | 1 if no correction attempt was needed |
| `attempts`, `latency_ms` | attempts used and end-to-end latency |
| `final_sparql` | the query that was executed |

## Citation

TODO: BibTeX entry, once the paper is published.

## Licence

The code is released under the MIT License (see `LICENSE`). The datasets are not part of this
repository and keep their own licences.
