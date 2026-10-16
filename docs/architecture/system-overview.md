# System Overview

This diagram shows the main runtime surfaces of the project and how ontology packages connect onboarding, activation, querying, and evaluation.

```mermaid
flowchart LR
    file[Ontology file\n.ttl .owl .rdf]
    endpoint[External SPARQL endpoint]
    onboard_entry[Onboarding entrypoints\nonboard.py\nPOST /ontology/load]
    onboard[onboarding_workflow]
    package[Ontology package\nmetadata + context + chunks + settings]
    active[Active package pointer\nontology_packages/.active_package]
    activate[Activation\nset_active_package]
    fuseki_client[FusekiService\nupload/reload]
    sparql_exec[sparql_execution\nHTTP query endpoint]
    fuseki[(Managed Fuseki dataset)]
    query[query.py / POST /query\nruntime pipeline]
    eval[evaluate.py\nevaluation runner]
    llm[LLM API]

    file --> onboard_entry
    endpoint --> onboard_entry
    onboard_entry --> onboard
    onboard --> package
    package --> activate --> active
    package -->|file source upload| fuseki_client --> fuseki

    active --> query
    active --> eval

    query --> llm
    query --> sparql_exec --> fuseki
    eval --> query
```

## Code Map

| Area | Main entrypoint | Main domain modules |
|---|---|---|
| Onboarding | `onboard.py`, `POST /ontology/load` in `app/api/routes/ontology.py` | `app/domain/ontology/onboarding_workflow.py` |
| Activation | `activate.py` | `app/domain/ontology/package_activation.py` |
| Querying | `query.py`, `app/api/routes/query.py` | `app/domain/runtime/pipeline.py` |
| Evaluation | `evaluate.py` | `evaluation/experiment_runner.py` |
| Fuseki upload/reload integration | n/a | `app/clients/fuseki.py` |
| Runtime SPARQL execution | n/a | `app/domain/runtime/sparql_execution.py` |
| LLM integration | n/a | `app/clients/llm.py` |
| Package state | n/a | `app/domain/package.py` |

## Invariants

- `query.py` always queries the active package.
- `/query` always queries the active package.
- For local file packages, activation is what guarantees the managed Fuseki dataset matches the package.
- Evaluation names a package, but only to verify it is already active.
