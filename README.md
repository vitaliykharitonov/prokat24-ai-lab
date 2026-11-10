# Prokat24 AI Lab

[![Check](https://github.com/vitaliykharitonov/prokat24-ai-lab/actions/workflows/check.yml/badge.svg)](https://github.com/vitaliykharitonov/prokat24-ai-lab/actions/workflows/check.yml)

**AI evaluation and observability project built around a real equipment-rental website.**

The project evaluates retrieval, RAG answers, LLM-as-a-Judge systems, and a tool-using agent using deterministic checks,
human-labelled datasets, holdout evaluation, adversarial testing, statistical analysis, and trace-level observability.

## What this project demonstrates

| Area                                  | What is demonstrated                                                                                                                                    |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Retrieval**                         | Keyword search, OpenAI embeddings, hybrid Reciprocal Rank Fusion, Top-1/Top-3 evaluation, dev/holdout datasets                                          |
| **RAG**                               | Grounded answer generation, required/forbidden content checks, retrieved context handling, citation validation                                          |
| **LLM-as-a-Judge**                    | Holistic and criterion-level judges, human-labelled benchmarks, grader comparison, confusion matrices, precision/recall, stability across repeated runs |
| **Agents**                            | Tool selection and arguments, multi-step execution, outcome evaluation, capability evaluation, grounding against tool results                           |
| **Safety**                            | Direct and indirect prompt injection, system-prompt extraction attempts, malicious retrieved content, unauthorized tool behavior                        |
| **Observability**                     | OpenTelemetry tracing, Phoenix annotations, structured logging, eval-to-trace correlation, token usage and cost estimation                              |
| **Evaluation analysis & reliability** | pandas, Jupyter, matplotlib, case-level diagnostics, historical run analysis, confidence intervals, paired comparisons, judge stability                 |
| **Engineering**                       | pytest, Ruff, mypy, GitHub Actions, reproducible datasets, JSON reports, Git metadata, automated quality gates                                          |

## System overview

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "fontSize": "18px",
    "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
    "primaryTextColor": "#ffffff",
    "lineColor": "#c9d1d9",
    "textColor": "#ffffff",
    "clusterBkg": "#161b22",
    "clusterBorder": "#8b949e",
    "edgeLabelBackground": "#0d1117"
  }
}}%%

flowchart LR
    U["User query"]

    subgraph Retrieval
        K["Keyword"]
        E["Embeddings"]
        H["Hybrid RRF"]

        K --> H
        E --> H
    end

    subgraph Runtime
        R["RAG"]
        A["Tool-using agent"]
        P["Product tools"]
        N["Knowledge tools"]

        A --> P
        A --> N
    end

    U --> K
    U --> E
    H --> R
    U --> A

    subgraph Evaluation
        ER["Retrieval"]
        EA["Answer / RAG"]
        EJ["LLM-as-a-Judge"]
        EG["Agent: outcome / capability / grounding"]
        ES["Safety / adversarial"]

        EJ --> EG
        EJ --> ES
    end

    H -.-> ER
    R -.-> EA
    A -.-> EG
    A -.-> ES

    subgraph Observability
        OT["OpenTelemetry traces"]
        PH["Phoenix annotations"]
        LG["Structured logs"]
        CO["Token / cost tracking"]
    end

    A -.-> OT
    OT --> PH
    OT --> LG
    OT --> CO

    classDef user fill:#0f766e,color:#ffffff,stroke:#5eead4,stroke-width:2px;
    classDef retrieval fill:#1e3a5f,color:#ffffff,stroke:#60a5fa,stroke-width:2px;
    classDef runtime fill:#5a3518,color:#ffffff,stroke:#fb923c,stroke-width:2px;
    classDef evaluation fill:#4a3b08,color:#ffffff,stroke:#f2cc60,stroke-width:2px;
    classDef judge fill:#3b2452,color:#ffffff,stroke:#d2a8ff,stroke-width:2px;
    classDef observability fill:#173d2a,color:#ffffff,stroke:#3fb950,stroke-width:2px;

    class U user;
    class K,E,H retrieval;
    class R,A,P,N runtime;
    class ER,EA,EG,ES evaluation;
    class EJ judge;
    class OT,PH,LG,CO observability;

    style Retrieval fill:#0f2238,stroke:#60a5fa,stroke-width:2px,color:#ffffff
    style Runtime fill:#3a2414,stroke:#fb923c,stroke-width:2px,color:#ffffff
    style Evaluation fill:#302907,stroke:#f2cc60,stroke-width:2px,color:#ffffff
    style Observability fill:#102b1f,stroke:#3fb950,stroke-width:2px,color:#ffffff

    linkStyle default stroke:#c9d1d9,stroke-width:2px
```

The project separates **runtime logic**, **evaluation datasets**, **evaluation logic**, and **reporting/observability**,
allowing both system behavior and evaluator behavior to be tested independently.
