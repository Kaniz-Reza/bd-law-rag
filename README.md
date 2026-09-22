# bd-law-rag

![License](https://img.shields.io/badge/license-MIT-green)

Bangla and English question answering over Bangladeshi laws. Answers cite the
exact Act and section, the service refuses when the law does not cover the
question, and it is monitored for retrieval quality, latency, and cost.


**Data:** [Bangladesh Legal Acts Dataset](https://www.kaggle.com/datasets/sakhadib/bangladesh-legal-acts-dataset)
(BLAD) by Adib Sakhawat, CC BY-SA 4.0
**Pilot scope:** 30 acts (12 Bengali, 18 English) · ~3,691 sections · ~3,920 chunks

---

## 📋 Status

| Stage | Status |
|---|---|
| Project setup — tooling, config, pre-commit | ✅ Done |
| Data ingestion — clean and validate the raw BLAD dataset | ✅ Done |
| Pilot act selection — 30 acts locked | ✅ Done |
| Chunking — split sections into retrieval-sized chunks | ✅ Done |
| Embeddings and a searchable vector index | ⬜ Next |
| Hand-written test set and retrieval metrics (recall@k, MRR) | ⬜ Planned |
| Answer generation with an LLM, citations, refusal path | ⬜ Planned |
| FastAPI service (`/health`, `/ask`) | ⬜ Planned |
| Docker and docker-compose | ⬜ Planned |
| CI/CD (GitHub Actions) with an evaluation gate | ⬜ Planned |
| Deployment with a public URL | ⬜ Planned |
| Monitoring (Prometheus/Grafana) and a drift simulation | ⬜ Planned |

---

## 🚀 Setup

```bash
git clone https://github.com/Kaniz-Reza/bd-law-rag.git
cd bd-law-rag
uv venv --python 3.11
source .venv/Scripts/activate   # Windows Git Bash
make setup
pre-commit install
```

---

## 📁 Project structure

```
src/
  ingestion/   # download, clean and validate the raw legal text
  retrieval/   # chunking, embeddings, and search over the chunks
  generation/  # LLM prompting and answer generation
  evaluation/  # retrieval and answer-quality metrics
  serving/     # the FastAPI service
  monitoring/  # latency, cost, and drift metrics
configs/       # config.yaml: pilot acts, chunking, retrieval, and LLM settings
tests/         # unit tests for each module
```


## 🗂️ Reproducing the data

The `data/` folder is not committed to this repo (see `.gitignore`), since it
holds a ~60 MB dataset and generated files. To rebuild it:

1. Download the [Bangladesh Legal Acts Dataset](https://www.kaggle.com/datasets/sakhadib/bangladesh-legal-acts-dataset)
   from Kaggle and unzip it into `data/raw/`.
2. Run the pipeline:

```bash
   python -m src.ingestion.load    # -> data/processed/acts.jsonl, sections.jsonl
   python -m src.retrieval.chunk   # -> data/processed/chunks.jsonl
```
