# sih-2026

Lunar image-registration system (Chandrayaan-2 OHRC/TMC/IIRS ↔ lunar
reference basemap) — Smart India Hackathon 2026.

## Microservices

| Service                  | Member | Path                       | Status        |
| ------------------------ | ------ | -------------------------- | ------------- |
| Ingestion                | 1      | `ingestion-svc/`           | (TBD)         |
| **Preprocessing**        | **2**  | `preprocessing-svc/`       | **delivered** |
| Coarse Matching          | 3      | `coarse-matching-svc/`     | (TBD)         |
| Verification             | 4      | `verification-svc/`        | (TBD)         |
| Refinement / Registration| 5      | `refinement-svc/`          | (TBD)         |
| Evaluation               | 6      | `evaluation-svc/`          | (TBD)         |
| Orchestrator             | 0      | `orchestrator/`            | (TBD)         |

## Preprocessing service (Member 2)

The preprocessing service is the **most scientifically critical stage**
of the pipeline. It consumes RawProducts from ingestion-svc and produces
illumination-invariant structural maps that allow the matching stage
downstream to find correct correspondences despite very different
illumination conditions on the lunar surface.

See [`preprocessing-svc/README.md`](preprocessing-svc/README.md) for the
full documentation. Quickstart:

```bash
cd preprocessing-svc
pip install -e .
pytest --cov=preprocessing_svc                  # 68 tests, ~92% coverage
PREPROC_OUTPUT_DIR=./var/outputs \
    python -m preprocessing_svc.api              # runs on :8080
```

The contract that downstream services (Coarse Matching) consume is
documented in
[`preprocessing-svc/pb/CONTRACT.md`](preprocessing-svc/pb/CONTRACT.md).
