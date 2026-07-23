# SentinelGraph Project Charter

## 1. Product statement

SentinelGraph is a production-style decision intelligence platform that detects
fraud and account takeover from transaction streams. It combines classical
machine learning, anomaly detection, behavioural analytics, graph intelligence,
MLOps, and a grounded investigator copilot.

The product is intended to demonstrate end-to-end ownership:

- problem definition;
- data validation and leakage control;
- feature and model development;
- statistically sound evaluation;
- real-time serving;
- observability and retraining;
- human-centred investigation workflows;
- cloud deployment.

## 2. Target users

### Fraud analyst

Needs ranked cases, understandable reasons, related entities, and supporting
evidence.

### Risk manager

Needs control over approval/review/decline thresholds and visibility into false
positives, loss, and customer friction.

### ML engineer/data scientist

Needs reproducible training, model versions, reliable features, deployment,
monitoring, and feedback labels.

### Product manager

Needs business metrics, operational capacity, and clear trade-offs between fraud
recall and legitimate customer impact.

## 3. Decisions produced

Each transaction receives:

- calibrated fraud probability;
- anomaly score;
- graph risk score;
- combined risk score;
- approve, review, or decline recommendation;
- reason codes;
- model and feature versions.

The system does not automatically decline real transactions. All actions are
simulated, and high-impact investigator actions require explicit approval.

## 4. Core modelling tracks

### Track A: supervised classification

- Dummy and rule baselines
- Logistic regression
- LightGBM or CatBoost
- Probability calibration
- Cost-sensitive thresholding

### Track B: anomaly detection

- Account-specific behavioural deviation
- Isolation Forest
- Optional autoencoder
- Evaluation of novel or rare fraud patterns

### Track C: graph intelligence

Build a directed transaction graph:

- nodes: accounts;
- edges: transactions;
- edge properties: amount, time, type, label;
- node features: degree, velocity, counterpart diversity, balance behaviour;
- neighbourhood features: suspicious-neighbour rate and exposure.

Begin with engineered graph features. Add GraphSAGE or another GNN only if it
outperforms simpler features under a valid temporal evaluation.

### Track D: investigator copilot

The copilot can:

- summarize a case using stored evidence;
- explain model reason codes;
- construct an event timeline;
- identify connected suspicious accounts;
- suggest read-only investigation steps.

It cannot:

- modify labels without confirmation;
- block accounts;
- execute payments;
- access data outside the selected case;
- invent evidence not present in the case record.

## 5. Leakage risks

The project must explicitly test for:

- future transactions used in historical features;
- account-level overlap that reveals repeated fraud rings;
- target-derived graph features;
- post-event balances that may not be available at decision time;
- preprocessing fitted on validation/test data;
- random splitting that ignores time;
- duplicate transactions across splits.

Every feature must have an availability timestamp and an explanation of whether
it exists at scoring time.

## 6. Evaluation design

Use at least two test settings:

1. Future-time holdout: train on earlier steps and test on later steps.
2. New-entity holdout: measure performance on accounts not observed in training.

Report:

- PR-AUC;
- recall at selected false-positive rates;
- false positives per 10,000 legitimate transactions;
- fraud amount captured;
- review queue size;
- calibration and Brier score;
- performance by transaction type and amount band;
- performance on new versus known accounts;
- bootstrap confidence intervals;
- latency and throughput.

## 7. Decision policy

The system has three outcomes:

| Outcome | Meaning |
| --- | --- |
| Approve | Risk is below the review threshold |
| Review | Human investigation is required |
| Decline | Simulated high-risk recommendation only |

Thresholds will be optimized under:

- investigator capacity;
- false-positive customer cost;
- fraud-loss proxy;
- required minimum recall.

## 8. Production architecture

### Offline

```text
Raw transactions
-> schema validation
-> temporal split
-> reproducible features
-> model training
-> calibration
-> policy simulation
-> model registry
```

### Online

```text
Transaction request
-> request validation
-> point-in-time features
-> model ensemble
-> calibrated risk
-> decision policy
-> explanation
-> prediction and case storage
```

### Feedback

```text
Investigator decision
-> verified label
-> quality monitoring
-> retraining dataset
-> champion/challenger evaluation
```

## 9. Technology plan

| Concern | Initial choice |
| --- | --- |
| Data | Pandas or Polars, Parquet |
| Validation | Pydantic and Pandera |
| Models | Scikit-learn, LightGBM/CatBoost, PyTorch |
| Graph | NetworkX first, PyTorch Geometric if justified |
| Tracking | MLflow |
| API | FastAPI |
| Storage | PostgreSQL |
| Streaming | Batch replay first; Kafka/Redpanda later |
| Dashboard | Streamlit initially |
| Monitoring | Prometheus/Grafana plus drift reports |
| Packaging | Docker |
| Automation | GitHub Actions |
| Cloud | AWS S3, ECR, ECS, RDS, CloudWatch |

## 10. Definition of portfolio quality

The project is portfolio-ready only when:

- a new developer can reproduce the baseline;
- data leakage and time availability are documented;
- advanced methods are compared with simple baselines;
- metrics connect to investigation capacity and financial cost;
- one rejected modelling idea is documented honestly;
- tests cover data, features, models, API contracts, and policy decisions;
- a Dockerized demo runs locally;
- one cloud deployment is demonstrated;
- monitoring detects a simulated distribution shift;
- the copilot is evaluated for grounding and unauthorized-action resistance;
- the README includes measured results and limitations;
- a short technical demo and a longer system-design presentation exist.

## 11. CV evidence to earn

The final project should support truthful bullets covering:

- number of transactions processed;
- PR-AUC and recall at a fixed false-positive rate;
- fraud amount captured or expected-value improvement versus a rule baseline;
- graph/behavioural feature contribution;
- p95 API latency and throughput;
- automated test and CI/CD coverage;
- detected drift scenario;
- cloud deployment;
- investigator-copilot grounding and safety results.
