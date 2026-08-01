# v0.6 interview questions and model answers

## ML serving and feature parity

1. **What is training-serving skew?** A difference between feature logic, data,
   ordering, or preprocessing used in training and inference. It can invalidate
   production predictions even when the model artifact is unchanged.
2. **How does SentinelGraph prevent feature-order skew?** The server constructs
   the vector in the canonical `GRAPH_FEATURE_NAMES` order and validates the
   bundle against the same exact list before serving.
3. **Why does the request contain a historical snapshot?** Velocity, behaviour,
   and graph features require state before the current transaction; one event
   alone cannot reproduce them.
4. **Why must `as_of_step < step`?** Same-step edges and transactions are not
   available strictly before the decision and would leak concurrent/future data.
5. **Why reject missing features instead of filling zero?** Zero has business
   meaning for many counters. Imputation would silently turn an unavailable
   feature into false evidence.
6. **Why reject extra features?** They often signal a version mismatch or a
   misspelled field; silently ignoring them hides integration defects.
7. **What does `feature_version` solve?** It makes producer/consumer
   compatibility explicit and allows intentional migrations.
8. **Which features does the service derive?** Amount, log amount, cyclic hour,
   five type indicators, and merchant-destination indicator—everything known
   from the current event.
9. **Why not pass all 60 values from the client?** The server can deterministically
   derive event values and prevent a caller from contradicting the transaction.
10. **How would you add an online feature store?** Materialize the same
    point-in-time aggregations in state keyed by account, return a versioned
    snapshot, and parity-test online values against the offline pipeline.

## Model runtime and calibration

11. **Why load the model in FastAPI lifespan?** It is loaded once per worker
    before traffic, not on every request or during unrelated module imports.
12. **What happens if the bundle is missing?** Startup fails, so the worker never
    reports ready or serves an unintended fallback.
13. **Why validate bundle feature names?** A syntactically valid estimator can
    still accept the wrong columns and return plausible but meaningless scores.
14. **Why verify SHA-256 before Joblib load?** Pickle-based deserialisation may
    execute code. Integrity must be checked before any deserialisation occurs.
15. **Does SHA-256 make Joblib safe?** Only if the expected digest came from a
    trusted release process. It proves identity, not benign intent.
16. **Why return calibrated probability?** Ranking alone cannot support stable
    capacity/cost policies or interpretable risk communication.
17. **Probability versus risk points?** Probability is the calibrated analytical
    quantity; 0–1000 points are an operational presentation derived from it.
18. **Why return component scores?** They support audit and debugging of the
    behavioural, anomaly, and graph evidence without replacing the final score.
19. **How are reason codes calculated?** One feature at a time is replaced by a
    legitimate-development median; positive probability drops become local
    model-sensitivity evidence.
20. **Are reason codes causal?** No. They explain sensitivity of this model under
    this baseline, not what caused fraud in the real world.

## HTTP and reliability

21. **What is idempotency?** Repeating the same intended operation returns the
    same durable result rather than creating another prediction or case.
22. **Why require both external ID and idempotency key?** External ID identifies
    the business event; the key identifies the submission intent and supports
    safe retries across transport failures.
23. **Why store a request fingerprint?** A reused key with a different payload is
    rejected as a conflict instead of silently replaying the wrong prediction.
24. **Why use HTTP 409?** The request is structurally valid but conflicts with
    existing resource state or an optimistic lock.
25. **Why limit batch size?** It bounds memory, transaction duration, model work,
    and tail latency, and prevents one client from monopolizing a worker.
26. **Why make a batch atomic?** Clients do not have to reconcile an ambiguous
    half-persisted submission; failure rolls back the unit of work.
27. **What is the difference between liveness and readiness?** Liveness tests the
    process; readiness tests whether its model and database can serve traffic.
28. **Why should a database outage not fail liveness?** Restarting healthy code
    does not repair a dependency outage and can amplify it.
29. **What is a correlation/request ID for?** It joins client, API, database, and
    later monitoring evidence for one attempt.
30. **Why expose model-call latency separately?** It distinguishes inference cost
    from validation, database, serialization, queueing, and network latency.

## PostgreSQL and transactions

31. **Why PostgreSQL rather than only JSON files?** Cases require concurrent,
    indexed, transactional updates and durable relationships.
32. **What is a request-scoped SQLAlchemy Session?** A short-lived unit of work
    created for one HTTP use case and always closed afterward.
33. **Why use SQLAlchemy 2-style `select()`?** It is the current typed query API
    and avoids legacy `Session.query` behavior.
34. **Why do scoring and case creation share a transaction?** The system cannot
    commit a high-risk prediction without its required case, or vice versa.
35. **Why is investigator feedback a separate table?** Feedback is appendable
    training/governance evidence; case status is mutable workflow state.
36. **Why have a separate case-events table?** It preserves who changed what and
    when instead of retaining only the latest state.
37. **What is optimistic locking?** Update only if the row still has the version
    the client read; otherwise return a conflict and require refresh.
38. **Why not hold a database lock while an investigator reads?** Human think
    time is long and would cause blocking, deadlocks, and poor concurrency.
39. **Why use Alembic?** Schema changes become ordered, reviewable, repeatable,
    and separable from application startup.
40. **Why not run migrations automatically in every worker?** Concurrent replicas
    can race and a bad schema change becomes inseparable from application health.

## Security and privacy

41. **Why are account IDs not model features?** High-cardinality IDs encourage
    memorization, harm new-account generalization, and expand privacy risk.
42. **Why hash account IDs before persistence?** Cases can be linked by stable
    pseudonym without exposing the raw identifier.
43. **Why salt the hash?** It prevents straightforward dictionary/rainbow-table
    matching of predictable identifiers.
44. **Is a salted hash anonymous data?** Usually no; it remains pseudonymous and
    must still be protected and governed.
45. **Why compare API keys in constant time?** It reduces timing leakage about
    how many prefix characters matched.
46. **Why is API-key auth not the final design?** It lacks individual identity,
    roles, easy revocation, and strong investigator attribution; OIDC/RBAC is a
    later deployment requirement.
47. **Why is the dashboard HTML public?** It contains no case data or secret; all
    data operations still require the API key. A gateway can protect the shell too.
48. **Where does the browser keep its key?** `sessionStorage`, which ends with the
    tab session; it is not placed in a URL or server-rendered markup.
49. **Why restrict request bytes as well as field lengths?** Parsing an oversized
    body consumes resources before field validation can reject it.
50. **What additional production controls are needed?** TLS, secret manager,
    identity-aware RBAC, global rate limiting, network policy, backups, audit
    export, vulnerability scanning, and incident runbooks.

## Fraud operations and product trade-offs

51. **Why does approve create no case?** Investigator capacity is reserved for
    flagged work, while the prediction still remains auditable.
52. **Why do review and decline both create cases?** Decline is not an automated
    action in this project; it is a higher-priority recommendation for a human.
53. **Why store policy version separately from model version?** Thresholds and
    capacity policy can change without retraining the model.
54. **Why store feature version on every prediction?** Historical reproduction
    requires knowing the exact input semantics, not only estimator bytes.
55. **How should cases be prioritized?** Start with decision/risk points, then
    validate a business policy using amount, customer harm, SLA, and capacity.
56. **How does investigator feedback help later?** It supports label maturation,
    error analysis, threshold governance, and monitored retraining—after quality
    and bias checks.
57. **Why not train immediately on all feedback?** Investigator decisions can be
    delayed, selective, inconsistent, or biased; direct feedback loops amplify
    those effects.
58. **What is selection bias in the case queue?** Only high-score transactions
    are reviewed, so observed feedback is not representative of all traffic.
59. **What would you monitor in v0.7?** API errors/latency, DB pool, feature
    missingness/drift, component/risk distributions, policy rates, queue SLA,
    outcomes, and calibration after labels mature.
60. **What does a good rollback restore?** A compatible tuple of model,
    calibrator, policy, feature contract, and database/application version—not
    just one model file.

## Testing and deployment depth

61. **Why use SQLite in fast tests if production is PostgreSQL?** It quickly tests
    application transaction logic; PostgreSQL migration/integration tests are
    still required before deployment because dialect behavior differs.
62. **What should a PostgreSQL integration test cover?** Migration up/down or
    forward-only policy, JSONB persistence, unique races, indexes, isolation,
    timestamps, and connection failure behavior.
63. **Why is local latency not an SLA?** It excludes network, database, concurrent
    load, autoscaling, cold starts, and production hardware variability.
64. **Which percentiles matter?** p50 describes typical work; p95/p99 expose tail
    behavior that governs timeouts and user experience.
65. **What makes an API contract test valuable?** It protects status codes,
    required headers, validation rules, and response fields across refactors.
66. **Why test invalid paths?** A fraud system must fail closed and audibly when
    data, auth, versions, or persistence are wrong.
67. **How would you load-test the service?** Use realistic score/case mixes,
    concurrency and arrival rates; measure full HTTP p50/p95/p99, throughput,
    saturation, pool waits, errors, and recovery after dependency faults.
68. **When would async database access help?** For high concurrent I/O waits after
    profiling; it does not make CPU-bound scikit-learn inference async.
69. **How can CPU inference block an async server?** Running it on the event loop
    delays every coroutine. Sync endpoints let Starlette use a worker thread;
    process count and load tests still need tuning.
70. **What is the v0.6 definition of done?** A validated transaction can be scored,
    durably audited, converted into a human case, investigated, and closed with
    feedback through tested API/UI contracts.
