# Explicit resource-feasibility amendment and bounded V10 recovery

The user approved **33,660 scheduled GPU seconds (9.35 hours)**, retaining the
strict actual hard stop **before 36,000 seconds**. The original nine-hour
scheduled target was **not met**. This is a resource-feasibility amendment, not
evidence of scientific success. The hypotheses, comparisons, validators, model,
request, independent units, and held-out review requirement are unchanged.
The two authoritative plans remain preserved; this document records the approved
exception to their original scheduled resource target.

The executable amendment is `configs/study/bounded_recovery_v10.json`; the
scheduled limit in `configs/study/resource_limits.json` now reflects it. Prior
reports, measurements, failures, V9 authorization, and ledger snapshots remain
historical records and are not rewritten under the new limit.

## Admission before the one authorized start

| Component | Seconds |
| --- | ---: |
| Historical actual allocation, including unattended/failed service time | 2,936.238858 |
| Entire remaining registered inventory and reserves | 29,987.34434394846 |
| Earmark exactly one existing long reserve slot | −240 |
| New bounded recovery envelope | 840 |
| All-in conservative bound | **33,523.58320194846** |
| Amended scheduled reserve | **136.41679805154** |
| Projected strict-hard margin | **2,476.41679805154** |

No inference slot is added. The effective accounting inventory includes the one
new service start (291 accounting events, still 278 maximum inference attempts).
Five remaining base service-load slots and every mandatory comparison remain.
No unmeasured throughput or preparation saving is credited. Fresh valid model
timings feed the existing conservative forecast procedure.

## Recovery caps and continuation

Startup 300s; live checks 120s; unchanged `fallback-c1-01` retry 240s;
validation/resource drain 120s; shutdown 60s. Total: **840s**. Append-only stage
records bind the authorization, run, boot identity, monotonic clock, and UTC time.
The independent guardian monitors both the stage and whole deadline, reserving
60s for shutdown. Polling triggers one second early. Cooperative shutdown gets
10s before guardian takeover, leaving 50s for identity-bound cleanup. Guardian
cleanup uses a 10s TERM and bounded KILL path; the original hard-stop protection
also remains active.

C1 transport, HTTP, decoding, schema, or scientific failure is terminal: no
automatic repair and no further startup. Restricted pre-parse HTTP journals
preserve status, safe headers, response fragments, completion data, and exception
chains. An independent CPU preflight storage fixture verifies their actual remote
durability. It is not represented as inference or a model response.

A valid C1 plus fresh all-in admission releases only the recovery envelope. It
does not complete acceptance. The remaining three fallback calls, full acceptance
gates, and 24 development calls run on the same service only when admitted.
The separate-controller PID/checkpoint adoption proof is retained. The complete
development adapter and scientific validators are reused. No held-out execution
is allowed while independent human review remains missing. The service stops
when no authorized work can proceed.

## Operator commands (existing remote project root)

CPU preflight:

```sh
PYTHONPATH=src .venv/bin/python scripts/control_bounded_recovery.py preflight
```

The single authorized persistent launch, only after source association/preflight:

```sh
PYTHONPATH=src .venv/bin/python scripts/control_bounded_recovery.py launch
```

Read-only status:

```sh
PYTHONPATH=src .venv/bin/python scripts/control_bounded_recovery.py status
```

The launch command refuses an existing V10 run directory or result. V10 cannot
replay orchestration or allocate a replacement model process. Its controller
handoff is ordinary adoption of the same live process, not a second service start.
Restricted logs, stage records and checkpoints are under
`artifacts/restricted/fallback-development-v10/`. No results PDF is regenerated
for this amendment or recovery update.

## Focused verification before deployment

97 current-recovery checks pass: 64 cap/controller-preparation/development/resource
unit tests, 11 selected controller/source regressions, and 22 real loopback HTTP
tests. Full logs/XML are preserved under `artifacts/restricted/v10_validation/`.
The three C1 failure branches and successful continuation use authored CPU
fixtures, not model outputs. The resource-limit expectation was updated to the
explicitly approved 33,660s value. Two initially selected historical V9 overlay
tests reject the new resource configuration because their immutable forecast
binds 32,400s; their failures remain preserved. Historical 32,400s records and
their previous passing checks were not rewritten to fit the amendment.
