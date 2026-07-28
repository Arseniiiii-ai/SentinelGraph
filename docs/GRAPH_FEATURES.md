# SentinelGraph v0.4 Graph Feature Dictionary

## Graph definition

Each account is a node and each transaction is a directed edge from
`nameOrig` to `nameDest`. Edge attributes include the transaction step and
amount. Raw account identifiers are used only to construct the graph and are
not written to a model matrix.

For a transaction at step `t`, all graph features are calculated from edges
with `step < t`. PaySim does not define within-hour ordering, so transactions
from the current step are excluded from one another's histories.

## Origin endpoint features

| Feature family | Definition |
| --- | --- |
| Incoming transaction count | Prior transactions received by the origin node |
| Outgoing transaction count | Prior transactions sent by the origin node |
| In-degree | Unique prior senders to the origin node |
| Out-degree | Unique prior recipients of the origin node |
| Total degree | Sum of prior in-degree and out-degree |
| In/out ratio | Log ratio of prior incoming to outgoing transaction counts |
| Received amount | Log cumulative amount previously received |
| Sent amount | Log cumulative amount previously sent |
| Flow ratio | Log ratio of received to sent amount |
| Prior role count | Whether the node previously appeared in zero, one, or both roles |

Destination endpoint features use the same definitions from the destination
node's point of view.

## Weak-component features

| Feature | Definition |
| --- | --- |
| `origin_log_component_size` | Log size of the origin's prior weak component |
| `destination_log_component_size` | Log size of the destination's prior weak component |
| `endpoints_same_component_prior` | Whether a prior undirected path already connects the endpoints |
| `log_combined_component_size` | Component size that the current edge would create |
| `component_size_log_ratio` | Difference between endpoint log-component sizes |
| `origin_component_is_isolated` | Origin has no prior graph edge |
| `destination_component_is_isolated` | Destination has no prior graph edge |
| `both_components_established` | Both endpoints already belong to non-trivial components |

Components are maintained incrementally with union-find. Every transaction in
one step is scored before any edge from that step is added.

## Label exposure policy

PaySim provides `isFraud` but does not provide the time when fraud was
confirmed. A suspicious-neighbour feature based on fraud labels would therefore
make an unsupported assumption about label availability. v0.4 excludes all
label propagation and records that decision in the graph feature manifest.

## Dataset limitations

- All 6,362,620 directed account pairs occur exactly once.
- There are no reciprocal directed pairs.
- Only 1,769 of 9,073,900 accounts appear in both transaction roles.
- The final weak graph has no undirected cycle.
- The largest weak component has only 121 accounts.

These properties make PaySim a weak benchmark for repeated pair behaviour and
GNN message passing. They do not represent the topology of a real payment
network.
