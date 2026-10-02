# Swarm Capability Allocation — Simulators

Simulation code for studying where capabilities should live in a heterogeneous robot swarm: on every agent, on a one-hop neighbour, or behind a central sink. The simulators measure control-loop deadline misses as swarm size, deadlines and radio configuration change.

## Layout

| Path | Contents |
|---|---|
| `sim/swarmlat.py` | Slot-level discrete-event latency simulator. Covers local hosting (L), one-hop hosting (E), a sink with slotted ALOHA, and a sink with an ideal TDMA schedule and k receivers. |
| `sim/mission.py` | Kinematic mission simulator. M1 is stressed locomotion, using latencies drawn from `swarmlat.py`. M2 is an allocation mission under a matched sensing budget. |
| `sim/make_figs.py` | Plots the results in `sim/results/` into `sim/figures/`. |
| `sim/results/` | CSV outputs from the runs. |

## Running

```bash
cd sim
pip install -r requirements.txt
python3 swarmlat.py      # latency experiments, about 6 min
python3 mission.py       # mission experiments M1 and M2, about 10 min on 2 cores
python3 make_figs.py     # writes figures to sim/figures/
```

All runs use fixed seeds. Parameters are listed at the top of `swarmlat.py` (`P`) and `mission.py` (`M`).

## Modelling choices

- Hops other than the sink hop are modelled as contention-free.
- Relay queueing is ignored.
- The S-TDMA baseline is an ideal collision-free schedule.
- The one-hop model uses mean-field contention.

## Authors

Madhavan Vivekanandan and Sasimegala Ramasamy
