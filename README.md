# Generalist Substrate, Specialist Organs

Code and LaTeX source for *"Generalist Substrate, Specialist Organs: A Latency-Derived Criterion for Capability Allocation in Heterogeneous Robot Swarms"*, by Madhavan Vivekanandan and Sasimegala Ramasamy.

The paper asks which capabilities every agent in a heterogeneous swarm must carry. It proves three results:

1. **Threshold (Theorem 1).** A capability serving a loop with deadline τ must be on every agent that executes the loop if and only if τ < τ*_f = T_s + T_c + mT_a + fT_o.
2. **Sink saturation (Theorem 2).** Hosting through a sink with k radios misses at least a fraction 1 − N\*/N of deadlines, where N\* = k/(mλT_a), whatever the MAC protocol.
3. **Distance floor (Proposition 1).** Multi-hop distance alone forces a miss fraction of at least 1 − N_H(τ)/N.

The simulators in `sim/` test each bound, and every number and figure in the paper is produced by this code.

## Layout

| Path | Contents |
|---|---|
| `sim/swarmlat.py` | Slot-level discrete-event latency simulator. Covers local hosting (L), one-hop hosting (E), a sink with slotted ALOHA, and a sink with an ideal TDMA schedule and k receivers. |
| `sim/mission.py` | Kinematic mission simulator. M1 is stressed locomotion, using latencies drawn from `swarmlat.py`. M2 is the allocation mission under a matched sensing budget. |
| `sim/make_figs.py` | Generates the paper figures from `sim/results/`. |
| `sim/results/` | CSV outputs used in the paper. |
| `paper/` | LaTeX source (IEEE conference format), figures, and notes mapping each reviewer comment to its revision. |

## Reproducing the results

```bash
cd sim
pip install -r requirements.txt
python3 swarmlat.py      # latency experiments, about 6 min
python3 mission.py       # mission experiments M1 and M2, about 10 min on 2 cores
python3 make_figs.py     # writes the figures to ../paper/
cd ../paper && latexmk -pdf main.tex
```

All runs use fixed seeds. Parameters are listed at the top of `swarmlat.py` (`P`) and `mission.py` (`M`), and in Table II of the paper.

## Modelling choices that favour centralised hosting

- Hops other than the sink hop are modelled as contention-free.
- Relay queueing is ignored.
- The S-TDMA baseline is an ideal collision-free schedule.

Each of these favours the sink-hosted architectures, so their failure beyond N\* is conservative. The one-hop model uses mean-field contention.
