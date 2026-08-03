# Rebuild results

## What was tested

The logical query is an ordered conjunction. Ground truth stayed outside the runtime. The custom path used FP8 KV pages and called the pinned vLLM model runner without the vLLM scheduler or KV manager.

## Custom runtime versus stock vLLM

- Repetitions: 11
- Median custom time divided by stock time: 0.9646
- 95 percent upper bootstrap bound: 1.0010
- Answer flips: 2
- Median runtime overhead above model-runner time: 1.64%
- Gate: failed

| Run | Custom | Stock | Custom / stock | Runtime overhead | Flips |
|---|---:|---:|---:|---:|---:|
| `20260803T074035953884Z-packing-compare-6e270416` | 0.6120 s | 0.6164 s | 0.9929 | 0.0107 s | 0 |
| `20260803T074041885221Z-packing-compare-6d75f598` | 0.6203 s | 0.5956 s | 1.0414 | 0.0104 s | 0 |
| `20260803T074236227072Z-packing-compare-70a64b59` | 0.5868 s | 0.7261 s | 0.8082 | 0.0075 s | 0 |
| `20260803T074522220592Z-packing-compare-90c42a0c` | 0.6045 s | 0.6267 s | 0.9646 | 0.0091 s | 0 |
| `20260803T074842417162Z-packing-compare-bcb0a9c7` | 0.5988 s | 0.7079 s | 0.8458 | 0.0101 s | 0 |
| `20260803T075211613310Z-packing-compare-1bdd8f66` | 0.6021 s | 0.8020 s | 0.7508 | 0.0090 s | 1 |
| `20260803T075340912937Z-packing-compare-c2fecaa4` | 0.6033 s | 0.8446 s | 0.7143 | 0.0090 s | 1 |
| `20260803T075550062839Z-packing-compare-dc19f03e` | 0.6035 s | 0.6753 s | 0.8938 | 0.0100 s | 0 |
| `20260803T075706172634Z-packing-compare-84b35ee0` | 0.6082 s | 0.6076 s | 1.0010 | 0.0104 s | 0 |
| `20260803T075854606155Z-packing-compare-049f37ca` | 0.5976 s | 0.5992 s | 0.9974 | 0.0091 s | 0 |
| `20260803T075902404194Z-packing-compare-7ff5e672` | 0.6028 s | 0.5871 s | 1.0267 | 0.0097 s | 0 |

## Headline scale results

| Documents | Custom k=1 | Stock vLLM | Stock / custom | Answers | Flips |
|---:|---:|---:|---:|---:|---:|
| 2000 | 11.16 s | 11.45 s | 1.026 | 4570 | 0 |
| 10000 | 60.31 s | 108.26 s | 1.795 | 24083 | 0 |

Model-runner time is the measured lower reference for the same chosen batches. It is not a hardware lower bound over all possible schedules.

## Fixed-work fused k

Each row executes all four filters. These are one-document operator checks.
The verified fused implementation runs one prefix group per forward. Multi-group full-model runs are excluded because they changed Boolean answers.

| Target document tokens | k | Time | Steps | Accuracy | Flips from k=1 |
|---:|---:|---:|---:|---:|---:|
| 300 | 1 | 0.2201 s | 4 | 75.00% | 0 |
| 300 | 2 | 0.1251 s | 3 | 75.00% | 0 |
| 300 | 4 | 0.0988 s | 2 | 75.00% | 0 |
| 3000 | 1 | 0.2511 s | 4 | 75.00% | 0 |
| 3000 | 2 | 0.1806 s | 3 | 100.00% | 1 |
| 3000 | 4 | 0.1566 s | 2 | 100.00% | 1 |
| 30000 | 1 | 1.3790 s | 5 | 75.00% | 0 |
| 30000 | 2 | 0.9699 s | 4 | 50.00% | 1 |
| 30000 | 4 | 0.9219 s | 3 | 50.00% | 1 |

Fused k changed 4 fixed-work answers. It does not pass the semantic shipping gate.

### Fused 2,000-document semantic check

| k | Time | Steps | Shared answers | Flips from k=1 | Survivors |
|---:|---:|---:|---:|---:|---:|
| 2 | 108.59 s | 4710 | 4266 | 371 | 646 |
| 4 | 83.64 s | 4000 | 4266 | 371 | 646 |

These verified one-prefix-group runs fail semantics and performance. Fused FP8 cascade must not ship.

## Attention cost estimator

- Held-out median absolute error: 3.09%
- Held-out 95th-percentile absolute error: 9.48%
- Gate: passed

Cascade attention is not always faster. The planner must use the measured shape table and choose ordinary attention when cascade merge overhead is larger than the saved prefix work.
