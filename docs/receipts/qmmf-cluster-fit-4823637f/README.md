# qmmf: clusters only where they fit, receipt (#3681)

| GLM-5.3-Flash FP4-Local, TP=2 on 2 × GB10, 128-token decode, classic expert form | main | this branch | change |
| --- | --- | --- | --- |
| Plain, median of 5 runs a serve (s; serves a1 b1 b2 a2, a3 b3 b4 a4) | 5.399, 5.417, 5.424, 5.441 | 5.349, 5.339, 5.343, 5.367 | −1.3% (−0.55 ms a token) |
| Drafted, median of 3 runs a serve (s) | 3.490, 3.395, 3.502, 3.505 | 3.321, 3.482, 3.499, 3.427 | −1% (inside noise) |
| Reply sha256, 128 tokens / a ~3k-token prompt | `bd5e5ca12ac6` / `a7ecc532c706` | the same in all 8 serves | equal |
| `tf-cuda-test qmmf-cluster-equal` (GB10) | — | 360 cases byte-equal, launches as planned, control red | |

Tested implementation: binaries built from `4823637f89` (branch) and `9489c38179` (main), nvcc 13.5 (V13.5.37),
Zig 0.17.0, driver 580.178.04, Ubuntu 24.04, NCCL 2.30; 2 × GB10 (sm_121) at TP=2 over a direct RoCE link;
`nvidia/GLM-5.3-Flash-FP4-Local` at `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`. The branch's later commit
(`ab9bf9b7e5`: FlashNext's partials bound, a GPU test, comments) changes nothing the GLM serve runs. Test date
2026-11-17.

## What it measures

On GB10 the 8-slice qmmf tile fits 12 resident clusters of 8, so GLM's 2048-column FP8 projections (shared expert
gate+up, DSA q_a+kv_a; 53 launches a token) ran as 32 clusters in three waves at 146–154 GB/s. With the fit rule they
run as 256 plain CTAs and the reduce, in the same slice order. Both binaries' load races kept the classic expert
form (`runs/qab-forms.txt`; each serve's form is in `runs/qab.txt`), so only this change differs.

## Checks

| Check | Result |
| --- | --- |
| Served A/B, alternated in two ABBA blocks (`runs/qab.txt`, `runs/qab-*/`) | Table above; every branch median is below every main median |
| Exactness | 128-token greedy reply and a ~3k-token prompt's reply: one sha each across all 16 sets of runs on both binaries |
| Cluster against reduce on GB10 (`runs/qmmf-cluster-equal.log`) | 5 modes × widths 512 and 2048 × sk 2, 4, 8 × rows 1, 16, 17, 32, 33, 64 × both outputs: byte-equal, finite, each run's launches recorded (one clustered launch, or one plain launch and a reduce); the reversed-order reduce of four cancelling slices gives 0 against 1 |
| Host | `zig build test-cuda-host compile-cuda-tests test-gradient-serving` pass; lean 221 on main and on this branch |

## Reproduction

Serves: `tensorfold-native serve CKPT --host 127.0.0.1 --port 8090 --name local-model --no-update-check --context
65536 --parallel 1 --prompt-cache-gib 0` on both ranks (`TF_GLM_RANK`). `tools/init2.sh` gives each binary its first
load (its expert-form race); `tools/ab-qcf.sh` runs the A/B with `tools/one_sha.py` (greedy, drafts off, reply
sha256) and `tools/one.py` (drafted). GPU check: `tf-cuda-test qmmf-cluster-equal`.
