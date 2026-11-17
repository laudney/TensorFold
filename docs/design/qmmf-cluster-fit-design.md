# qmmf: a split meets in a cluster only when the clusters fit one wave (#3681)

## Problem

qmmf's decode lane splits K into `sk` slices (`splitK`: 1, 2, 4 or 8) and, on sm_90 and newer, meets the slices
in a cluster of `sk` CTAs (`planSlices`: `cluster = !fused and sk > 1 and clusters`). A cluster of 8 must sit in
one GPC, so the device holds few of them at once: on GB10 (48 SMs) `cuOccupancyMaxActiveClusters` gives 12 for
the 8-slice tile, 24 for 4 slices and 48 for 2. GLM-5.3-Flash's 2048-column FP8 projections (shared expert
gate+up, DSA q_a+kv_a; N 2048, K 4096, 53 launches a decode token) launch 32 clusters of 8, so they run in three
waves and reach 154 GB/s (64% of DRAM). A prototype that meets these slices through qmmf's `reduce_kernel`
instead (256 plain CTAs, one wave) gave -0.5 ms a token (-1.2%) with equal replies.

## Goal

A split uses a cluster only when every cluster of its launch is resident at once on this device; otherwise it
runs plain CTAs and the existing reduce. The rule comes from the driver's occupancy query, so it follows the GPU,
not a slice count.

Non-goals: new tiles, a new split plan (`splitK` is unchanged, so every row's slices and their order stay the
same), the prompt tiles, the routed experts, the L2 prefetch.

## Design

1. **Driver.** `cuOccupancyMaxActiveClusters` (CUDA 11.8) is loaded as an optional entry, like `cuFuncGetName`
   (`Driver.max_clusters: ?MaxActiveClusters`). A driver or test driver without it leaves today's behaviour.

2. **Lane load.** `Lane.load` records, for each cluster instance the plan can pick (every mode's 16-, 32- and
   64-row cluster tiles, both outputs, and the half tiles' cluster forms) and each cluster size 2, 4 and 8, the
   resident clusters the driver reports for one 128-thread CTA with the tile's dynamic shared memory. Zero means
   unknown. About 100 queries, once a lane.

3. **Plan.** One helper, `Lane.shape(m, w)`, decides everything the launcher picks today: row tile, half or
   full columns (the half-tile rule moves into it), slices, fused, and now cluster or reduce. It keeps `cluster`
   only when the launch's clusters (row tiles x column tiles of that instance) are at most the recorded fit,
   taking the smaller fit of the BF16 and FP32 instances, so the decision does not depend on the output; an
   unknown fit keeps today's plan. `partBytes`, `usesPrompt` and `matmulOut` all read `shape`, so planning and
   launching cannot disagree.

4. **Partials over every row count.** Scratch demand is not monotonic in the row count (a 1-row half-tile launch
   can reduce while a 17-row full-tile launch keeps its clusters). `Lane.maxPartBytes(w, rows)` returns the
   largest `partBytes` over 1 .. min(rows, fused_rows - 1) rows, and every family sizes its lane partials with it
   for the largest window and prompt tail it runs, instead of one `partBytes` at one row count.

5. **Families.** Each family that runs qmmf's lane sizes its partials with `maxPartBytes` and passes them:
   - GLM: `cuda_proj.matmul` (FP8 projections) and `cuda_mlp.mm` (NVFP4 dense decode) pass `mm_part`; the state
     sizes `mm_part` from `maxPartBytes` over those weights; the loader check becomes "mm_part holds it"; GLM
     vision either passes partials or is shown never to reduce at its shapes.
   - FlashNext: its native lane partials, today left out on cluster-capable GPUs, are sized and passed.
   - MiniMax M3: its prompt partials, sized at a row count where slices are fused, are sized over the tails.
   - Qwen3.5 ModelOpt: `Kernels.reserveParts`, sized at 255 rows only, moves to `maxPartBytes`.
   - cuda_dense, dory, kolibri1, nemotron: their existing `partBytes` sizing moves to `maxPartBytes`.
   A launch given no partials still fails with `MissingLanePart`; the sizing tests below cover a buffer that is
   too small, which that check cannot see.

## Alternative considered: clusters that loop

Keeping the clusters and capping the grid at the resident clusters, each cluster looping over tile groups as
`nvfp4_expert_waves` does, needs no partials and no reduce. It does not remove the waves: 32 tile groups on 12
resident clusters of 8 still take three rounds (12, 12, 8), so the clusters' residency, not the launch, still sets
the time. Plain CTAs put all 256 on the device at once (about 5 an SM),
which is why the prototype recovered 0.5 ms; the loop would also change every qmmf instance's body. The plain
form is kept.

## Bits

`splitK` and the slice order do not change; only where the slices meet changes. qmmf's cluster form and its
`reduce_kernel` sum the `sk` slices in the same order (slice 0 first), which the prototype's equal replies
support. The PR adds a GPU check that runs each moved class both ways and compares the bytes, with a planted
control that sums the slices in another order.

## Tests

- Host: `shape` keeps a cluster when the fit holds and drops it when it does not, from a recorded fit table, for
  both outputs, row buckets 16/32/64, half and full columns and slice counts 2/4/8, across the half/full
  transition; an unknown fit keeps today's plan; `partBytes` and `maxPartBytes` follow `shape`.
- Sizing: each family's lane partials hold `maxPartBytes` for its largest window and prompt tail (a table test
  per family over its real weight shapes).
- GB10: a byte comparison of the reduce launch (`clusters = false`) against the cluster launch (`clusters = true`
  with the fit set above the grid), over every mode, both outputs, row counts 1, 16, 17, 32, 33 and 64 (each
  bucket and its edge), slice counts 2, 4 and 8, and full and half column tiles, from random packed weights;
  each pair asserts which launch ran (clusters, or plain CTAs and the reduce), and a planted control that sums
  the slices in another order goes red.
- Served: GLM-5.3-Flash at TP=2, main against branch, the expert form pinned, plain and drafted, several streams,
  more alternated runs than the prototype (its gain is near the noise), tokens equal; one other family whose
  class moves, tokens equal.

## Risks

- A family that passed no partials on sm_90 now needs them for a moved class: covered by step 5, and by
  `MissingLanePart` failing loudly rather than reading garbage.
- The reduce adds one launch a split; for a class that fits one wave the cluster stays, so only classes that
  wave change.
