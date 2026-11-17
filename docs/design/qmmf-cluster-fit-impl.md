# qmmf cluster fit: implementation plan (#3681)

Design: `docs/design/qmmf-cluster-fit-design.md`. Five commits, each reviewed on its own.

## 1. cuda: the occupancy query

- `zig/src/cuda/driver.zig`: `max_clusters: ?MaxActiveClusters = null`, looked up as `cuOccupancyMaxActiveClusters`
  beside `cuFuncGetName` (signature `fn (*c_int, Function, *const abi.LaunchConfig) callconv(.c) Result`).
- `zig/src/cuda/launch.zig`: `pub fn activeClusters(f: Function, cfg: Config) ?u32`, the same `LaunchConfig` and
  attributes as `launch` (grid = one cluster), null when the entry is missing, the call fails or the cluster is
  absent. A host test with a recording driver checks the config it passes and the null cases.

## 2. qmmf: shape, fit and partials

- `Lane.fit: [modes][2 outs][3 tiles + half][3 sizes]u16`, 0 unknown; `Lane.load` fills it with `activeClusters`
  for each cluster instance (128 threads, the instance's dynamic shared memory, cluster `sk` along z as the
  launcher sets it). `fit` defaults to zeros (today's plan) in struct literals; every test builder that starts
  from `Lane = undefined` (grep `Lane = undefined`) sets `fit` and every other field `shape` reads (`half_below`).
- `Lane.shape(m, w) Shape`: row tile, half or full columns (the half rule moves out of `matmulOut`), slices,
  fused, cluster, reduce, and the instance index. Cluster only when the grid's clusters are at most the smaller
  fit of the two outputs (or that fit is 0). `planFor` stays for callers outside the lane (it is the
  cluster-agnostic slice plan); `partBytes`, `usesPrompt`, `matmulOut` and `qmmf_shared.zig` read `shape`.
- `Lane.maxPartBytes(w, rows)`: the largest `partBytes` over 1 .. min(rows, fused_rows - 1).
- Host tests: shape against a recorded fit table across modes, outputs, rows 1/16/17/32/33/64/255, sk 2/4/8,
  half and full; unknown fit keeps today's plan; `maxPartBytes` covers the half/full transition example.

## 3. Families

- GLM: `cuda_proj.matmul` passes `part` in `Rows`; `cuda_mlp.mm` gets `s.mm_part`; the load-time refusal in
  `cuda_formats.laneRuns` (`cuda_proj.fits`, run while the weights load, before any scratch exists) goes;
  `cuda_state.zig` sizes `mm_part` as the max of today's bf16 sizing and `maxPartBytes` over the loaded FP8 and
  NVFP4 dense lane weights at the largest window and prompt tail, so the engine (`cuda_engine.zig`, after the
  weights, where the buffers are made) has room by construction, and a test checks the sizing; vision
  (`cuda_vision.zig`) passes its own scratch sized the same way.
- FlashNext: `native_part` sized with `maxPartBytes` over the native lane weights whenever it is nonzero, not only
  under `lane_reduce`.
- MiniMax M3: the prompt partials sized with `maxPartBytes` over its tails.
- Qwen3.5 ModelOpt: `Kernels.reserveParts` uses `maxPartBytes`.
- cuda_dense, dory, kolibri1: `partBytes` sizing becomes `maxPartBytes`. Nemotron already sizes with
  `clusters = false` (`cuda_state.zig:99`), which covers every reduce; it is left as is and noted.
- A sizing test per changed family over its real shapes where one exists; otherwise the family's existing load
  test gains the check.

## 4. tf-cuda-test qmmf-cluster-equal (GB10)

Random packed weights per mode; for each mode, output, rows 1/16/17/32/33/64, sk 2/4/8 (pinned), full and half
tiles: run with `clusters = false` and with `clusters = true` and the fit raised above the grid, assert which
launch ran (the recorded launch config), compare bytes. Planted control: a deterministic cancellation fixture
with 4 slices (partial sums 1e8, 1, -1e8, 1 in fp32: slice order gives 1, the reverse order gives 0) run through
a reduce that adds the slices last first must differ from the lane's result. Registered in
`zig/tests/cuda/main.zig`.

## 5. Receipt

GLM-5.3-Flash FP4-Local at TP=2 on 2 x GB10, main against branch, expert form pinned (cache flag, noted), plain
and drafted, 1 and 4 streams, alternated with 5+ runs a serve, token shas equal; one other family whose class
moves (a Qwen3.5 or dense model on GB10), tokens equal; the qmmf-cluster-equal output; the kernel contract
unchanged (no kernel source changes).
