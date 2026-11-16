# GLM-5.3-Flash decode as dependent launches: implementation

Builds `docs/design/glm-decode-pdl-design.md`. Base: upstream main `077998f7d9`. Four commits, each reviewed and
checked on its own. The kernel entries and the launch switch land together in commit 1: an early trigger alone
already releases `glm_mm`'s existing dependent launches sooner, so splitting them would not be scheduling-neutral.

## Commit 1: every GLM kernel opens with the dependency wait, and GLM launches dependent under TF_GLM_PDL

- New `zig/kernels/cuda/glm_pdl.cuh` (`#pragma once`, namespace `glm_pdl`): `enter()` = `launch_dependents` then
  `wait`, and `trigger()` / `wait()` alone, each `__device__ __forceinline__`, the asm under
  `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900` (the five served SMs include sm_80/86/89).
- Included once by each GLM image unit: `glm_glue.cu`, `glm_ops.cu`, `glm_kda.cu`, `glm_quant.cu`, `glm_index.cu`,
  `glm_dflash.cu`, `glm_vision.cu` (the glue headers get it through `glm_glue.cu`'s first include or
  `glm_triton.cuh`; whichever header a kernel's file includes first must bring it, so each `.cuh` that defines a
  `__global__` includes `glm_pdl.cuh` itself; the include is idempotent).
- Every `__global__` body in `glm_*.cu` / `glm_*.cuh` gets `glm_pdl::enter();` as its first statement, macro bodies
  included (`RMSNORM`, `TF_GLM_SWIGLU_512/64`, `TF_GLM_DEC_TILE`, the other macro-defined kernels the scan finds).
  Only `glm_mm.cu`'s matmul bodies (the `glm_mm{B}_{16,32,64,128}` and prompt entries, which call `grid_launch()`,
  load weights, then `grid_wait()`) are a named exception; `glm_mm_reduce` (today: a trigger, no wait) and the
  load-time tilers (`glm_mm_tile*`) get `glm_pdl::enter()` like every other kernel.
- Scan test: `zig/build/glm_pdl_scan.zig` adds `test-glm-pdl-scan` (listed in `host_ci.zig` with the other host
  steps). At build time it lists `zig/kernels/cuda/glm_*.cu` and `glm_*.cuh` and copies them, with
  `zig/tests/glm_pdl_scan.zig` as `main.zig`, into a write-files directory, the way `gradient_host.zig` does; the test
  embeds each file and checks that every `__global__` definition's body (found by matching the parameter list's
  parentheses, then the body's braces; declarations and explicit instantiations end in `;` and are skipped) starts
  with `glm_pdl::enter();` after whitespace and comments, unless its name is on the exception list
  (`glm_mm`'s matmul bodies only). Its own unit tests feed three fixtures that must fail: a late wait, a wait under an
  `if`, and no wait; and one macro-body fixture that must pass.
- Re-pin `zig/tests/fixtures/gradient-serving.tsv` for each changed pinned file (`glm_glue*.cuh`, `glm_glue.cu`,
  `glm_kda.cu`, `glm_ops.cu`, `glm_quant.cu`, `glm_triton.cuh` if touched), the note naming this change.
- Kernel contract: rebuild the GLM images with nvcc 13.0 and 13.3; every kernel whose registers or static shared
  bytes moved gets its records re-measured (`glm_cuda_records.zig`, `measurements/pending_133.zig`,
  `nvcc_133_records.zig` as they hold it); `zig build test-kernel-contract` and the limits audit clean.

Launch side (same commit):

- New `zig/src/families/glm/cuda_pdl.zig`:
  - `pub fn on() bool`: `TF_GLM_PDL` through `environment.value(bool, "TF_GLM_PDL", profile.lookup(...), true)`,
    read once and kept (an atomic once-flag; the switch is process-wide and does not change while serving).
  - `pub fn launch(f: cuda.Function, cfg: cuda.launch.Config, s: cuda.Stream, a: *cuda.Args) !void`: `cfg.pdl = on()`,
    then `cuda.launch.launch`. Under `std.debug.runtime_safety` it checks the kernel's name
    (`f.d.functionName(f.handle)`) starts with one of the GLM images' symbol families, `tf_glm_`, `glm_` (`glm_mm*`,
    `glm_vis_*`) or `df_` (`glm_dflash`), and panics otherwise ("?" passes): only GLM image kernels wait first.
  - Unit tests: the name rule (`tf_glm_hc_post`, `glm_mm_reduce`, `glm_vis_rmsnorm`, `df_attn` accepted; `tf_draw`,
    `tf_pack_dense`, `roce_gather` refused; "?" passes), and that `launch` sets `.pdl` from `on()`.
- `cuda_mm.zig`: `k.pdl = cuda_pdl.on()` (one reader of the switch); its `k.reduce` launch passes `.pdl = k.pdl`.
- Every launch of a GLM-image kernel in `zig/src/families/glm` calls `cuda_pdl.launch` instead of
  `cuda.launch.launch` (sites listed by `git grep "launch.launch(" zig/src/families/glm`, and the shared KDA launcher
  below: `cuda_glue`, `cuda_sparse`,
  `cuda_latent`, `cuda_b16`, `cuda_kda_block`, `cuda_dsa_block`, `cuda_index`, `cuda_engine_pick`,
  `cuda_structured`, `cuda_follow`, `cuda_moe` (`put_slot`), `cuda_mlp` (`copy_rows`), `cuda_kv_host`,
  `cuda_quant` / `cuda_q4` where the function is a `glm_quant` kernel, `dflash_block`, `cuda_vision`,
  `cuda_images` / `cuda_taps` where the function is a GLM kernel). Launches of non-GLM kernels stay
  `cuda.launch.launch` (L2 prefetch, the sampler's `tf_draw` / `tf_logprob_rows` / `tf_grammar_mask`, `cuda_q4`'s
  `tf_pack_dense` / `tf_transpose_pad16` and `cuda_weights`' `pack_dense` from the shared pack images, the expert
  kernels); the runtime-safety name check catches a wrong pick in Debug test runs.
- Shared KDA launcher `zig/src/cuda/kda.zig` (prep, step, out, chain, replay launches): its `Kernels` gains
  `pdl: bool`, passed as `.pdl` on each launch. `Kernels.load` builds the struct field by field from `undefined`,
  so it sets `k.pdl = false` explicitly; GLM sets it from `cuda_pdl.on()` after loading. Other users keep false
  (their kernels have no entry wait); `kda_test.zig` checks a default load launches plain.
- Host test drivers that record launches (`cuda_images_interleave_test.zig`, `cuda_taps_tp_test.zig`, and any other
  GLM test whose fake driver fills only `cuLaunchKernel`): fill `cuLaunchKernelEx` too (recording the config and its
  attributes) and `func_name` (or leave it null, which `functionName` reports as "?").
- `profile.zig`: `TF_GLM_PDL`'s effect becomes "Exact 0 turns off the forward's dependent launches";
  `docs/environment.md` and the recipe text that names the switch say the same.

## Commit 2: qmmf lanes that launch dependent, weight-first prologue

- `zig/src/cuda/qmmf.zig`: `Lane.pdl: bool = false`. In `matmulOut`, `.pdl = l.pdl and !prompt` on the main launch
  (`k32` implies `prompt`) and `.pdl = l.pdl` on the reduce launch. A unit test on `Lane` with fake functions checks
  the prompt choice stays plain with `pdl` set (the existing `lane` test fixture at the file's end).
- `zig/kernels/cuda/qmmf.cu`: `qmmf_kernel` opens with `trigger` (`launch_dependents`) and defines
  `TF_QMMF_WFIRST` around its `#include "qmmf_body.cuh"`; `reduce_kernel` opens with the entry wait.
- `zig/kernels/cuda/qmmf_body.cuh`: `load(s, g, what = 3)`: bit 1 the input rows, bit 2 the weights and scales
  (cp.async and the sm_120 bulk-copy path). Under `TF_QMMF_WFIRST` the prologue issues
  `if (s < per) load(s, g0 + s, 2)` for the `STAGES - 1` prologue stages, then `griddepcontrol.wait`, then
  `if (s < per) load(s, g0 + s, 1)` and an unconditional `commit()` per stage (short slices, K of 64 or 128, keep
  fewer groups than prologue slots and must not load past them; the commit count stays the same).
  The first commit group then holds every prologue weight copy and stage 0's rows; the loop's
  `wait<STAGES - 2>` still covers stage 0 (it waits for more, never less). The bulk copies complete on their own
  mbarriers as before. Without `TF_QMMF_WFIRST` (K32, strided) the prologue is unchanged.
- GLM: `cuda_forward.zig` sets `m.lane.pdl = cuda_pdl.on()` after `Lane.load`.
- Re-pin `qmmf.cu` and `qmmf.zig`; qmmf contract records re-measured where registers moved (PoC: 10 of 90
  instances on sm_121); `test-kernel-contract` clean.

## Commit 3: the dependency test

- `zig/kernels/cuda/probe.cu`: `tf_probe_pdl_slow_write(in, n, value, spins)`: `launch_dependents`, spin about
  `spins` clock cycles, write `value` into `in[0..n)`; `tf_probe_pdl_early_read(in, out, n, seen)`: reads `in`
  BEFORE `griddepcontrol.wait`, records `in[0]` in `seen`, then waits, writes `out = in_read * 2` (the control).
- `zig/tests/cuda/glm_pdl_order.zig`, `tf-cuda-test glm-pdl-order` (sm_90+ only; skips cleanly below):
  - Control: poison `in`, slow write, early read launched dependent; up to 8 tries; a try with `seen == poison` must
    give `out != reference`; no such try fails with "overlap not exercised".
  - For each kernel under test, one or more from every GLM image (`glm_glue`: `tf_glm_hc_post`,
    `tf_glm_rmsnorm_4096_f32w`, a multi-wave `tf_glm_router_tile`; `glm_kda`: `tf_glm_kda_out` through the shared
    launcher; `glm_ops`: `tf_glm_put_slot`; `glm_index`: a scores kernel; `glm_quant`: `tf_glm_nvfp4_quantize`;
    `glm_dflash`: `df_swiglu`; `glm_vision`: one `glm_vis_` kernel; `glm_mm`: the decode tile and `glm_mm_reduce`;
    qmmf: the 16-row decode instance plain and cluster, a K=64 short slice, and `reduce_kernel`): its input buffer
    starts poisoned, the slow writer fills it, the kernel launches dependent through the production launcher
    (`cuda_pdl.launch`, the KDA launcher, `Lane.matmulOut` with `pdl`), and its output bytes must equal the same
    launch made plain after a synchronize.
  - Byte equality to main: `--base-images DIR` loads main's fatbins (the same images built from the base tree) by
    path and runs every kernel under test on the same inputs plain; each output must equal the head's bytes.
  - A chain: dependent, plain, host copy, event wait, dependent, ending in the same bytes as all-plain.
- Wired next to the other `tf-cuda-test` subcommands in `zig/tests/cuda/main.zig` (one dispatch line; the file
  stays within 600 lines).

## Commit 4: receipt

- `docs/receipts/glm-decode-pdl-<base>/`: alternated base, head, head, base served launches (`suite-ab.sh`: plain
  decode 128 tokens 3 reps, cold 8k / 32k prompts), `TF_GLM_PDL=0` on the head, `suite-ex.sh` exactness on both,
  the dependency test and contract outputs, tools, `SHA256SUMS`.

## Checks for every commit

`zig build` (native, the GB10 kernels), `zig build test` for the touched host tests, `test-glm-pdl-scan`,
`test-gradient-serving`, `test-kernel-contract` on spark7 (nvcc 13.0 records via a 13.0 runner), `lean_check.py`
count unchanged against base, `git diff --check`.
