# GLM-5.3-Flash decode as dependent launches: design

Status: draft, 2026-11-16. Scope: GLM-5.3-Flash on CUDA (`zig/src/families/glm`, the GLM kernel images) and the
shared qmmf lane (`zig/src/cuda/qmmf.zig`, `zig/kernels/cuda/qmmf*.cu*`). Same bits: no arithmetic changes.

## Problem

A GLM-5.3-Flash FP4-Local decode step at TP=2 on two GB10 launches about 1,320 kernels on the forward's stream
(Nsight Systems, 64 tokens, one rank). Between two kernels the stream sits idle for 2 to 3 us: the time GB10 takes
from one kernel's end to the next one's first instruction. The gaps add up to 3.1 ms a token, 7.4% of a 41.6 ms
step. The largest shares sit before the qmmf linears (192 a token, 0.64 ms), `hc_post`, `kda_out`, the experts,
`hc_partial8`, the RoCE gathers, and the router chain.

Programmatic dependent launch (PDL, sm_90 and newer) removes most of a gap: a kernel launched with the
`programmatic_stream_serialization` attribute may start while the kernel before it still runs, once that kernel has
executed `griddepcontrol.launch_dependents` (or ended). The new kernel must execute `griddepcontrol.wait` before it
reads anything the earlier kernels wrote or writes anything they read; the wait returns when the kernel before has
completed and its writes are visible. Today only `glm_mm` (the BF16 linears) launches this way in GLM (`TF_GLM_PDL`,
default on, #2370); the glue, the KDA and MLA kernels, and the qmmf linears do not.

## Measurement that motivates this (proof of concept, never pushed)

A measurement build of main `e86a0bed1d` put `launch_dependents` + `wait` at the top of every kernel body and a
driver shim made the forward stream's launches of a named kernel list dependent. Four greedy 256-token replies,
alternated launches on one pair of GB10, the same binary each time:

| Kernels launched as dependent | Sum of 4 replies (s) | Change | Replies |
| --- | --- | --- | --- |
| none | 43.78, 43.44, 43.01 | | `a1934e30` |
| every kernel of ours on the forward stream | 41.97, 42.04, 42.04 | -3.6% | `a1934e30` |
| GLM's own kernels and qmmf (experts, `plan_kernel`, `roce_gather` as today) | 40.83, 41.12 | -5.6% | `a1934e30` |

Making the shared expert kernels, the experts' plan and the RoCE gather dependent costs time, so they stay as they
are. qmmf alone with a weight-first prologue (below) measured -0.6% over its plain entry wait.

## Design

### One rule for every GLM kernel: it opens with the dependency wait

Every `__global__` body in the GLM kernel images (`glm_glue`, `glm_ops`, `glm_kda`, `glm_mm`, `glm_quant`, the index
images, `glm_dflash`, `glm_vision`) begins with `glm_pdl::enter()`: `griddepcontrol.launch_dependents` then
`griddepcontrol.wait` (both compile to nothing below sm_90). A body may instead place its own wait later (weight-first,
below) if nothing before the wait reads data an earlier kernel writes and nothing before it writes global memory.

With the wait first, a dependent launch behaves like a plain launch for everything the kernel reads and writes; the
only change is that the next kernel may become resident early. Since every kernel of the chain waits before it
touches memory, a kernel past its wait sees every earlier kernel of the stream complete (each one passed its own wait
only after the one before it completed). The rule holds for every GLM kernel, so whether a launch is dependent is a
launch-site choice that can never make a kernel read early, on any stream. A unit test scans the GLM kernel sources
and fails when a `__global__` body's first statement is not `glm_pdl::enter()`, unless the kernel is on a short named
list of reviewed weight-first bodies (`glm_mm`'s, qmmf's); fixtures with a late wait, a conditional wait and no wait
must fail the scan.

`launch_dependents` at the top is safe for occupancy: the next grid launches only after every CTA of this grid has
executed it or ended, so a multi-wave grid's last wave is already resident when the next grid's CTAs arrive.

### One switch: TF_GLM_PDL covers every GLM launch

`TF_GLM_PDL` (default on, exact 0 turns it off) already decides `glm_mm`'s dependent launches. It now decides every
GLM launch: `glue.Kernels` carries `pdl`, read once at load like `mm.Kernels.pdl`, and each GLM launch site passes
`.pdl = <that flag>`. `profile.zig`'s effect line widens to "the forward's dependent launches". No new switch.

The stream does not matter for safety: a GLM kernel waits first wherever it runs, so the paired prompt's second
stream (`cuda_forward_pair.zig`, a copy of the forward on its own stream) launches dependent too. Launches that stay
plain are those of kernels outside the GLM images, which do not wait: the shared experts and their plan,
`roce_gather`, NCCL, the NVFP4 prompt kernels, and the L2 prefetch kernels on their side stream. After a host copy or
an event wait the driver orders the next kernel as a plain launch; the kernel still waits first.

### qmmf: a lane that launches dependent, a weight-first prologue

`cuda.qmmf.Lane` gains `pdl: bool = false`. GLM's forward sets its lane's `pdl` from `TF_GLM_PDL`; every other model
keeps false and launches as today. With `pdl`, `matmulOut` launches dependent only the entries that wait: the
`qmmf_kernel` decode instances (the 16/32/64-row tiles plain and with a cluster, the 32-column half tiles, the
fused entry) and `reduce_kernel`: `.pdl = lane.pdl and !prompt`. The prompt choices (the 128-row
`qmmf_kernel<BF16, 128, 128, ...>` instances and `qmmf_prompt_k32_kernel`) and the shared-down entries stay plain
whatever the flag says (GLM loads none of them today); the 128-row instances still carry the prologue, being the same
template, so a later dependent launch of them would also be safe.

`qmmf_kernel` (every instance of the template; the K32 and strided shared-down entries keep their own bodies) starts
with `launch_dependents`, issues its prologue stages' weight loads (cp.async or the bulk copies on sm_120), then
`griddepcontrol.wait`, then the same stages' input rows. The weights stream from DRAM while the kernel before still
runs. Weights do not change after load: the engine synchronizes its stream at the end of load before the first
forward, and nothing rewrites a GLM linear's weights afterwards (living weights' runtime rewrites cover Nemotron and
Flash Next, whose lanes keep `pdl` false; a lane that may see its weights rewritten must keep it false).
`reduce_kernel` opens with the plain wait. Below sm_90, or launched plain, the wait is a no-op and the loads only
change order inside the prologue: same bits, same results.

### What does not change

No arithmetic, no sum order, no buffer layout: every output keeps its bits (the PoC's replies above, and the
kernels' outputs byte for byte). No change to the expert kernels (#2266 and #3249 own them), the RoCE gather, the
graphs path (GLM builds none; `graph.zig` refuses PDL nodes and is not reached), or other models' launches.

## Costs and risks

- Kernel resources: the entry wait changes a few kernels' register counts (PoC on sm_121: 10 of 90 qmmf instances
  -1 register, `tf_glm_chunks_16` 122 -> 128, `tf_glm_stream_mean` 40 -> 44). Their kernel-contract records are
  re-measured for nvcc 13.0 and 13.3 on the five served SMs.
- A kernel added later without the entry wait and launched with `.pdl` would read early; the source scan test
  refuses it.
- A dependent launch after a host-visible doorbell (the RoCE gather) or NCCL is plain by construction (those are not
  GLM kernels), so their timing does not change.

## Verification

- Dependency test (new, `tf-cuda-test glm-pdl-order`): a slow writer kernel (spins about 1 ms, then writes the
  input) is followed by the kernel under test launched dependent; its output bytes must equal the same launch made
  plain after a synchronize. Covered: one kernel of each GLM image, a multi-wave glue grid, the decode qmmf (plain
  and cluster instance), its reduce, and a chain of dependent launches through a plain launch, a host copy and an
  event wait. The input starts as a poison value; the slow writer executes `launch_dependents` before its delay,
  then writes the real input. A negative control (a test kernel that reads before its wait) records the value it
  actually read; a try counts as exercised only when that value was the poison, and then the control's output must
  differ from the reference. The test retries a bounded number of times and reports "overlap not exercised" (a
  failure on sm_90 and newer) when no try read the poison.
- Same bits: kernel outputs byte-identical to main (the contract's gates and the dependency test against main's
  bytes); served replies equal to main for greedy decode, drafted = plain, concurrent = alone, resumed = cold
  (`suite-ex`).
- Speed: alternated base, head, head, base served launches: plain decode tok/s (128 tokens, 3 reps) and cold
  8,192 / 32,768-token prompts; `TF_GLM_PDL=0` on the head for the switch's off state.
- Kernel contract: changed kernels' records current with nvcc 13.0 and 13.3.
