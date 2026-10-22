# Serving GLM-5.3-Flash (321B) on CMP 170HX + RTX 3090 with Project Maya

**PROD lane: Maya-L (156.3 GB, 99.2% of FP8 fidelity) on 4 GPUs, native 1M context,
~45 t/s warm decode** (since 2026-10-21). Previously: Maya-M on the 170HX pair
(2026-10-20, kept below as the speed config); vLLM Flash-Next before that (one
`docker start` rollback).

## Hardware

| | |
|---|---|
| GPUs | CMP 170HX 64 GB + CMP 170HX 40 GB (GA100, BAR1-unlocked, PCIe **Gen2 x4**) + 2× RTX 3090 |
| CPU | Xeon E5-2690 v4 (AVX2) |
| RAM | 184 GB |
| Disk | NVMe — ~160 GB free needed for Maya-L (model + pack + vision) |

## Model

GLM-5.3-Flash (zai-org, MIT) — 321B-parameter MoE, ~18B active/token, native 1M context,
3:1 linear/sparse-attention hybrid: only ~14 layers even have a KV cache, so 1M context
costs ~11 GB of streamed latent KV, not hundreds of GB.

Two quants, both measured on this rig:

| Quant | Size | FP8 zero-shot | Token-identical | Warm decode |
|---|---|---|---|---|
| **Maya-L** — IQ3_S/IQ4_XS experts, Q5_K in the sensitive layers | 156.3 GB | **99.2%** | **90%** | 44-46 t/s (4 GPUs) |
| Maya-M — IQ2_S/IQ3_S, Q6_K, calibrated toward tool calls / front-end code | 116 GB | 97.9% | 86% | 50 t/s (170HX pair) |

## Topology: size-conditional GPU choice (the hard-won rule)

Maya splits by **layers** and sizes per-GPU expert pools from free VRAM.

- **Model fits the 170HX pair (~≤116 GB total experts) → `gpu: [2,3]` ONLY.** Adding the
  3090s when they're not needed is a trap: the auto-split gives them one thin layer each
  with **zero expert slots** — pure Gen2 pipeline hops. Measured: decode 20-36 t/s and
  prefill collapse (~15 t/s) from prompt buffers borrowing a packed card's expert pool.
- **Model does NOT fit the pair (Maya-L) → `gpu: [0,1,2,3]`.** The 3090s host real expert
  slots (~21 GB each), the RAM tier shrinks ~30 GB, decode nearly doubles (18.3 → 44-46
  t/s). The calculus flips because the overflow must live somewhere — 936 GB/s VRAM beats
  35 GB/s pinned RAM even with the hops.
- `MAYA_GLM_EXPERT_HELPER` needs a single-GPU trunk — not viable on a two-card trunk.
- **DFlash2 drafter: no gain here** (28-34 t/s ≈ no-draft) — on a mixed 3090/170HX pool
  the slowest card paces every draft round. Upstream's 61.6 t/s (3090 + 8× 5060 Ti) did
  not reproduce on this topology.
- Engine build is per-arch: `CMAKE_CUDA_ARCHITECTURES="80;86"` covers both card types.

**⚠️ GPU-selection gotcha (cost three failed launches):** the engine picks GPUs by
**physical NVML index** from the config's `gpu: [...]` and **ignores `CUDA_VISIBLE_DEVICES`
at serve time**. A setup run under a UUID pin with `--gpus 0,1` silently records the 3090s.
Write physical indices into the config and verify against `nvidia-smi` before starting.

## Production config (Maya-L, 1M context, 4 GPUs)

```jsonc
// maya-maya-l.json
{ "gpu": [0, 1, 2, 3],
  "args": ["--glm-pack", "<models>/Maya-L/pack", "--max-context", "1048576",
           "--expert-cache", "auto", "--prefill", "32768", "--kv-streaming", "on",
           "--expert-profile", "<data>/expert-profile-glm.bin", "--parallel", "4"],
  "env": { "MAYA_GLM_CKPT_EVERY": "0", "MAYA_GLM_RAM_HEADROOM_GB": "24" },
  "port": 8012, "host": "0.0.0.0", "api_key": "<key>" }
```

- **`--parallel 4` at 1M:** each conversation keeps its own ~11 GB latent-KV host copy;
  8× would crowd system RAM. Total pinned ≈ 128 GB of 184.
- **`--prefill 2048`** (swept on this rig: 2048→454, 8192→442, 32768→418 t/s on 4K unique
  prompts — smaller chunks borrow less expert pool; monotonic, ~+9% vs the old 32768).
  At 24 GB-card scale (community report) the effect is much larger — tune per rig.
- **`MAYA_GLM_CKPT_EVERY=0`:** fresh long reads −37% (each mid-read checkpoint piece pays
  a fixed ~14.6 s fill; upstream measured 148.8 → 93.2 s on a 100K read).
- **`--kv-streaming on`:** non-negotiable at 1M — hot window in VRAM, latent tail in
  pinned RAM. A 900K prompt reads at ~500 t/s with it; ~15 t/s without.

## Measured (this rig, warm, temp 0.7)

| Metric | Maya-L 4-GPU | Maya-M pair |
|---|---|---|
| Decode, single stream | **44-46 t/s** (cold-boot reads ~34 — always warm before quoting) | 50 t/s after calibration |
| Prefill @8K | 420-600 t/s | ~503 t/s |
| Prefill 821K (1M mode) | — | 576-594 t/s sustained, TTFT ~23 min |
| Needle @821K, 93% depth | — | **verified ×2, verbatim quote**, expert cache 93.2% at depth |
| Vision / tool calls / model-alias passthrough / wrong-key 401 | all PASS | all PASS |

## Install (non-interactive)

```sh
git clone --depth 1 https://github.com/mw00/project-maya.git && cd project-maya
./maya.sh --setup --yes --plain --model Maya-L --gpus 0,1,2,3 \
  --context 65536 --parallel 8 --port <port> --host 0.0.0.0 --api-key <key> \
  --host-compiler /usr/bin/g++-13 --no-start --download-model --no-speed-calibration
# then edit the written maya-maya-l.json: gpu indices, --max-context, --prefill, --parallel, env
```

Gotchas hit on this host:

- **Host compiler:** system gcc 15 is too new for CUDA 12.4's nvcc —
  `--host-compiler /usr/bin/g++-13` (gcc-13 installed side by side).
- **`--no-start`:** first start pins almost all free RAM for the expert tier.
- Long builds/downloads must be spawned detached
  (`subprocess.Popen(..., start_new_session=True)`) — anything tied to a terminal
  session's lifetime can be killed mid-ninja. Downloads are `curl -C -` resumable.
- **Calibration on L-4GPU wedges** (log frozen ≥90 min, process alive; M calibrated in
  15 min). Kill it and serve pre-calibration — warm-cache performance made it moot.
- Thinking model: small `max_tokens` gets eaten by reasoning before `content` — probe ≥300.

## Ops notes

- The chat path **ignores the request's `model` field** (llama.cpp-style) — legacy client
  aliases keep working. Both OpenAI (`/v1/chat/completions`, `/v1/responses`) and
  Anthropic (`/v1/messages`) APIs are served.
- **Cold 800K+ prompts cost ~20-25 min each** — load-once / interrogate-many is the right
  pattern; repeated cold huge ingests want a batched vLLM lane.
- Kill servers by the **exact PID holding the port** (`ss -tlnp`). `pgrep -f maya`
  matches your own shell (if the pattern appears in your command) and the model
  *downloader* (also `maya.py`) — put kills in a minimal standalone call, launches in
  another.
- Rollback ladder: Maya-M config (`maya-maya-m.json`, pair, 50 t/s) → vLLM container
  (`docker start`). Pause any vLLM auto-restart watchdog **before** any swap, or it
  relaunches vLLM into GPUs Maya owns.

## Host tuning (free, measured where noted)

- CPU governor `performance` + 170HX SM clock locks (`nvidia-smi -lgc 1695` / 1410) via
  `glm-perf.service` (systemd, enabled) — clock-ramp jitter removal.
- **Check HBM clocks per card:** this rig's 40 GB card advertises a 1215 MHz HBM max vs
  1728 on the 64 GB card — a 30% bandwidth deficit the placement search doesn't know
  about. Community rigs have found mismatched VBIOS HBM clocks worth +3-6% when
  equalizable; ours is a hard max. Bias hot layers toward the faster card if you can.
- Engine upgraded to v1.1.21 (prefill time-breakdown tracing, stop-during-long-prompts,
  EXL3/DFlash2 improvements). Warm decode on v1.1.21: ~38-40 t/s sustained.

## Not applicable, checked (revisit on a second 64 GB card)

- **GPUDirect Storage:** no cuFile path in the engine (O_DIRECT + pinned staging +
  `cudaMemcpy`); at mostly-experts-in-VRAM the disk tier is idle. Relevant only for a
  heavily disk-tiered model (DeepSeek-V4.1 Maya-S, 340 GB) AND a cuFile read path.
- **EAGLE-4:** no GLM-5.3-Flash draft exists (only Qwen3.8-27B), and Maya's engine has no
  EAGLE integration. Would also require vLLM/SGLang target support, which GLM lacks.
- **Flun/glm53-flash-cmp170hx-exl3** (full-HBM EXL3 + DFlash2 K7, 384K): requires a
  matched 2×64 GB pair — our 64+40 doesn't qualify. The blueprint for a true pair.
- Maya can run turboderp's EXL3 3.05/4.05bpw quants natively (with tiering), which also
  unlocks `MAYA_GLM_PREFILL_CALC` (+25-48% prompt speed on no-P2P pairs) — untested here,
  the natural next experiment.
