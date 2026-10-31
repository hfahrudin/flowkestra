# Zero-install / no-sudo / no-Docker experiment runner across SSH GPU hosts: competition and demand (as of 2026-10-31)

Note on method: repo stats (stars, last push, license) come from the GitHub API, queried 2026-10-31. Web searches did not return any Reddit threads. Demand evidence is mostly indirect: GitHub issues, blog posts, and the number of new homemade tools.

## Q1. Which tools run jobs across several SSH hosts without root, Docker, Kubernetes or a daemon install?

### Takeaway
None of the big orchestrators qualify. SkyPilot SSH Node Pools installs k3s with sudo. dstack SSH fleets require Docker plus passwordless sudo. Ray needs a pip install and open ports on every node. But 2026 has produced a burst of tiny tools (roughly 0–2 stars, most created Sept–Oct 2026) that target exactly this "SSH + rsync + nvidia-smi only" niche: gsched-gtop, gpu-coordinator/gpuc, GPUOps, gpu-spooler, gpuq and others. The idea itself is not empty; what is missing is a mature, well-known tool that does it.

### Cited Findings
**Major platforms (fail the no-sudo/no-Docker test)**
- SkyPilot SSH Node Pools prerequisites: a Debian-based OS (tested on Debian 11), SSH from the API server, nodes able to reach each other on port 6443, and GPU drivers already installed. The docs only describe a `password` field for use "if passwordless sudo is not enabled" — [SkyPilot docs](https://docs.skypilot.ai/en/latest/reservations/existing-machines.html)
- SkyPilot's SSH node pool install runs `sudo -E -A` to pass `INSTALL_K3S_EXEC`, `K3S_URL`, `K3S_TOKEN` to the k3s installer, so a node pool is a k3s cluster installed with root (issue opened 2026-05-21) — [skypilot#9680](https://github.com/skypilot-org/skypilot/issues/9680)
- Without passwordless sudo, users hit "sudo: a terminal is required to read the password" (issue opened 2026-07-30, still open) — [skypilot#10303](https://github.com/skypilot-org/skypilot/issues/10303)
- SkyPilot: 10,770 stars, Apache-2.0, very active (pushed 2026-10-31) — [GitHub](https://github.com/skypilot-org/skypilot)
- dstack SSH fleets: "Hosts must be Linux-based and have Docker pre-installed" and "The user specified must have passwordless sudo access" — [dstack docs](https://dstack.ai/docs/concepts/fleets/)
- dstack's shim panicked with "cannot get Docker info" when hosts lacked Docker (2024-10-15); this was fixed by failing cleanly, not by dropping the Docker requirement — [dstack#1834](https://github.com/dstackai/dstack/issues/1834). An open-ended issue floats a static, non-root OpenSSH build to relax image requirements — [dstack#1535](https://github.com/dstackai/dstack/issues/1535). dstack: 2,298 stars, MPL-2.0.
- Ray manual on-prem setup: install Ray on each node, run `ray start --head --port=6379`, and point workers at the head's address. Connection failures are blamed on firewalls or version mismatches. The docs say nothing about root — [Ray docs](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html). Ray: 44,169 stars.
- gpuc's author dismisses the major tools: SkyPilot and dstack have problems for his use case, and their SSH clusters "require sudo" (2026-09-28) — [Brendan Long blog](https://www.brendanlong.com/gpuc-a-single-user-gpu-queue-that-doesnt-require-sudo.html)

**Single-host queues (no sudo, but no multi-host)**
- pueue: 6,403 stars, Apache-2.0, active (pushed 2026-10-27). It works per machine; remote use means connecting to a remote daemon over TCP or a socket (e.g. issues [#557](https://github.com/Nukesor/pueue/issues/557), [#543](https://github.com/Nukesor/pueue/issues/543)). It is not GPU-aware and has no fleet view — [GitHub](https://github.com/Nukesor/pueue)
- task-spooler (justanhduc GPU fork): 446 stars, GPL-2.0, last push 2024-03-06 (stale) — [GitHub](https://github.com/justanhduc/task-spooler)
- simple_gpu_scheduler: 189 stars, GPL-3.0, last push 2021-07-07 (abandoned) — [GitHub](https://github.com/ExpectationMax/simple_gpu_scheduler)
- gpuq (lzy248): 2 stars, MIT, created 2026-07-21. A per-user opportunistic queue for "unmanaged shared servers"; it waits for free VRAM. Single host — [GitHub](https://github.com/lzy248/gpuq)
- compute-queue (ap08960695): 0 stars, MIT, created 2026-10-20. "Small Python job queue for shared GPU/Mac machines: measured profiles, hard memory caps, backfill, lease preemption" — [GitHub](https://github.com/ap08960695/compute-queue)
- gflow: search describes it as a single-machine scheduler that runs each job in a tmux session. I could not pin down the canonical repo, so its stats are unverified.
- qjob: a lab-scale scheduler for one GPU server — [GitHub](https://github.com/ShineiArakawa/qjob). gpu-experiment-scheduler: one machine only — [GitHub](https://github.com/comiam/gpu-experiment-scheduler)

**Multi-host, agentless or user-space (the direct competitors)**
- **gsched-gtop** (zephyr-zdz): MIT, 0 stars, a single commit on 2026-10-28. One controller uses ssh+rsync. GPU hosts need only python3, nvidia-smi, rsync and bash/setsid/nohup, and "nothing from gsched is installed remotely". It has host tiers with fallback and Slurm overflow, idle detection with memory/utilization thresholds held over time, memory-declared packing on shared cards, and a `--retries` requeue. Its README warns that a controller crash can start a job twice and that concurrent outputs can overwrite each other — [GitHub](https://github.com/zephyr-zdz/gsched-gtop)
- **gpu-coordinator / gpuc** (brendanlong): MIT, 1 star, created 2026-09-15. "Queue hosts only need SSH, rsync, and the NVIDIA driver." It is decentralized: one dispatcher per host, with no cross-host prioritization. It supports RunPod rentals with health checks and auto-shutdown, has a CLI built for Claude plus a web dashboard and Snakemake integration, and marks cards as "owned" or "shared". It has no multi-user support — [blog 2026-09-28](https://www.brendanlong.com/gpuc-a-single-user-gpu-queue-that-doesnt-require-sudo.html), [GitHub](https://github.com/brendanlong/gpu-coordinator)
- **GPUOps** (OHHHHHHei): MIT, 1 star, created 2026-10-22, v0.2.1. "The Worker needs no Docker, sudo, systemd, or server-side HTTP service." It does run a user-level worker that needs Python 3.10+ venv and about 1 GiB of home quota, so it is not strictly zero-install. Placement is `host: "auto"` or a named host. It double-checks idle GPUs, binds jobs to GPU UUIDs, keeps a durable queue, and has a `reconcile` command. It does no file sync, and the worker does not survive a reboot. It is built for terminal agents — [GitHub](https://github.com/OHHHHHHei/GPUOps)
- **gpu-spooler** (woodRock): 0 stars, no license detected, created 2026-04-03, 91 commits, pushed 2026-10-31. A laptop-side daemon polls nvidia-smi across 30 servers / 78 GPUs at VUW. It has a GPU cap, model pinning, minimum memory, git-snapshot sync (`task sync`) and result fetching. It is hard-wired to the VUW jump hosts — [GitHub](https://github.com/woodRock/gpu-spooler)
- **gpu-jobq** (tangang8): 0 stars, MIT, created 2026-10-01. Multi-machine mode needs a shared filesystem with atomic rename and cross-machine locks — [GitHub](https://github.com/tangang8/gpu-jobq)
- **cowork** (rrrrrrzy): needs shared NFS and runs per-node agents in tmux — [GitHub](https://github.com/rrrrrrzy/cowork)
- **ΣΣJob / sumsjob** (lululxvi): 12 stars, GPL-3.0, created 2019, last push 2024-08-16. It "submits a job to one of the multiple GPU servers" — [GitHub](https://github.com/lululxvi/sumsjob). This is the oldest prior art found.
- gpuq (gluesys): a shared GPU queue for SSH research users, with a Korean README — [GitHub](https://github.com/gluesys/gpuq)
- The "experiment-queue" Claude Code skill handles ML experiment batches on SSH remote GPU servers, with state tracking, OOM retry and stale-job cleanup — [skills.sh](https://skills.sh/wanshuiyin/auto-claude-code-research-in-sleep/experiment-queue)
- Monitors only (no queue): gpuwho is "nvidia-smi for multiple servers… Nothing to install on the servers" — [GitHub](https://github.com/htyvv/gpuwho); gpu-monitor was built to speed up finding a free GPU in institutions that share multiple GPU servers — [GitHub](https://github.com/mseitzer/gpu-monitor)

**Others on the checklist**
- ClearML agent: 315 stars, Apache-2.0, active — [GitHub](https://github.com/clearml/clearml-agent). Install requirements on hosts were not verified this session.
- dask-jobqueue: 256 stars, BSD-3. It targets PBS/SLURM/SGE, not raw SSH hosts — [GitHub](https://github.com/dask/dask-jobqueue)
- Lightning (31,435 stars) is a training framework, not a multi-host SSH queue — [GitHub](https://github.com/Lightning-AI/pytorch-lightning)

### Inferences
- In late 2026 the niche is crowded with homemade, low-maturity tools. At least 6 multi-host SSH queues were created in 2026, none with more than about 2 stars. That reads as broad latent demand that no single tool has captured, rather than open white space. Being first is not possible; the remaining edges are polish, robustness, multi-user fairness and discoverability.
- Two of the newest tools (gpuc, GPUOps), plus a Claude Code skill, are explicitly designed to be driven by AI coding agents. This looks like the active 2026 trend.
- These tools differ on the hard parts: file sync (gsched/gpu-spooler do it, GPUOps doesn't), a persistent remote worker (GPUOps) versus purely agentless (gsched/gpuc), and central versus decentralized queue state.
- pueue is the mature single-host baseline (6.4k stars). A thin multi-host, GPU-aware layer over plain SSH has no mature equivalent.

### Gaps
- Not verified this session: Dask SSHCluster requirements (needs dask installed on each host), GNU parallel `--sshlogin` (needs no remote install but is not GPU-aware; no source fetched), Fabric/Ansible runners, W&B Launch, Modal (not applicable: hosted compute only), ClearML agent host requirements, and Slurm-without-root options.
- The canonical gflow and gpueue repos were not identified (kaito4681/gpueue exists with 0 stars, pushed 2026-10-31; not inspected).
- Ray non-root feasibility is my inference: a pip/venv install plus `ray start` on high ports should not need root, but no source confirms it.

## Q2. Demand evidence: issues, Reddit, HN, forums

### Takeaway
Direct demand from the big projects is thin: only a couple of sudo-related SkyPilot issues, and dstack treats Docker as a fixed requirement. The strongest demand signal is people repeatedly building this tool for themselves in 2026. Web search turned up no Reddit or HN discussion threads, so community pull is unproven.

### Cited Findings
- SkyPilot users on non-sudo hosts hit "sudo: a terminal is required to read the password" (2026-07-30, open) — [skypilot#10303](https://github.com/skypilot-org/skypilot/issues/10303)
- A dstack user tried SSH fleets on hosts without Docker and the fleet hung in `pending` (2024-10-15) — [dstack#1834](https://github.com/dstackai/dstack/issues/1834)
- gpuc's motivation: cheap experiments spread across a personal GPU, SSH-accessible remote GPUs and RunPod; "RunPod clusters silently hang forever in the (very common) case where the pod is completely broken"; SkyPilot/dstack SSH clusters "require sudo" (2026-09-28) — [blog](https://www.brendanlong.com/gpuc-a-single-user-gpu-queue-that-doesnt-require-sudo.html); also cross-posted to [LessWrong](https://www.lesswrong.com/posts/PPzBbskPM2Baz7CL8/gpuc-a-single-user-gpu-queue-that-doesnt-require-sudo)
- gpu-spooler was built for the VUW ECS cluster: 30 servers and 78 GPUs reached through jump hosts with OTP, with actual servers recorded 2026-10-17 — [GitHub](https://github.com/woodRock/gpu-spooler)
- The current manual workflow (nvidia-smi to find free GPUs, tmux to persist runs) appears in university GPU server docs — [wi2trier/gpu-server](https://www.github.com/wi2trier/gpu-server) — and is the motivation for gpu-monitor and gpuwho ([gpu-monitor](https://github.com/mseitzer/gpu-monitor), [gpuwho](https://github.com/htyvv/gpuwho))

### Inferences
- The demand is real but individual and low in intensity. Each person writes a few hundred lines of scripts and moves on, which is why none of these tools gains stars. A new entrant would face the same adoption problem: users see the problem as "small enough to script."
- The evidence that pain is small is that tmux + nvidia-smi + rsync remains the default workflow, and a monitor (gpuwho) is often enough.

### Gaps
- No Reddit (r/MachineLearning, r/LocalLLaMA, r/HPC) or HN threads were found through search. Searching those sites directly would be needed to quote user complaints.
- No quantitative data on how many users want this.

## Q3. How common is "no sudo / no Docker" on academic GPU servers?

### Takeaway
It is plausible and anecdotally common, but I found no survey. One university doc shows that even sudo users are restricted (no apt; Nix-managed). Tool authors cite no-sudo as their reason for building.

### Cited Findings
- Trier University's 8×V100 ML server: software is managed declaratively with Nix, users cannot install via apt even with sudo, access is VPN-only, GPUs are in exclusive compute mode, and users rely on nvidia-smi and tmux — [wi2trier/gpu-server](https://www.github.com/wi2trier/gpu-server)
- gpu-spooler's VUW setup: access is only via jump hosts with OTP, and nothing beyond python3 is needed on servers — [GitHub](https://github.com/woodRock/gpu-spooler)
- nvidia-smi stats can be read as a normal user, so GPU-aware placement is possible without root — [Linuxize](https://linuxize.com/post/nvidia-smi-command-in-linux/), [computingforgeeks](https://computingforgeeks.com/nvidia-smi-command-linux/)

### Inferences
- Shared lab and university machines typically lack passwordless sudo and Docker group membership (Docker group ≈ root). This rules out SkyPilot SSH pools and dstack SSH fleets in that environment.

### Gaps
- No survey or quantitative source on the share of academic GPU servers without sudo or Docker.

## Q4. Is a multi-host agentless queue technically hard?

### Takeaway
A basic version is easy, which is why many hobby versions exist. The edge cases are where tools are fragile: exactly-once launch after a controller crash, reboot recovery, output collisions, contention with other users' processes, and the lack of a remote daemon when the laptop is offline. The existing tools document these as known limitations.

### Cited Findings
- gsched: a controller crash at a specific moment can start a job twice; jobs that vanish without an exit code need `--retries`; unreachable hosts keep their jobs but get no new placements; concurrent experiments writing the same output paths can overwrite each other; rsync uploads do not delete — [GitHub](https://github.com/zephyr-zdz/gsched-gtop)
- GPUOps: a durable queue that survives the control machine going offline, a `reconcile` command for uncertain submissions, idle GPUs confirmed twice, and cancellation that only signals verified managed processes; the worker does not restart after reboot — [GitHub](https://github.com/OHHHHHHei/GPUOps)
- gpuc: being decentralized avoids cross-host coordination but gives up global prioritization; it needs health checks and timeouts for broken rentals — [blog](https://www.brendanlong.com/gpuc-a-single-user-gpu-queue-that-doesnt-require-sudo.html)
- Contention with other users: gsched treats a GPU as idle only after a quiet period under memory/utilization thresholds and never signals others' processes — [GitHub](https://github.com/zephyr-zdz/gsched-gtop); gpuq waits for a stable free-VRAM window without killing or reserving memory from anyone — [GitHub](https://github.com/lzy248/gpuq); gpuwho defines idle as no processes, under 1 GiB used and under 5% load — [GitHub](https://github.com/htyvv/gpuwho)
- Truly agentless means the laptop or controller must stay up to dispatch queued jobs; gpuc and GPUOps solve this with a per-host user-level dispatcher or worker — [gpuc blog](https://www.brendanlong.com/gpuc-a-single-user-gpu-queue-that-doesnt-require-sudo.html), [GPUOps](https://github.com/OHHHHHHei/GPUOps)
- Shared-filesystem designs (gpu-jobq, cowork) depend on atomic rename and cross-host locks over NFS — [gpu-jobq](https://github.com/tangang8/gpu-jobq), [cowork](https://github.com/rrrrrrzy/cowork)

### Inferences
- The defensible part of a new tool would be correctness and UX: idempotent launches (e.g. remote PID/lockfiles plus reconciliation), reboot detection, per-run working-directory isolation, multi-user fair-share without a daemon, environment provisioning (venv/uv/conda) and agent-friendly JSON output. None of these is novel, but no existing tool has all of them with any user base.
- Overall verdict: the niche is wanted but not unfilled. It is filled by many immature 2026 tools and dominated in practice by DIY scripts. Prospects depend on out-executing and out-distributing these, not on unique capability.

### Gaps
- No usage or download data for these small tools (PyPI stats were not checked).
