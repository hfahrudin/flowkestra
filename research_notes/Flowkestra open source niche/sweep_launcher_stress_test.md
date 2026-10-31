# Stress test: SSH sweep launcher (Hydra multirun + uv/PEP 723 scripts → plain list of SSH hosts, nvidia-smi free-GPU picking, no host install beyond Python/uv)

As-of date: 2026-10-31. Live data pulled via GitHub API (`gh`), PyPI JSON API and pypistats API on 2026-10-31 unless noted. I looked for disconfirming evidence first.

## Q1. Hydra health (late 2026), and is the community moving away?

### Takeaway
Hydra is **not** dying. It has been revived: it moved from `facebookresearch` to a new **`hydra-ecosystem`** org, the original author Omry Yadan is committing almost daily, 1.3.8 and 1.4.0rc1–rc3 shipped in October 2026, and downloads are 13–23M/month. The "Hydra is abandoned, so there's a gap" argument no longer holds. The "Hydra plugins are maintained, so contribute upstream or ride along" argument now does.

### Cited Findings
- `facebookresearch/hydra` now redirects to **`hydra-ecosystem/hydra`**. Not archived, 10,726 stars, last push 2026-10-30. — [GitHub API repos/facebookresearch/hydra](https://github.com/hydra-ecosystem/hydra)
- Recent releases: v1.4.0rc1 (2026-10-10), v1.4.0rc2 (2026-10-21), v1.3.8 (2026-10-25), v1.4.0rc3 (2026-10-25). PyPI "latest" stable is 1.3.8. — [GitHub releases](https://github.com/hydra-ecosystem/hydra/releases), [PyPI hydra-core](https://pypi.org/project/hydra-core/)
- 1.4.0rc3 ships core plus **all eight bundled plugins** in a coordinated bump (ax, colorlog, joblib, nevergrad, optuna, ray, rq, submitit). Security fix: "Prevent arbitrary code execution through unsafe PyYAML loaders in `hydra.utils.instantiate()`." — [v1.4.0rc3 release notes](https://github.com/hydra-ecosystem/hydra/releases/tag/v1.4.0rc3)
- Commit cadence on main (my count from the GitHub API): **47 commits in all of 2025**, **191 in Jan–Aug 2026**, **133 in Sep 1–Oct 31 2026**. Authors since 2026-01-01: Omry Yadan 263, dependabot 18, himkt 5, others ≤4 each. That is a **bus factor of 1**. — [commits](https://github.com/hydra-ecosystem/hydra/commits/main)
- Open issues: **50** (plus 1 open PR). **221 issues/PRs closed since 2026-09-01**, which looks like mass triage. Recent policy commit: "Require maintainer agreement before contributors take on issues (#3584)", 2026-10-28. — [issues](https://github.com/hydra-ecosystem/hydra/issues), [commits](https://github.com/hydra-ecosystem/hydra/commits/main)
- Roadmap issues show active pruning and planning: "Retire the Ax Sweeper after Hydra 1.4" ([#3506](https://github.com/hydra-ecosystem/hydra/issues/3506)), "Post-1.4: drop Python 3.10" ([#3461](https://github.com/hydra-ecosystem/hydra/issues/3461)), "Evaluate replacing ANTLR in the override parser" ([#3321](https://github.com/hydra-ecosystem/hydra/issues/3321)), "Explore async execution model for launchers and sweepers" ([#3201](https://github.com/hydra-ecosystem/hydra/issues/3201)).
- No "maintenance mode" statement found. The evidence points the other way, to active revival. I found no blog post or announcement explaining the org move (a web search returned nothing relevant).
- hydra-core downloads (pypistats, without mirrors), by month: 2026-04 13.0M, 05 16.1M, 06 23.1M, 07 22.0M, 08 22.0M, 09 13.3M, 10 16.4M. The last 30 days were 13.6M. — [pypistats hydra-core](https://pypistats.org/packages/hydra-core)
- Comparison, last month (pypistats): omegaconf 27.8M, tyro 6.06M, jsonargparse 1.69M, pydantic-settings 362M (inflated because it is a transitive dependency of many non-ML apps, so not comparable). — [pypistats tyro](https://pypistats.org/packages/tyro), [jsonargparse](https://pypistats.org/packages/jsonargparse), [omegaconf](https://pypistats.org/packages/omegaconf), [pydantic-settings](https://pypistats.org/packages/pydantic-settings)

### Inferences
- Hydra is still the dominant ML config/sweep CLI by downloads (around 2–3× tyro). Raw downloads include CI and transitive installs, though, so they overstate active users.
- The revival raises two risks: (a) upstream may add something that overlaps (the async launcher exploration in #3201, and the Reploy launcher in #3326, see Q2). (b) "Maintainer agreement before contributors take on issues" signals a tightly controlled, single-maintainer project, so getting an SSH launcher merged upstream is uncertain. A third-party plugin is the realistic path.
- Bus factor 1 (Omry) means the revival could reverse. Still, as of today, the claim "Hydra is legacy and people are leaving" is not supported by the data.

### Gaps
- No primary-source announcement of the facebookresearch → hydra-ecosystem move or of who funds it.
- No reliable survey data on migration away from Hydra to tyro, jsonargparse or others. Downloads are only a proxy.

## Q2. Existing Hydra launchers, and SSH launcher demand inside Hydra

### Takeaway
The official launchers (submitit, joblib, ray, rq) don't fit "plain SSH boxes". joblib is local-only, submitit is SLURM-only, Ray needs a Ray cluster, rq needs Redis plus workers. **No dedicated Hydra SSH launcher exists on PyPI.** But Mila's **cluv** (Feb 2026) already ships a `launcher=cluv` Hydra launcher over SSH, aimed at SLURM clusters. In the Hydra tracker I found **no explicit "SSH launcher" feature request**. That is weak evidence of low demand from the Hydra side.

### Cited Findings
- Bundled plugins in the repo: hydra_ax_sweeper, hydra_colorlog, hydra_joblib_launcher, hydra_nevergrad_sweeper, hydra_optuna_sweeper, hydra_ray_launcher, hydra_rq_launcher, hydra_submitit_launcher. No SSH launcher. — [repo plugins/](https://github.com/hydra-ecosystem/hydra/tree/main/plugins)
- Launcher downloads, last month: hydra-submitit-launcher 67.3K, hydra-joblib-launcher 42.8K, hydra-ray-launcher 3.4K, hydra-rq-launcher 195. — pypistats ([submitit](https://pypistats.org/packages/hydra-submitit-launcher), [joblib](https://pypistats.org/packages/hydra-joblib-launcher), [ray](https://pypistats.org/packages/hydra-ray-launcher), [rq](https://pypistats.org/packages/hydra-rq-launcher))
- **cluv exists** (verified): `mila-iqia/cluv`, created 2026-02-12, last push 2026-10-30, 30 stars, described as "A powerful and lightweight CLI tool to use UV and submit jobs across multiple clusters". Requirements: "SSH access configured for each cluster in `~/.ssh/config`" and "A GitHub repository with your project". It has an optional Hydra launcher: `uv add cluster-uv[hydra]`, and its docs show `python main.py -m launcher=cluv lr=0.01,0.001 seed=1,2,3`. PyPI package `cluster-uv` gets about 1.5K downloads/month. — [GitHub mila-iqia/cluv](https://github.com/mila-iqia/cluv), [cluv docs](https://mila-iqia.github.io/cluv/), [PyPI cluster-uv](https://pypi.org/project/cluster-uv/)
- cluv's docs target "HPC clusters" with `submit` / `squeue`. They don't say whether plain workstations without SLURM or GPU selection are supported (`cluv run` runs a command on a cluster). — [cluv docs](https://mila-iqia.github.io/cluv/)
- `hydra-ssh` on PyPI is **not** a Hydra plugin. It is a standalone asyncssh tool that runs one command across hosts listed in a CSV. — [PyPI hydra-ssh](https://pypi.org/project/hydra-ssh/)
- `hydra-ray-jobs-launcher` is a third-party Ray Jobs launcher (it needs Ray). — [PyPI hydra-ray-jobs-launcher](https://pypi.org/project/hydra-ray-jobs-launcher/)
- Searching the Hydra tracker for "ssh" returned only incidental hits (a Ray launcher config bug, a SLURM question). Examples: [#2927 "Ray Launcher: Additional properties are not allowed"](https://github.com/hydra-ecosystem/hydra/issues/2927), [#3058 "How to Submit SLURM Jobs Eagerly?"](https://github.com/hydra-ecosystem/hydra/issues/3058). Nearest launcher-design threads: [#2874 "Dynamic hydra launcher configuration"](https://github.com/hydra-ecosystem/hydra/issues/2874) (2024, open) and [#3201 async launchers/sweepers](https://github.com/hydra-ecosystem/hydra/issues/3201) (2026-06, open).
- [#3326 "Add a Reploy launcher for immutable, reproducible job execution"](https://github.com/hydra-ecosystem/hydra/issues/3326) (open, 2026-07-29). Motivation: "A long-running Hydra job or multirun can remain exposed to changes in the live working tree… Hydra records the composed configuration and overrides, but that does not preserve the executable environment." Reploy is hosted at reploy.yadan.net, i.e. Omry Yadan's own domain.

### Inferences
- The Hydra creator is building his own launcher-adjacent execution/reproducibility tool (Reploy). If Reploy grows remote-host support, it will occupy the exact niche with upstream blessing. This is the single biggest Hydra-side threat. Whether it supports SSH is **unverified**: the site didn't resolve when I tried.
- No SSH launcher requests in a heavily used tracker suggests Hydra users with multiple boxes either have SLURM/Ray or use W&B agents and scripts instead.

### Gaps
- Couldn't load reploy.yadan.net (DNS failure), so Reploy's scope (local only vs remote) is unknown.
- GitHub Discussions weren't searched (the issue search API doesn't cover them).
- Didn't verify whether cluv's Hydra launcher works against non-SLURM hosts.

## Q3. Non-Hydra alternatives that fan sweeps out over SSH

### Takeaway
"Sweep across my 3–5 SSH boxes" is **already covered by several tools**. The most damaging find is **mlsweep** (first release Mar 2026, v3.2.0 on 2026-10-07): a manager that bootstraps workers over SSH ("no install needed"), schedules runs across per-host GPU lists, uploads code, installs dependencies from pyproject, rsyncs artifacts back, and has a dashboard plus Optuna mode. That is close to this exact idea. Others: gpuc (Sep 2026, SSH + rsync + NVIDIA driver only), MeinSweeper (SSH nodes plus GPU management, stale since 2024), SkyPilot SSH Node Pools and dstack SSH fleets (heavier, need sudo or Docker), W&B agents launched by hand on each box (de-facto practice), Optuna with a shared DB, and GNU parallel `--sshlogin`. On the other side of the ledger: almost all of these see little use (mlsweep about 460 downloads/month), and none clearly covers **Windows/macOS hosts** or **Hydra-multirun-native** dispatch.

### Cited Findings
- **mlsweep** (PyPI): "Remote workers get mlsweep bootstrapped automatically over SSH, no install needed." "It builds wheels from the local source at startup, SCPs them to the remote, and installs them into `/tmp/mlsweep_venv/`. Your code is uploaded with each sweep. Its dependencies are installed on the remote from `pyproject.toml` / `requirements.txt` and cached per machine." `workers.toml` takes `host`, `remote_dir`, `ssh_key`, `devices = [0,1,2,3]`, `jobs`. It also claims "`mlsweep` does pretty much everything that wandb does." Releases: 1.0.0 on 2026-03-13 through 3.2.0 on 2026-10-07 (fast iteration). Downloads: 463 last month. — [PyPI mlsweep](https://pypi.org/project/mlsweep/), [pypistats](https://pypistats.org/packages/mlsweep)
  - Caveat: the `/tmp/mlsweep_venv/` paths and the sshpass option imply POSIX hosts. No Windows mention seen. GPUs come from a static `devices` list per worker, and I saw no evidence of nvidia-smi free-memory probing in the README excerpt.
- **gpuc**, Brendan Long, 2026-09-29: "I need to queue work on various GPUs to run experiments, and I want to do that as cheaply as possible." "Queue hosts only need SSH, rsync, and the NVIDIA driver." It is decentralized, with priority queueing, preemption, RunPod auto-shutdown, and a "CLI optimized for Claude use". He says such a tool "doesn't seem to exist", lists SkyPilot and dstack as alternatives and explains why they didn't fit. A commenter shared their own vibe-coded equivalent. — [LessWrong gpuc post](https://www.lesswrong.com/posts/PPzBbskPM2Baz7CL8/gpuc-a-single-user-gpu-queue-that-doesn-t-require-sudo)
- **MeinSweeper**: "a light-weight framework for running experiments on arbitrary compute nodes" with SSH and local node types plus GPU management. Last release 0.3.8 on 2024-11-21 (looks stalled). — [PyPI meinsweeper](https://pypi.org/project/meinsweeper/), [GitHub afspies/meinsweeper](https://github.com/afspies/meinsweeper)
- Other 2026 single-author SSH GPU schedulers on GitHub (0 stars): `aapdo/research-job-scheduler`, "Research-independent SSH experiment scheduler with RQ metadata, DAG dependencies, GPU admission and failure recovery" (pushed 2026-09-26), and `Threeausj/deepqueue`, "GPU experiment queues with Codex app-server: SSH scheduling, live tmux logs" (2026-09-11). Also an ARIS "experiment-queue" Claude Code skill: an SSH job queue for ML grids with OOM-aware retry. — [research-job-scheduler](https://github.com/aapdo/research-job-scheduler), [deepqueue](https://github.com/Threeausj/deepqueue), [ARIS skill](https://openskillindex.com/skills/wanshuiyin-auto-claude-code-research-in-sleep-experiment-queue), [gpu-experiment-scheduler (single machine)](https://github.com/comiam/gpu-experiment-scheduler)
- **SkyPilot SSH Node Pools**: `~/.sky/ssh_node_pools.yaml`, then `sky ssh up` "installs the necessary dependencies", then `sky launch --infra ssh/my-cluster --gpus H100:1`. Requirements: "Debian-based OS (tested on Debian 11)", "Nodes should not be part of an existing Kubernetes cluster", "All nodes within a SSH Node Pool must have access to port 6443 to its peers". There is a password option if "passwordless sudo is not enabled", so it needs sudo and deploys Kubernetes onto the boxes. — [SkyPilot docs: existing machines](https://docs.skypilot.ai/en/latest/reservations/existing-machines.html)
- **dstack SSH fleets**: "Hosts must be Linux-based and have Docker pre-installed." "Hosts with NVIDIA GPUs must also be pre-installed with CUDA 12.1 and NVIDIA Container Toolkit." "The user specified must have passwordless `sudo` access." `AllowTcpForwarding yes` is required. Hosts can be split into `blocks` for multiple jobs per host. — [dstack fleets docs](https://dstack.ai/docs/concepts/fleets/)
- **W&B sweep agents**: the official parallelization page covers only one host: "launch parallel agents on multi-CPU and multi-GPU machines so you can use all available compute on a single host". It says to "Run sweep agents in parallel across multiple GPUs on the same machine by using `CUDA_VISIBLE_DEVICES`", one `wandb agent <sweep_id>` per GPU. Multi-machine use means running the same command on each box by hand. (W&B docs now redirect to docs.coreweave.com following the CoreWeave acquisition.) — [W&B/CoreWeave docs: parallelize agents](https://docs.coreweave.com/models/sweeps/parallelize-agents), [W&B report "Multi-GPU Hyperparameter Sweeps in Three Simple Steps"](https://wandb.ai/wandb_fc/articles/reports/Multi-GPU-Hyperparameter-Sweeps-in-Three-Simple-Steps--Vmlldzo1NDQ0MDEy)
- **Optuna**: distributed optimization works by pointing several processes, on any machines, at one RDB storage URL. You still start the workers on each host yourself. — [Optuna tutorial: easy parallelization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html) (well-known documented feature; not re-fetched this session)
- **GNU parallel** `--sshlogin` / `--slf` spreads a job list across SSH hosts, with `--transferfile`/`--return` for files. It has no GPU awareness (you can approximate with the `{%}` slot number mapped to `CUDA_VISIBLE_DEVICES`). — [GNU parallel manual](https://www.gnu.org/software/parallel/parallel_tutorial.html) (not re-fetched this session)
- **Dask SSHCluster** and **Ray manual clusters** need dask/ray installed on every host and a long-lived scheduler or head node. That violates "nothing but Python/uv". — [Dask SSHCluster docs](https://docs.dask.org/en/stable/deploying-ssh.html), [Ray on-prem cluster docs](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html) (not re-fetched this session)
- **ClearML agent** needs the agent installed and running on each host, plus a ClearML server. — [ClearML agent docs](https://clear.ml/docs/latest/docs/clearml_agent) (not re-fetched this session)

### Inferences
- For a W&B user, the de-facto answer is `ssh boxN "CUDA_VISIBLE_DEVICES=k wandb agent ID"` in a loop or tmux. The marginal value of a new tool for them is small: free-GPU detection, code sync, a single command.
- For a non-W&B user, **mlsweep is already a near-clone of the proposal**, minus Hydra-native integration, Windows/macOS support and dynamic free-GPU probing. gpuc covers the no-sudo, queue-centric version.
- The pattern of four or more one-person SSH GPU schedulers appearing in 2026, all with tiny adoption, says two things at once. **The pain is real** (people keep building it). **The market is fragmented**: each person writes their own in an afternoon, often with an AI agent. A new entrant faces an "everyone has their own script" problem and a "can't get attention" problem.
- SkyPilot and dstack are the "serious" answers, but their sudo, Docker or Kubernetes requirements leave room for a zero-install tool. mlsweep and gpuc are already in that room.

### Gaps
- Didn't verify whether mlsweep, gpuc or MeinSweeper probe nvidia-smi for *free* GPUs (vs static device lists), or whether any supports Windows hosts.
- No usage numbers for gpuc (it isn't on PyPI as far as I found).

## Q4. Does uv have or plan remote/sweep features? uv-based experiment runners?

### Takeaway
I found no evidence that uv has or plans remote execution or sweep features. The uv-based remote runners that exist are third-party: cluv (Mila) and, indirectly, mlsweep's pyproject-based remote dependency install.

### Cited Findings
- A web search for uv SSH or remote-run feature requests found only an unrelated commit about SSH Git dependencies ("Clarify adding SSH Git dependencies (#13534)"). A GitHub issue search on astral-sh/uv for "remote ssh run" in titles returned nothing. — [uv commit](https://app.semanticdiff.com/gh/astral-sh/uv/commit/1bee9320f157f064a1781fbf9e5868de843457df)
- cluv: "sync and submit UV-based Python projects across HPC clusters". — [GitHub mila-iqia/cluv](https://github.com/mila-iqia/cluv)

### Inferences
- uv itself is unlikely to compete. Astral's focus is packaging, plus their pyx registry, which I didn't verify here. The "uv on the remote does the environment" design is sound and already used by cluv.

### Gaps
- I didn't do an exhaustive search of uv issues or discussions. An empty result from one title search is weak evidence.

## Q5. Demand: posts about running sweeps across several workstations

### Takeaway
I found real but **thin, builder-side** demand: people writing their own tools (gpuc, mlsweep, deepqueue, research-job-scheduler) rather than many end users asking for one. I found **no Reddit or HN threads** in my searches. The strongest demand quote is from the gpuc author.

### Cited Findings
- "I need to queue work on various GPUs to run experiments, and I want to do that as cheaply as possible". The author felt such a tool "doesn't seem to exist" (2026-09-29). — [LessWrong gpuc](https://www.lesswrong.com/posts/PPzBbskPM2Baz7CL8/gpuc-a-single-user-gpu-queue-that-doesn-t-require-sudo)
- mlsweep author: the sweep config is "The thing that I've been missing all my machine learning life, and the reason I wrote this library." Yet it gets only 463 downloads/month. — [PyPI mlsweep](https://pypi.org/project/mlsweep/), [pypistats](https://pypistats.org/packages/mlsweep)
- W&B's official docs only address one host. Multi-box use is left to the user. — [docs.coreweave.com parallelize agents](https://docs.coreweave.com/models/sweeps/parallelize-agents)
- No SSH-launcher requests in the Hydra issue tracker (see Q2).

### Inferences
- The target user (3–5 personal or lab workstations, no SLURM) exists: academic labs, indie researchers, small startups. But they are comfortable with scripts, and many already use W&B. Willingness to adopt *someone else's* tool looks low, as shown by mlsweep's and cluv's sub-2K monthly downloads despite active development.

### Gaps
- The Reddit and HN searches returned no matching threads (the search tool surfaced none). This is a real gap, not proof of absence. A manual search on r/MachineLearning and HN Algolia for "multiple workstations sweep" is recommended.

## Q6. Verdict

### Takeaway
**Marginal, leaning dead end as a standalone "new tool".** It is viable only as a very narrow wedge: a **Hydra launcher plugin** (`hydra/launcher=ssh`) plus a PEP 723 mode with **Windows/macOS hosts** and **dynamic nvidia-smi free-GPU selection**, positioned against mlsweep and cluv rather than in empty space.

### Cited Findings
- Hydra is alive and actively maintained (Oct 2026 releases, 133 commits in Sep–Oct), so a plugin has a stable host. The single maintainer is also building Reploy, a launcher-adjacent tool. — [releases](https://github.com/hydra-ecosystem/hydra/releases), [#3326](https://github.com/hydra-ecosystem/hydra/issues/3326)
- Direct competitors already shipped in 2026: mlsweep (SSH bootstrap, GPU lists, dependency install, dashboard), cluv (uv + SSH + Hydra launcher, for SLURM clusters) and gpuc (SSH, rsync, NVIDIA driver only). — [mlsweep](https://pypi.org/project/mlsweep/), [cluv](https://github.com/mila-iqia/cluv), [gpuc](https://www.lesswrong.com/posts/PPzBbskPM2Baz7CL8/gpuc-a-single-user-gpu-queue-that-doesn-t-require-sudo)
- Heavy alternatives need sudo, Docker or Kubernetes (SkyPilot, dstack). The zero-install gap they leave is already being filled. — [SkyPilot](https://docs.skypilot.ai/en/latest/reservations/existing-machines.html), [dstack](https://dstack.ai/docs/concepts/fleets/)

### Inferences
- **Reasons it's a dead end:** (1) The core claim "nothing does SSH sweeps with zero install" is false as of 2026 (mlsweep, gpuc). (2) The W&B-agent-per-box habit covers most W&B users with 10 lines of bash. (3) The space is flooded with one-person, AI-assisted clones getting under 2K downloads/month, so distribution is the bottleneck, not the technology. (4) Upstream Hydra (Reploy, async launchers) or cluv could add plain-SSH support cheaply.
- **Reasons it's still marginally viable:** (1) No `hydra-*-launcher` for plain SSH exists on PyPI, and Hydra launchers get real traffic (joblib 43K/month, submitit 67K/month), so a plugin is discoverable through Hydra's ecosystem. (2) Windows hosts and *dynamic* free-GPU selection look unaddressed by mlsweep, gpuc and SkyPilot/dstack (Linux-only). (3) Small scope means low build cost, roughly a weekend for the MVP.
- **Recommendation:** Don't build it as a flagship project. If you build it, ship it as a thin `hydra-ssh-launcher` plugin plus a `uv run` script mode, under 1K lines of code. Make Windows/macOS and nvidia-smi free-memory probing the headline features. Measure interest within 4–6 weeks (stars, downloads compared with mlsweep's ~460/month as the bar), and abandon it quickly if it stays flat. Before writing code, check Reploy's scope and whether cluv works on non-SLURM hosts. Either could erase the wedge.

### Gaps
- Reploy's scope, cluv's non-SLURM support, and Windows support in mlsweep and gpuc are all unverified, and each could change the verdict toward "dead end".
- I gathered no direct user-interview or Reddit/HN demand evidence.
