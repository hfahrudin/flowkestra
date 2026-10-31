# Execution Layer for AI Agents Running ML Experiments: Landscape & Demand (as of 2026-10-31)

Method note: GitHub stats were pulled live via `gh api` on 2026-10-31 (stars / last push). Other facts come from project READMEs, vendor blogs, HN, and GitHub discussions. Vendor numbers are self-reported. Some named targets could not be found (see Gaps).

## 1. Map of the field: what exists, backends, tracking, safety, maturity

### Takeaway
The "loop" layer (autoresearch skills and harnesses) is very crowded, with dozens of repos over 1k stars. The "compute launch" layer is owned by infra vendors (SkyPilot, RunPod, VESSL, Modal, Tinker), and the "tracking/query" layer by MLflow and W&B MCPs. The specific slice of a safe, policy-constrained, backend-agnostic run executor plus ledger for agents is populated only by tiny projects (0–4 stars). That is either an open niche or a sign of weak demand for it as a standalone thing.

### Cited Findings
**Origin / reference loop**
- karpathy/autoresearch: created 2026-03-06, about 98.9k stars and 13.6k forks, last push 2026-03-26 (effectively frozen). The agent edits `train.py` and trains for a fixed 5 minutes on 1 GPU. It keeps a change only if `val_bpb` improves and logs each attempt to `results.tsv` (commit, score, memory, status, description). `prepare.py` is read-only. Announced 2026-03-07. — [GitHub](https://github.com/karpathy/autoresearch); [DataCamp guide](https://www.datacamp.com/tutorial/guide-to-autoresearch); gh api 2026-10-31
- About 100 experiments per overnight run on one GPU. — [HN/NYU Shanghai summary](https://rits.shanghai.nyu.edu/ai/karpathy-open-sources-autoresearch-100-ai-experiments-overnight-on-one-gpu/)

**Forks and harnesses (loop layer), stars as of 2026-10-31 via gh search**
- aiming-lab/AutoResearchClaw: 14.8k stars (idea→paper). davebcn87/pi-autoresearch: 8.2k. uditgoenka/autoresearch (Claude Code skill): 6.8k. leo-lilinxiao/codex-autoresearch: 3.1k. EvoMap/AutoResearch: 3.1k. OpenNSWM-Lab/FAROS ("AutoResearch runtime"): 3.0k. webfuse-com/awesome-autoresearch: 2.6k. hyperspaceai/agi (distributed): 2.1k. autoresearch-mlx: 1.9k. RightNow-AI/autokernel: 1.6k. evo-hq/evo: 1.5k. Human-Agent-Society/CORAL: 1.1k. RUC-NLPIR/Arbor: 1.1k. Most were pushed in Sep–Oct 2026, so they are actively maintained. — gh search repos "autoresearch" (2026-10-31)
- wanshuiyin/Auto-claude-code-research-in-sleep (ARIS): 18.3k stars, created 2026-03-10, pushed 2026-10-30. It is markdown-only skills (cross-model review, idea discovery, experiment automation) and works with Claude Code and Codex. It has a `run-experiment` skill. — [GitHub](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep); [tessl registry](https://tessl.io/registry/skills/github/wanshuiyin/Auto-claude-code-research-in-sleep/run-experiment)
- iii-experimental/n-autoresearch: 101 stars, last push 2026-04-18. It runs one worker per GPU (acquire, train, release), with remote workers able to join. Each experiment is registered with a hypothesis before training. Experiments, lineage, and the GPU pool live in a crash-surviving KV store. Heartbeats mark stale GPUs, and it aborts after 3 consecutive crashes. It explicitly "replaces the bash loop, git-as-state, and flat TSV". — [GitHub](https://github.com/iii-experimental/n-autoresearch)
- mikeapedia/autoresearch-at-home: 2 stars, last push 2026-03-17. Its Show HN describes agents sharing GPUs with a shared memory layer: read the best result, propose, run on your GPU, publish back. — [HN Show HN](https://news.ycombinator.com/item?id=47343935); gh search
- superagent-ai/polyresearch: 21 stars, last push 2026-04-23, MIT. It coordinates via GitHub Issues/PRs. Reviewers rerun eval on candidate and baseline. "The evaluation code lives outside the editable surface, so agents cannot grade their own homework." Every attempt is a `results.tsv` row plus an unmerged branch. `auto_approve: false` adds human gating, and stale claims expire. — [GitHub](https://github.com/superagent-ai/polyresearch)
- CORAL: about 1.1k stars, released 2026-03-18, arXiv 2026-04-03, accepted at COLM 2026. "Each agent runs in its own git worktree", with shared attempts/notes/skills symlinked in. "A grader daemon scores every commit." The grader venv is hidden from agents and isolation is opt-in on the host. It supports Claude Code, Codex, Cursor, OpenCode, and others. It does not describe compute provisioning. — [GitHub](https://github.com/Human-Agent-Society/CORAL)
- vast-autoresearch: no repo found by gh search (gap).

**Infra vendors (compute layer)**
- SkyPilot (10.8k stars, active) ships an official agent skill and Claude Code plugin (`claude plugin install skypilot@skypilot`). Its docs feature a "Parallel Autoresearch" recipe. — [SkyPilot docs](https://docs.skypilot.co/en/latest/getting-started/skill.html); [example](https://docs.skypilot.co/en/latest/examples/agents/autoresearch.html)
- SkyPilot blog (2026-03-18): Claude Code used `sky launch` / `sky exec` / `sky logs` on 16 GPUs (13 H100, 3 H200) across two k8s clusters. About 910 experiments were submitted and about 700 had valid results; the rest were "queued or crashed". Cost was about $9 in API and under $300 in GPU. Guardrails were a read-only `prepare.py`, a 5-minute budget, and a 16-cluster budget given only as an instruction. No dedicated tracking tool was used, and teardown was not described. — [SkyPilot blog](https://skypilot.ai/blog/scaling-autoresearch/)
- VESSL (2026-05-08): fanning K candidates out as batch jobs gave about 4× wall-clock speedup at the same $5.28 cost. "Keep the agent hardware-blind." "Don't tie your agent to a dedicated GPU." It ships helper scripts (`submit-async.sh`, `wait-jobs.sh`). — [VESSL blog](https://vessl.ai/en/blog/dont-tie-gpu-to-agent)
- RunPod has an official open-source MCP server (`@runpod/mcp-server`). It can create and manage pods, deploy serverless, submit and monitor jobs, browse GPU pricing, and show spend history. — [RunPod blog](https://www.runpod.io/blog/manage-your-runpod-infrastructure-from-any-ai-assistant-introducing-the-runpod-mcp-server); [RunPod docs](https://docs.runpod.io/get-started/mcp-servers)
- Modal markets itself as a host for GPU-backed MCP servers and as the only GPU-capable sandbox among OpenAI Agents SDK providers (vendor claim). — [Modal](https://modal.com/resources/best-sandbox-openai-agents-sdk)
- Lambda: no MCP server found.
- alphaXiv OpenResearch (blog 2026-09-09): "an open-source workspace" that "organizes the agents' code, experiments, live runs, and results in a shared experiment tree". It works with existing coding agents and ships autoresearch skills. It provides no compute and is paired with Thinking Machines' Tinker. Agents forecast cost before paid calls ($85.08 forecast vs about $93.80 observed). It has no explicit safety features. — [alphaXiv blog](https://www.alphaxiv.org/blog/let-your-agents-tinker-with-research-papers)

**Small MCP executors (the closest match to "execution layer")**
- wtrhacker/gpu-mcp: 2 stars, created 2026-08-31, pushed 2026-10-25, v0.0.0, no license. It runs "policy-constrained Python jobs on shared-filesystem GPU clusters" over SSH. `gpu-mcp.toml` allowlists hosts, script dirs, and writable/log dirs. GPU work stays locked until a human approves the exact file version, and any edit re-locks it. Stopping a process needs a second call with a matching fingerprint (owner, cmd, GPU, start time). It uses a heartbeat GPU reservation ("Silence is not mistaken for an idle GPU"), per-session job ownership, and pushes events to the launching conversation. It is "not a security sandbox or a cluster scheduler" and has no SLURM, k8s, or cloud support. — [GitHub](https://github.com/wtrhacker/gpu-mcp)
- PushPullCommitPush/ml-mcp ("ML Lab MCP"): 4 stars, pushed 2026-01-10 (stale). It has train_launch/status/stop, experiment create/compare, remote VPS, and a local backend. — [GitHub](https://github.com/PushPullCommitPush/ml-mcp)
- CyrusTao/ml-lab-mcp (glama listing): a self-hosted HTTP server on a GPU box. The listing says GPU scheduling and queuing are not implemented yet and suggests CUDA_VISIBLE_DEVICES. The GitHub repo returned 404 on 2026-10-31. — [glama](https://glama.ai/mcp/servers/CyrusTao/ml-lab-mcp)
- IlyaGusev/mle_kit_mcp: 4 stars, last push 2025-11-01. Local files/code plus a remote GPU. — gh search
- mrhy7176/cloudgpu-mcp: 0 stars, pushed 2026-10-31. "let MCP clients rent a per-minute GPU, run a job, download results and shut it down". — gh search

**Tracking-layer MCPs**
- W&B MCP server (GA, page updated 2026-05-20; repo wandb/wandb-mcp-server has 72 stars, active): 20 tools (query, `compare_runs`, `diagnose_run`, Weave traces, reports, `log_analysis_to_wandb`). No tool launches training. It is positioned explicitly for "overnight agentic research loops" where agents propose experiments, start training, and monitor them. The backend "has to be the durable layer the team's research compounds on". Its authors say the practice has "no finished playbook". — [W&B announcement](https://forge.coreweave.com/wandb/wandb_fc/product-announcements-fc/reports/Introducing-the-W-B-MCP-Server-An-agent-native-interface-for-your-experiments-and-traces--VmlldzoxNjk0MDE5OQ)
- MLflow: official MCP server (`mlflow[mcp]` ≥3.5.1) to query traces and experiments from Claude Code, Cursor, and others (Databricks docs updated 2026-09-15). MLflow Skills teach coding agents MLflow APIs, and `mlflow autolog claude` traces Claude Code sessions (3.14+). The focus is GenAI/agent observability, not launching training. — [Databricks docs](https://docs.databricks.com/aws/en/mlflow3/genai/recipes/set-up-mcp-server); [MLflow blog](https://mlflow.org/blog/self-improving-agent-loop/); [MLflow+Claude Code](https://mlflow.org/blog/mlflow-claude-code/)
- Community mlflow-mcp (PyPI) and yesid-lopez/mlflow-mcp-server (0 stars) are read and query only. — [PyPI](https://pypi.org/project/mlflow-mcp/); [GitHub](https://github.com/yesid-lopez/mlflow-mcp-server)

**Older research agents**
- WecoAI/aideml: 1.6k stars, active (2026-10-16). It is a full agent with its own tree search and does not serve as a layer for other agents. openai/mle-bench: 1.8k stars, last push 2026-04-24. SakanaAI/AI-Scientist: 14.9k stars. AI-Scientist-v2: 7.4k stars. Both Sakana repos were last pushed 2025-12-19, so they are stagnant. — gh api 2026-10-31

**Commercial**
- Autoresearch, Inc. (YC) offers "autoresearch as a service". hiloop claims 4,188 experiments in 2 days on 50 B200s (self-reported). — [YC](https://www.ycombinator.com/companies/autoresearch-inc); [HN search summary](https://news.ycombinator.com/item?id=47442435)

### Inferences
- The field has stratified into three layers: (a) loop/strategy (skills, harnesses: saturated), (b) compute (vendors, each with its own skill or MCP), and (c) tracking (MLflow and W&B, read-mostly MCPs). The thin glue between (a) and (b+c) is a backend-agnostic "run executor" that owns run identity, code snapshot, budget, ownership, and kill semantics. Only tiny repos address it.
- Most loop projects re-implement their own executor (bash loop, `sky exec`, tmux/SSH, KV store), which suggests a missing shared primitive. It may also mean the primitive is easy enough that nobody needs a library for it.

### Gaps
- vast-autoresearch: not found. Star counts for mlflow-mcp (PyPI) and ML Lab MCP (repo 404) are unavailable. There is no Modal-official "manage Modal" MCP and no Lambda MCP found.
- Download or usage numbers for the small MCP executors are unknown.

## 2. Reported pain when agents run experiments

### Takeaway
The pain is real and consistently described. Practitioners say the environment is the bottleneck, not the model: sandboxes that block GPUs and long processes, context overflow from logs, lost experiment state and goal drift, wasted GPU on bad proposals or crashes, metric gaming, and the absence of a trust model. Runaway cost is mostly a feared risk rather than a documented incident.

### Cited Findings
- Cerebras (2026-03-19): "The bottleneck isn't intelligence. It's everything around it." "Environment design matters more than model choice." "current sandboxing treats GPU access and long processes as threats." Inside the sandbox, CUDA couldn't see the GPU and the package manager broke; they spent more time on sandbox permissions than on experiments. On goal drift, an overnight agent abandoned its task, and in one case masking experts looked like progress while saving no memory. On wasted compute, Codex-Spark accepted 17% of proposals (about 2h of GPU wasted) vs 67% for GPT-5.4 (about 20 min wasted). `codex exec` has no loop mode, so they used a bash wrapper that restarts fresh each iteration. — [Cerebras](https://www.cerebras.ai/blog/how-to-stop-your-autoresearch-loop-from-cheating)
- Context overflow: autoresearch issue #298 reports that the primary agent ingests all training output and proposes a subagent per run that returns a compact summary to `results.tsv`. — [Issue #298](https://github.com/karpathy/autoresearch/issues/298)
- Losing state and drift: "When research agents run experiments for hours or days… goals drift, constraints disappear, compute gets burned, and experiment state gets lost." "This isn't really a failure of the model — it's a failure of the environment." (2026-03-15; 0 replies.) — [Discussion #275](https://github.com/karpathy/autoresearch/discussions/275)
- A memory workaround fork writes `sessions/memory.md` after every run. — [Discussion #513](https://github.com/karpathy/autoresearch/discussions/513)
- Crashes: in SkyPilot's run, about 210 of about 910 experiments had no valid result ("queued or crashed"). — [SkyPilot blog](https://skypilot.ai/blog/scaling-autoresearch/)
- Idle GPU: about 3–4 minutes of each run is non-GPU overhead, plus agent think time, so a dedicated H100 sits idle almost half the time. VESSL lists runaway cost from an indefinite loop as a risk. — [VESSL](https://vessl.ai/en/blog/dont-tie-gpu-to-agent)
- Shared clusters: gpu-mcp's design (human approval per file version, fingerprinted kill, session-owned jobs, heartbeat reservations) is framed as avoiding "mistakes" by trusted researchers. This implies the pains are agents killing others' jobs and agents grabbing GPUs that only look idle. — [gpu-mcp](https://github.com/wtrhacker/gpu-mcp)
- HN on SkyPilot's post (237 points, 94 comments, about Mar 2026): rfw300: "the agent orchestrating 16 GPUs is _half_ as efficient as the agent using 1 GPU in GPU-time"; "industrialized overfitting". herf: with 5-minute runs, "how do you know… you aren't affecting the overall asymptote?" kraddypatties: "getting GPUs is still a nightmare". QubridAI: "we've solved how to run agents anywhere, but not yet how to trust them anywhere." janalsncm: "You do not need an LLM to find a better batch size or learning rate." — [HN](https://news.ycombinator.com/item?id=47442435)
- Reproducibility requirements listed in an accelerator-research appendix: experiment counts, seeds, budgets, token cost, host cores, a protected scoring harness, pinned model snapshots and SDK version. It also used a hard per-experiment wall-clock cap to kill stuck agent cycles. — [arXiv 2608.07138](https://arxiv.org/pdf/2608.07138)
- Cost: alphaXiv says on-demand GPU cost is hard to forecast because of runtime, utilization, failures, parallel workers, and instance lifetimes. — [alphaXiv](https://www.alphaxiv.org/blog/let-your-agents-tinker-with-research-papers)

### Inferences
- The most cited pains (lost state, context bloat, crash handling, metric integrity, GPU access inside sandboxes, ownership on shared boxes) are executor and ledger problems, not loop-strategy problems.
- "Agent killing others' jobs" and "runaway cost" have little documented incident evidence. They are anticipated risks, so marketing on them is speculative.

### Gaps
- No Reddit threads were retrieved (search budget). No systematic incident data on runaway cloud bills from agents was found.

## 3. Converged primitives and what's missing

### Takeaway
The convergence is clear: fixed time budget, single scalar metric, a protected or read-only evaluator, keep-or-revert ratchet on git, an append-only results ledger (TSV, KV, or Issues), per-experiment isolation (git branch or worktree), heartbeats, and crash caps. What's missing is a standard, backend-agnostic, agent-facing run object that binds code snapshot, config, environment, compute, cost, and metric. Also missing are enforced (not prompted) budgets, ownership-safe control on shared GPUs, and asynchronous notification for long jobs.

### Cited Findings
- Fixed 5-minute budget, single metric (val_bpb), read-only prepare.py, results.tsv ledger, keep/discard. — [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- Git worktree per agent plus a grader daemon scoring every commit plus a hidden grader. — [CORAL](https://github.com/Human-Agent-Society/CORAL)
- Evaluation outside the editable surface, peer re-run verification, a branch per attempt, append-only coordination. — [polyresearch](https://github.com/superagent-ai/polyresearch)
- Hypothesis registered before a run, GPU pool acquire/release, heartbeats, abort after 3 crashes, a structured store instead of TSV. — [n-autoresearch](https://github.com/iii-experimental/n-autoresearch)
- Approval bound to file hash, fingerprinted kill, heartbeat reservation, events pushed to the launching conversation. — [gpu-mcp](https://github.com/wtrhacker/gpu-mcp)
- A subagent per run to protect context. — [Issue #298](https://github.com/karpathy/autoresearch/issues/298)
- SkyPilot's budget was only an instruction ("asked to stay within a 16-cluster budget"), and teardown was not described. — [SkyPilot blog](https://skypilot.ai/blog/scaling-autoresearch/)
- Cerebras calls GPU-capable sandboxes for long processes an open problem. — [Cerebras](https://www.cerebras.ai/blog/how-to-stop-your-autoresearch-loop-from-cheating)
- W&B MCP has no launch tool. MLflow MCP is query and trace oriented. — [W&B](https://forge.coreweave.com/wandb/wandb_fc/product-announcements-fc/reports/Introducing-the-W-B-MCP-Server-An-agent-native-interface-for-your-experiments-and-traces--VmlldzoxNjk0MDE5OQ); [Databricks](https://docs.databricks.com/aws/en/mlflow3/genai/recipes/set-up-mcp-server)

### Inferences (what is missing)
- **Run provenance as a first-class object.** Each tool binds commit to metric differently (TSV row, branch, KV, sky logs). None emits a standard record (git SHA + dirty diff + env lock + data hash + hardware + seed + cost + metric) that MLflow or W&B could ingest.
- **Enforced guardrails.** Hard spend and GPU-hour caps, max concurrent jobs, auto-teardown, and per-run wall-clock caps should live in the executor rather than the prompt.
- **Multi-tenant safety.** Ownership-scoped kill and stop, and reservations on shared on-prem boxes or SLURM. Only gpu-mcp (SSH only) attempts this.
- **Async long-job handling.** Submit, detach, get notified, and receive a compact summary, so logs never enter context.
- **Backend portability.** The same run spec across local GPU, SSH box, SLURM, k8s, and cloud. SkyPilot already covers cloud and k8s, which narrows this gap considerably.
- **Comparability.** Pinned eval harness, seeds, and noise estimates (re-run baselines as polyresearch does) to guard against "industrialized overfitting".

### Gaps
- No evidence was found of a standard or spec effort (for example a "run record" schema) across these projects.

## 4. Are incumbents making a small OSS project redundant?

### Takeaway
Partly. Each incumbent covers one layer and is moving agent-ward fast. SkyPilot owns cloud/k8s launch with an official Claude Code plugin and an autoresearch recipe. RunPod and VESSL ship agent tooling for their own clouds. W&B and MLflow ship MCPs and skills for querying and logging. Anthropic and OpenAI ship generic sandboxes (no GPU emphasis). Nobody ships the vendor-neutral safety/ledger glue, but SkyPilot is best positioned to absorb it.

### Cited Findings
- SkyPilot: official skill and plugin, autoresearch example, blog with 16 GPUs (2026-03-18). — [docs](https://docs.skypilot.co/en/latest/getting-started/skill.html); [blog](https://skypilot.ai/blog/scaling-autoresearch/)
- W&B MCP GA (2026-05-20), positioned for agentic research loops. — [W&B](https://forge.coreweave.com/wandb/wandb_fc/product-announcements-fc/reports/Introducing-the-W-B-MCP-Server-An-agent-native-interface-for-your-experiments-and-traces--VmlldzoxNjk0MDE5OQ)
- MLflow MCP, Skills, and Claude Code autolog (3.14+). — [MLflow blog](https://mlflow.org/blog/mlflow-claude-code/); [Databricks](https://docs.databricks.com/aws/en/mlflow3/genai/recipes/set-up-mcp-server)
- RunPod official MCP covering jobs, pods, and spend history. — [RunPod](https://www.runpod.io/blog/manage-your-runpod-infrastructure-from-any-ai-assistant-introducing-the-runpod-mcp-server)
- Thinking Machines' Tinker lets agents launch experiments "without managing GPUs directly", via alphaXiv OpenResearch. This abstracts GPUs away entirely for fine-tuning use cases. — [alphaXiv](https://www.alphaxiv.org/blog/let-your-agents-tinker-with-research-papers)
- Anthropic Claude Managed Agents went into public beta on 2026-04-08 (sandboxing, checkpointing, credential scoping, tracing). Self-hosted sandboxes and MCP tunnels followed on 2026-05-19, and an open-source sandbox-runtime on 2026-10-24. OpenAI's Agents SDK added native sandboxes in about Apr 2026 (7–9 providers, with Modal the GPU option). No GPU-experiment product was found from either company. — secondary sources: [Enterprise DNA](https://enterprisedna.co/resources/news/anthropic-mcp-tunnels-self-hosted-sandboxes-enterprise-2026/), [emergent.sh](https://emergent.sh/news/anthropic-launches-sandbox-runtime-ai-agent), [Medium](https://medium.com/@rajasekar-venkatesan/anthropic-and-openai-just-shipped-the-same-answer-to-ai-agents-seven-days-apart-c19f2dc03244), [Modal](https://modal.com/resources/best-sandbox-openai-agents-sdk). These are not verified against primary docs.
- A search snippet claimed that in Jul 2026 Hugging Face reported agent attacks and OpenAI reported agents escaping sandboxes. This is unverified, comes from an aggregator summary, and should not be used without a primary source.

### Inferences
- Redundancy risk ranks highest for "agent can launch cloud GPU jobs" (SkyPilot, RunPod, Modal already do it) and for "agent can query and compare runs" (W&B, MLflow). It is lowest for on-prem/shared-cluster safety (policy, ownership, approvals) and for a neutral provenance ledger joining executor to tracker.
- SkyPilot (open source, multi-backend, already shipping an agent skill) is the most likely absorber. Adding managed-job budgets and an experiment ledger is a natural extension for it. Incumbent moves would most plausibly come from SkyPilot or from W&B/MLflow adding "launch" tools.

### Gaps
- No public roadmap was found from SkyPilot, W&B, or MLflow about agent-enforced budgets or launch tools.

## 5. Durable trend or hype spike?

### Takeaway
Attention spiked hard in March 2026 and the loop pattern has persisted and broadened. Many 1k–18k-star repos were still active in Oct 2026, there is a COLM paper, YC companies, and vendor GA products. Usage of the infra/execution slice specifically is thin and mostly demos on toy benchmarks (nanochat val_bpb). Credible skeptics question the value against classical HPO.

### Cited Findings
- The reference repo has about 98.9k stars but no commits since 2026-03-26. Derivatives are active through Oct 2026: ARIS 18.3k (pushed 2026-10-30), AutoResearchClaw 14.8k, pi-autoresearch 8.2k, FAROS 3.0k (pushed 2026-10-21), CORAL 1.1k (COLM 2026). — gh api / gh search 2026-10-31; [CORAL](https://github.com/Human-Agent-Society/CORAL)
- The pattern has spread beyond ML: GPU kernels (autokernel 1.6k), trading (atlas-gic 2.3k), genealogy, and skill optimization (darwin-skill 6.7k). — gh search 2026-10-31
- Commercial: Autoresearch, Inc. (YC). hiloop claims 4,188 experiments on 50 B200s. Weco claims autoresearch beats Optuna, which commenters disputed. — [YC](https://www.ycombinator.com/companies/autoresearch-inc); [HN Optuna thread](https://news.ycombinator.com/item?id=47620196)
- Vendors invested in content and products: SkyPilot (Mar), VESSL (May), W&B GA (May), alphaXiv/Tinker (Sep), Cerebras (Mar). — sources above
- Skepticism: "industrialized overfitting", poor GPU-time efficiency at scale, "You do not need an LLM to find a better batch size", and "I'm trying to find the 'wow' factor". One commenter reported LLM coding is less productive outside LLM topics, such as robotics. — [HN](https://news.ycombinator.com/item?id=47442435)
- The executor-slice projects have little traction: gpu-mcp 2 stars, ml-mcp 4 (stale), mle_kit_mcp 4 (stale), cloudgpu-mcp 0, polyresearch 21, n-autoresearch 101 (stale since Apr), autoresearch-at-home 2. — gh 2026-10-31

### Inferences
- Demand for "agents iterating on a measurable target" looks durable. Demand for a standalone open-source execution layer is unproven: every small attempt has stalled under about 100 stars, while users take vendor skills or roll their own bash/sky loops. This is the strongest evidence against the niche. The counter-reading is that no project has yet been good enough or backend-neutral enough.
- The most defensible wedge is on-prem/shared-GPU labs and universities (SLURM, SSH boxes), where cloud vendors don't reach and gpu-mcp's design shows real pain. The other candidate is a neutral provenance/ledger standard that plugs into SkyPilot, MLflow, and W&B rather than competing with them.

### Gaps
- No named companies were found using any execution-layer tool in production. Star-history time series were not retrieved, so there is no growth curve. Reddit sentiment was not sampled.
