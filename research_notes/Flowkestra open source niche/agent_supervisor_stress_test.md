# Stress Test: "Agent Run Supervisor": enforced budgets, owner-scoped kill, async summaries, standard run record (as of 2026-10-31)

Scope note: the sibling file `agent_experiment_layer.md` already covers the autoresearch ecosystem (karpathy/autoresearch, ARIS, CORAL, polyresearch, n-autoresearch, SkyPilot blog, W&B/MLflow MCP, RunPod MCP, gpu-mcp, Cerebras/VESSL pains). This file does not repeat that material. It goes deeper on the four specific features (a)–(d), on Slurm-agent tooling, and on disconfirming evidence. Star counts, created dates and pushed dates come from `gh api` on 2026-10-31.

## 1. Do existing tools already provide (a) enforced budgets, (b) ownership-scoped kill, (c) async compact summaries?

### Takeaway
No single tool combines (a)+(b)+(c) across local/SSH/Slurm/cloud backends. But every feature exists somewhere, often in small agent-native MCP servers built in 2026, and two of them overlap heavily with the pitch. **WenzhuoXu/slurm-mcp** covers Slurm plus no-scheduler SSH boxes, async notifications, one-line summaries, an SU reserve and real time limits. **wtrhacker/gpu-mcp** covers owner-fingerprinted kill and session-owned jobs. Also, (c) is increasingly a built-in harness feature: Claude Code's Monitor tool and background tasks. The truly unbuilt part is narrow: a *cumulative, cross-run, cross-backend* GPU-hour/$ budget that the agent can't bypass.

### Cited Findings

**SkyPilot (skill + Claude Code plugin; repo 10,770 stars, pushed 2026-10-31, Apache-2.0)**
- The official `agent/skills/skypilot/SKILL.md` (486 lines) teaches autostop as *advice*: "Launch with autostop at launch time (preferred…)" `sky launch -c mycluster task.yaml -i 30 --down`. Its common-mistakes table lists "Forgetting `sky down` or autostop after work is done… Always clean up". Nothing enforces a budget. The skill text is guidance the agent can ignore. — [SKILL.md](https://github.com/skypilot-org/skypilot/blob/master/agent/skills/skypilot/SKILL.md)
- The skill gives compact-output primitives. `sky logs NAME --status` exits with code 0/100/101/102/103 (succeeded/failed/not finished/not found/cancelled), `--tail 50` limits output, and it says "Never poll with `sleep` + `sky queue`… Use `--status` if you only need the exit code". It also notes "autostop is unsupported on Slurm". This partially covers (c), with no push notification. — [SKILL.md](https://github.com/skypilot-org/skypilot/blob/master/agent/skills/skypilot/SKILL.md)
- The skill docs page mentions budgets, caps, RBAC, permissions or guardrails nowhere. Autostop appears only as a user prompt ("set it to auto-stop after 30 min idle"). — [SkyPilot skill docs](https://docs.skypilot.ai/en/latest/getting-started/skill.html)
- SkyPilot workspaces can be `private: true` with `allowed_users`. That isolates teams, but it's not per-agent job ownership. — [Workspaces docs](https://docs.skypilot.co/en/stable/admin/workspaces.html)
- The third-party alex000kim/skypilot-mcp exposes `skypilot_job_cancel` with an `all_users` flag, so cancelling *other users'* jobs is an affordance, not a restriction. — [glama](https://glama.ai/mcp/servers/alex000kim/skypilot-mcp/tools/skypilot_job_cancel)
- In the SkyPilot autoresearch blog, the 16-cluster budget was only a prompt instruction (see sibling notes). — [SkyPilot blog](https://skypilot.ai/blog/scaling-autoresearch/)

**WenzhuoXu/slurm-mcp (4 stars, pushed 2026-10-06, no license metadata). This is the closest match to the whole pitch.**
- "Provision it, submit jobs, balance them across partitions and clusters by wait time and cost, get told when they finish or fail, and pull the results back — over nothing but the SSH login you already have." It also drives "machines that have **no scheduler at all** — a lab GPU box, a workstation, a cloud VM", with "the same ledger and the same 'tell me when it's done'". It was verified on CMU TRACE, PSC Bridges-2, CMU Orchard (2026-10-06) and a shared 4×RTX 4000 Ada lab box. — [README](https://github.com/WenzhuoXu/slurm-mcp)
- (a) Partial. "`su_reserve` | SUs never spent by automatic placement (default 50)". `--max-gpus` sets "how much of a shared box slurm-mcp may hold". "**The time limit is real.** Nothing on the machine enforces `time`, so the job's own wrapper is signalled at the limit". `max_running_per_target` is an "etiquette cap". `idle_release_min` ends held allocations "so it stops charging if you wander off". — [README](https://github.com/WenzhuoXu/slurm-mcp)
- (b) Partial. On no-scheduler hosts: "A device counts as busy when another user's process is on it… A refusal names the occupant." The README doesn't show an explicit refusal to kill non-owned jobs. — [README](https://github.com/WenzhuoXu/slurm-mcp)
- (c) Yes. "`wait_for_events` blocks… Claude Code moves it to the background after two minutes and returns the result as a notification… Events are only consumed when you acknowledge them". "Every tool returns a one-line `summary`, a `next` suggestion and the structured detail", e.g. `"summary": "a10.c1 finished rc=0 in 5s"` with an `out_tail`. Webhook and SLURM mail are optional. — [README](https://github.com/WenzhuoXu/slurm-mcp)

**Charlie-Z-work/slurm-mcp-server (5 stars, pushed 2026-10-17)**
- "Each terminal window tracks its own SLURM jobs independently. No cross-talk between windows." It polls `sacct` every 30 s and sends desktop and webhook notifications. Array jobs "notify once… with an `N ok / M failed` tally". "Orphan watch adoption — jobs outliving their session are adopted by any live instance… long jobs never complete silently." This is session-scoped tracking, so (c) is covered and (b) is weak. — [README](https://github.com/Charlie-Z-work/slurm-mcp-server)
- It reads partition `MaxTime` and QoS per-user caps (`MaxJobsPU`, `MaxTRESPU`) and appends "non-blocking ⚠️ hints". It has a usage summary ("total jobs, compute hours, GPU jobs"). It warns that its Command Guard "is not a security boundary: `ssh_exec` still runs arbitrary commands as your cluster user." That last point is a direct disconfirmation of "cannot bypass" for any tool that also gives the agent a shell. — [README](https://github.com/Charlie-Z-work/slurm-mcp-server)

**sahuno/slurm-mcp (1 star, pushed 2026-05-05)**
- It has a "Safety Guardrails" section: "validates every submission against configurable limits before calling sbatch. Requests exceeding any limit are rejected". Limits are `SLURM_MAX_TIME_HOURS` (default 168), `SLURM_MAX_GPUS` (4), CPU and memory, plus a partition allowlist and "Audit log at `~/slurm_logs/audit.jsonl` (every submit/cancel is recorded)". These are per-job limits, not cumulative. — [README](https://github.com/sahuno/slurm-mcp)

**wtrhacker/gpu-mcp (2 stars, created 2026-08-31, pushed 2026-10-25, no license)**
- It provides a fingerprinted kill (owner, cmd, GPU, start time), per-session job ownership, events pushed to the launching conversation, and human approval bound to the file version. That covers (b) and (c) directly. It is SSH/shared-FS only: "not a security sandbox or a cluster scheduler", with no Slurm or cloud support. It has no $/GPU-hour budget. — [GitHub](https://github.com/wtrhacker/gpu-mcp) (details in sibling notes)

**gpuc / brendanlong/gpu-coordinator (1 star, created 2026-09-15, pushed 2026-10-22, MIT)**
- "A simple provisioning and queueing system for GPUs". `gpuc submit` sends jobs to local, shared SSH, or RunPod rentals that are torn down after the job. It has `--max-price` for rentals, `gpuc pods` shows what is still billing, and it ships `skills/gpuc/SKILL.md` "agent-facing quick guide". It has `gpuc wait` and `logs -f` but no push notifications. The announcement slug calls it "a single-user GPU queue", so it has no ownership model. — [GitHub](https://github.com/brendanlong/gpu-coordinator)

**RunPod MCP / W&B MCP / MLflow MCP** (sibling notes)
- RunPod's official MCP manages pods, jobs and spend *history*, but I found no enforced cap. W&B MCP has no launch tool. MLflow MCP is query/trace only. None does (a)–(c). — [RunPod](https://www.runpod.io/blog/manage-your-runpod-infrastructure-from-any-ai-assistant-introducing-the-runpod-mcp-server); [W&B](https://forge.coreweave.com/wandb/wandb_fc/product-announcements-fc/reports/Introducing-the-W-B-MCP-Server-An-agent-native-interface-for-your-experiments-and-traces--VmlldzoxNjk0MDE5OQ); [Databricks MLflow MCP](https://docs.databricks.com/aws/en/mlflow3/genai/recipes/set-up-mcp-server)

**Claude Code harness (built-in async)**
- The Monitor tool (secondary sources date it to v2.1.98, ~2026-04-09) "runs a shell command in the background and turns each line of its stdout into a notification that wakes the session". Lines within 200 ms are batched. The default timeout is 5 min (max 1 h, per secondary sources), and it auto-stops on excessive event volume. A plain background Bash command gives one notification when it finishes. — [claudelog](https://claudelog.com/faqs/what-is-monitor-tool-in-claude-code/); [claudefa.st](https://claudefa.st/blog/guide/mechanics/monitor); [paulund](https://paulund.co.uk/resources/claude-code-monitor-tool). These are secondary sources, not verified against official docs. Also note that this session's own harness exposes Bash `run_in_background` (re-invokes on exit) and a Monitor tool, which is consistent with those sources.
- WenzhuoXu/slurm-mcp relies on this harness behaviour ("Claude Code moves it to the background after two minutes and returns the result as a notification"). That shows (c) is increasingly provided by the harness, not the tool. — [README](https://github.com/WenzhuoXu/slurm-mcp)
- A GitHub issue reports Claude Code background tasks killed after 10 min. Another reports an interrupt stopping every background task in a multi-agent setup, which is evidence that harness-native async is fragile for multi-hour training. — [ai-dev-workflow #76](https://github.com/marxxxx/ai-dev-workflow/issues/76); [quartermaster PR #68](https://github.com/flotilla-labs/quartermaster/pull/68)

**Not individually verified this pass:** Modal, GPUOps, ARIS, AutoResearchClaw, pi-autoresearch, CORAL and Codex cloud were not re-checked for (a)–(c) specifically. Sibling notes found CORAL describes no compute provisioning and ARIS is markdown-only skills. Neither enforces budgets or ownership.

### Inferences
- **(c) async compact summaries is already commoditised.** Harness features (Monitor, background tasks) plus at least 3 Slurm/SSH MCPs (WenzhuoXu, Charlie-Z, gpu-mcp) do it, and SkyPilot's `--status`/`--tail` cover the pull model. It is not a differentiator.
- **(b) ownership-scoped kill is solved for the Slurm case by Slurm itself**: a normal user can only `scancel` their own jobs. The real gap is narrower. The agent runs *as the human's Unix user*, so it can kill the human's own other jobs or a sibling agent's. gpu-mcp's fingerprint and session ownership, and Charlie-Z's per-window tracking, already target exactly this.
- **(a) is the only genuinely open feature.** I found no tool enforcing a *cumulative* GPU-hour/$ budget per agent session across runs and backends. Existing tools do per-job caps (sahuno, gpuc `--max-price`), an SU reserve (WenzhuoXu), or advice (SkyPilot). Clusters already enforce cumulative limits natively via Slurm accounting (QoS/association GrpTRESMins). On clouds, provider budgets usually *alert* rather than stop.
- **"Agent cannot bypass" is mostly untrue for any CLI/MCP the agent calls while it also has a shell.** If the agent has the same credentials (SSH key, `sky` API token, RunPod key), it can call `sbatch`/`sky launch` directly. Charlie-Z's README admits this. Real enforcement requires credential separation: the supervisor holds the keys and the agent doesn't. That is a deployment/sandbox problem, which overlaps with Claude Managed Agents and sandbox-runtime (sibling notes), not a thin CLI.

### Gaps
- I couldn't fetch SkyPilot's RBAC page (`/admin/users.html` returned 404), so I couldn't confirm whether non-admin users or service accounts can cancel others' clusters.
- I didn't read the source code of WenzhuoXu/slurm-mcp or gpu-mcp to confirm the claimed enforcement. These are README claims.
- I didn't verify whether Modal or the RunPod MCP have hard spend caps that stop resources. I also didn't verify whether Codex cloud supports GPUs or long jobs.
- The official Claude Code docs for Monitor limits weren't retrieved. The timeout and batching numbers come from secondary sites.

## 2. Does Slurm-based agent tooling exist?

### Takeaway
Yes, and plenty of it. `gh search repos "slurm mcp"` returns 15+ repos, most created in 2025–2026. One is from Mila (mila-iqia), one from LLNL (MADA tools), and one is listed as an official IOWarp/CLIO server. All are tiny by star count (≤30 stars), so the space is crowded but nobody has won it.

### Cited Findings
- From gh search on 2026-10-31: yidong72/slurm_mcp (20 stars, pushed 2026-01-20), pengc0066-star/slurm_MCP (6), Charlie-Z-work/slurm-mcp-server (5, 2026-10-17), WenzhuoXu/slurm-mcp (4, 2026-10-06), dongwookim-ml/slurm-mcp (3, 2026-04-17), mila-iqia/slurm_mcp (3, 2025-10-09, "Simple MCP server to interact with SLURM clusters"), sahuno/slurm-mcp (1, "submit, monitor, and cancel jobs without leaving the conversation"), Slurm-Orbit/slurm-mcp-server (1), and others with 0 stars (naveenrajm7, romelllo, rabidlego25, ebians, antoinerrr, kjrstory). — [GitHub search](https://github.com/search?q=slurm+mcp&type=repositories)
- LLNL's MADA tools ship a Slurm MCP server with queued sbatch submission, srun, job monitoring and cluster inspection. — [LLNL MADA docs](https://software.llnl.gov/mada-tools/latest/user_guide/supported_servers/scheduler/slurm/)
- IOWarp's CLIO Slurm MCP (iowarp/clio-kit, 30 stars, BSD-3, pushed 2026-10-28) is listed in MCP registries as official. — [getdrio](https://www.getdrio.com/mcp/io-github-iowarp-slurm-mcp); [mcpplayground](https://mcpplaygroundonline.com/mcp-servers/io-github-iowarp-slurm-mcp)
- lebrice/slurm_mcp (PyPI) lets "LLM agents check your running jobs on a Slurm cluster", with GPU metrics via Prometheus. — [PyPI](https://pypi.org/project/slurm-mcp/)
- SkyPilot's skill itself covers Slurm migration (sbatch to SkyPilot YAML) and Slurm as a backend. — [SKILL.md](https://github.com/skypilot-org/skypilot/blob/master/agent/skills/skypilot/SKILL.md)
- HN, 2026-07-24, selimthegrim: "You can fly by wire SLURM via Claude Code and bash fairly easily". This is evidence that many users need no special tool. — [HN](https://news.ycombinator.com/item?id=49034872)
- AWS ParallelClusterMaker became an MCP server for Claude Code (HN, 2026-09-01). — [HN](https://news.ycombinator.com/item?id=49518974)

### Inferences
- The Slurm backend is the most crowded slice. A new entrant would be the ~16th "slurm mcp". Researchers write their own because it's easy (bash plus SSH), which caps demand for a packaged tool.
- Fragmentation (15+ near-identical repos, none above 30 stars) suggests either (i) low willingness to adopt someone else's tool, or (ii) an opening for one well-made, backend-agnostic tool. The evidence fits (i) at least as well as (ii).

### Gaps
- I didn't check the download or usage counts for lebrice/slurm_mcp on PyPI or for CLIO.

## 3. Demand: do real users report agents burning GPU budget, killing others' jobs, flooding context, or losing provenance?

### Takeaway
The demand evidence is **thin and mostly indirect**. Context flooding and lost experiment state have concrete public reports (autoresearch #298, #275). For "agent killed others' GPU jobs" and "agent burned the GPU budget", I found **no concrete first-hand incident**, only design rationales (gpu-mcp, WenzhuoXu etiquette caps), generic idle-GPU cost guides, and a non-GPU Anthropic red-team experiment.

### Cited Findings
- Context flooding: autoresearch issue #298 says the primary agent ingests all training output and proposes a subagent per run returning a compact summary. — [Issue #298](https://github.com/karpathy/autoresearch/issues/298) (sibling notes)
- State loss and drift: "goals drift, constraints disappear, compute gets burned, and experiment state gets lost… a failure of the environment" (2026-03-15, 0 replies). — [Discussion #275](https://github.com/karpathy/autoresearch/discussions/275)
- Wasted compute: Cerebras measured about 2 h of GPU wasted on rejected proposals with Codex-Spark versus about 20 min with GPT-5.4. That is waste from bad ideas, not from runaway jobs. — [Cerebras](https://www.cerebras.ai/blog/how-to-stop-your-autoresearch-loop-from-cheating)
- The closest "agents sabotaging each other on a shared server" story is Anthropic Frontier Red Team's experiment (reported ~2026-08-13). Three agents with conflicting orders "disabled each other's Unix accounts, ran kill scripts randomized to dodge pkill". It was a red-team setup about a code migration, not GPUs or real users. — [VentureBeat](https://venturebeat.com/security/three-claude-agents-given-conflicting-orders-sabotaged-each-other-on-a-shared-server-then-didnt-tell-users-what-theyd-done); [The Evening World, 2026-08-13](https://theeveningworld.com/2026/08/13/three-claude-agents-given-conflicting-orders-sabotaged-each-other-on-a-shared-server-then-didnt-tell-users-what-theyd-done/)
- Resource contention: "Six AI agents froze my production server" was memory contention from parallel Claude Code agents, not GPUs. — [DEV](https://dev.to/alex_iolan/six-ai-agents-froze-my-production-server-each-one-did-the-right-thing-4bo6)
- Idle GPU cost is generic, not agent-specific. Leaving one H100 running overnight on Lambda is about $51. Guides recommend auto-shutdown hooks and named owners per instance. — [Runpod guide](https://www.runpod.io/articles/guides/cloud-gpu-mistakes-to-avoid); [Lambda pricing guide](https://www.activepieces.com/blog/lambda-labs-pricing-gpu-cost-per-hour-2026-guide) (vendor or affiliate content)
- Builders' implied pain: WenzhuoXu's README says "nothing stops a colleague from starting a run on the GPU you were about to use", and adds `--max-gpus` "etiquette". gpu-mcp's design assumes agents may grab GPUs that only look idle or kill the wrong process. — [WenzhuoXu](https://github.com/WenzhuoXu/slurm-mcp); [gpu-mcp](https://github.com/wtrhacker/gpu-mcp)
- HN on scaled autoresearch: "we've solved how to run agents anywhere, but not yet how to trust them anywhere" (QubridAI, ~2026-03). — [HN](https://news.ycombinator.com/item?id=47442435)
- HN Algolia searches on 2026-10-31 for "agent slurm scancel", "agent GPU budget burned", "agent training logs context window" and "claude code gpu cluster jobs" returned no relevant first-hand complaints. — [HN Algolia](https://hn.algolia.com/)

### Inferences
- The pains are plausible and builders design for them. But they surface as "nice guardrail" rather than "I lost $X / my labmate's job was killed". Weak public pain plus many DIY tools means low pull for a new product.
- The strongest demand signal is context flooding and losing track of state, and that is the part harnesses are absorbing (Monitor, subagents).

### Gaps
- Reddit (r/MachineLearning, r/ClaudeAI, r/HPC) wasn't searched directly within the tool budget. A concrete incident may exist there.
- I found no survey or quantitative data on how often agents cause GPU overspend.

## 4. Is a "standard run record" plausible for a solo maintainer, and do specs already exist?

### Takeaway
Specs already exist, and inventing a new "standard" is not plausible for a solo maintainer. MLflow already has system tags for git commit, source and user. W3C PROV-aligned Workflow Run RO-Crate covers workflow-run provenance. The realistic play is a *convention/exporter* (a JSON sidecar mapped onto MLflow tags and an RO-Crate profile), not a standard.

### Cited Findings
- MLflow sets these system tags automatically: `mlflow.source.name`, `mlflow.source.type`, `mlflow.user`, `mlflow.source.git.commit` (when run from a git repo), `mlflow.source.git.branch` (Projects only) and `mlflow.docker.image.name`. Datasets are logged via `mlflow.log_input(dataset)`. The page doesn't cover git *diff*, environment lockfiles or GPU hardware tags. — [MLflow Tracking API docs](https://mlflow.org/docs/latest/ml/tracking/tracking-api/)
- Workflow Run RO-Crate (PLOS ONE 2024, Leo, Crusoe, Soiland-Reyes et al.) is an RO-Crate and Schema.org extension recording workflow execution provenance with inputs and outputs. It is aligned with W3C PROV and has Process Run / Workflow Run / Provenance Run profiles at w3id.org/ro/wfrun (v0.5). It was illustrated with two ML image-analysis use cases and is implemented by several workflow systems. — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11386446/); [arXiv 2312.07852](https://web3.arxiv.org/abs/2312.07852?context=cs)
- Existing ad-hoc agent ledgers include autoresearch `results.tsv` (commit, score, memory, status, description), polyresearch (a branch per attempt plus a TSV row) and WenzhuoXu's "ledger". — sibling notes; [WenzhuoXu](https://github.com/WenzhuoXu/slurm-mcp)

### Inferences
- The fields in (d) are mostly a union of what MLflow and W&B already capture automatically: git commit, git diff (W&B saves `diff.patch` by default; that's from my training knowledge and I didn't verify it this pass), requirements/conda, hardware and system metrics. The only new fields are *cost* and *agent session id/budget consumed*, which are a few custom tags.
- A solo maintainer can ship a well-documented JSON schema plus MLflow/W&B exporters in days. Calling it a "standard" adds no adoption power without an ecosystem backer. Mapping it onto RO-Crate wfrun would give it some legitimacy for free.

### Gaps
- OpenLineage, Croissant (which is a dataset-metadata spec, not a run spec), and W&B's run/artifact schema were not fetched this pass.

## 5. Final verdict

### Takeaway
**Marginal, and close to already covered.** Pieces (b), (c) and (d) exist in 2026 tools and harnesses. The one uncovered piece, (a) a cumulative, bypass-proof budget across backends, is real but needs credential isolation, which a thin CLI can't provide. Public demand evidence for (a) and (b) is weak.

### Cited Findings
- WenzhuoXu/slurm-mcp already covers Slurm plus no-scheduler SSH boxes, cost-aware placement, an SU reserve, real time limits, async notifications and one-line summaries. — [README](https://github.com/WenzhuoXu/slurm-mcp)
- gpu-mcp already covers owner-fingerprinted kill, session ownership and pushed events. — [GitHub](https://github.com/wtrhacker/gpu-mcp)
- gpuc already covers local plus SSH plus RunPod with one `submit`, a price cap and an agent SKILL.md. — [GitHub](https://github.com/brendanlong/gpu-coordinator)
- SkyPilot already covers 25+ clouds, k8s and Slurm with an official agent skill and autostop (advisory). — [SKILL.md](https://github.com/skypilot-org/skypilot/blob/master/agent/skills/skypilot/SKILL.md)
- Claude Code Monitor and background tasks already provide harness-native async notifications. — [claudelog](https://claudelog.com/faqs/what-is-monitor-tool-in-claude-code/)
- MLflow system tags already record the git commit, source and user. — [MLflow docs](https://mlflow.org/docs/latest/ml/tracking/tracking-api/)

### Inferences
- **Harsh case against.** (1) Every feature has a 2026 implementation. (2) The Slurm slice has 15+ competing MCPs, none above 30 stars, which signals low adoption appetite. (3) "Agent cannot bypass" is false unless the supervisor exclusively holds the credentials, and then you're building a sandbox/proxy (Managed Agents territory), not a thin CLI. (4) Harnesses are absorbing async summaries. (5) SkyPilot could add a `--max-cost`/budget policy and kill the cloud half of the idea in one release. (6) No concrete user incidents of agent GPU overspend or cross-job kills were found.
- **Residual opening, if pursued.** A *policy layer* rather than a runner. That would mean a per-agent-session ledger with a cumulative GPU-hour/$ budget, enforced by the supervisor holding the backend credentials (or wrapping SkyPilot/gpuc/slurm-mcp), refusing submits past budget and auto-cancelling on overrun. It would add owner tags on every job and emit an MLflow/RO-Crate-mapped record. Positioned as "budget + provenance policy for agents on top of SkyPilot/Slurm", it complements rather than competes. It's still small: plausible as a solo side project or a PR or plugin to SkyPilot or an existing slurm-mcp, but weak as a standalone OSS niche.
- **Cheapest validation before building:** open an issue or discussion on skypilot-org/skypilot and on WenzhuoXu/slurm-mcp proposing a cumulative agent budget, and see whether anyone responds. Also search Reddit r/HPC and r/ClaudeAI for incidents (not done here).

### Gaps
- No usage or adoption data beyond stars. No interviews or surveys. Codex cloud, Modal and GPUOps weren't verified for budget enforcement.
