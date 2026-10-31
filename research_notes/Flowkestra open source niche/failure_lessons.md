# Why small OSS ML experiment-orchestration tools fail or succeed (devil's-advocate notes, as of 2026-10-31)

Method note: repo stats (stars, archived flag, last push, last release, top contributors) were pulled live from the GitHub API via `gh api` on 2026-10-31. Treat star counts as of that date. The GitHub API links below (github.com/<repo>) are the source for those numbers.

## Q1. Post-mortems / status of Guild AI, Ploomber, Sacred, Polyaxon, Determined, Aim, Sematic, Kubeflow, MLflow Recipes, Orchest, Mage, Flyte vs Prefect

### Takeaway
The pattern is stark: **VC-backed "platform" tools die when the company can't monetize or gets acquired** (Orchest, Ploomber, Sematic, Determined), and **single-maintainer tools with broad scope stall when the maintainer stops** (Guild AI). Only tools with a large company or big community behind them (MLflow, Prefect, Flyte, Aim, Mage, ClearML) are still active, and even MLflow cut a whole sub-product (Recipes) that tried to be an opinionated pipeline layer.

### Cited Findings
**Dead / archived / frozen**
- **Orchest** — GitHub repo archived; last push 2023-06-06; 4,134 stars. README notice: *"we're no longer actively developing Orchest. We could not find a way to make building a workflow orchestrator commercially viable. Check out Apache Airflow for a robust workflow solution."* — [github.com/orchest/orchest](https://github.com/orchest/orchest)
- **Ploomber** — repo archived (last push 2025-05-29; 3,621 stars). README now leads with a "Deploy AI apps for free on Ploomber Cloud!" banner — [github.com/ploomber/ploomber](https://github.com/ploomber/ploomber). YC profile now describes the company as "a cloud platform to help companies deploy and scale AI applications" — [YC](https://www.ycombinator.com/companies/ploomber). No official post-mortem found; the switch from pipeline tool to app hosting is an inference from these signals, not a stated pivot.
- **Sematic** (ML pipeline orchestrator) — reportedly acquired by Weights & Biases, Feb 2024 — [PitchBook](https://pitchbook.com/profiles/company/501496-93); [YesPress timeline](https://yespress.io/weights-biases.md) (both secondary). Final commits on 2025-01-09: "Remove EE License from Sematic, restore Apache 2.0" followed by "Release v0.41.0"; no activity since; 1,003 stars — [github.com/sematic-ai/sematic](https://github.com/sematic-ai/sematic). W&B itself was acquired by CoreWeave (announced March 2025, ~US$1.7B) — [YesPress](https://yespress.io/weights-biases.md) (secondary).
- **Determined AI** — acquired by HPE in June 2021, with a public promise to keep investing in it as open source — [HPE press release](https://www.hpe.com/us/en/newsroom/press-release/2021/06/hewlett-packard-enterprise-acquires-determined-ai-to-accelerate-artificial-intelligence-innovation-with-fast-and-simple-machine-learning-modeling.html); [Determined blog](https://determined.ai/blog/community-update). Last release v0.38.1-ee on 2025-03-19, last commit the same day; nothing in the ~19 months since; not formally archived; 3,244 stars — [github.com/determined-ai/determined](https://github.com/determined-ai/determined). Dealroom estimates the team shrank by half and web traffic fell sharply — [Dealroom](https://app.dealroom.co/companies/determined_ai) (third-party estimate). I found no official end-of-life announcement.
- **Guild AI** (open-source `guildai`, the closest analog to the planned tool: CLI, no server, runs/compares experiments) — last release 0.8.1 on **2022-05-11**; the only 2025 commits are domain changes (guild.ai → guildai.org, 2025-04-29); 909 stars; 241 open issues. Top contributor `gar1t` has 1,357 commits, against 133 for the next contributor — [github.com/guildai/guildai](https://github.com/guildai/guildai). The guild.ai domain now belongs to an unrelated agent-control-plane startup that raised $44M in 2026 — [Fast AI Jobs listing](https://www.fastaijobs.com/companies/guild-ai) (secondary).
- **MLflow Recipes** (formerly MLflow Pipelines) — *"completely removed from MLflow (#15250)"* in MLflow 3; users told to migrate to tracking/registry or MLflow Projects — [MLflow 3 breaking changes](https://mlflow.org/docs/latest/genai/mlflow-3/breaking-changes). It had been marked deprecated as of 2.21.3 — [MLflow Recipes docs](https://mlflow.org/docs/2.21.3/recipes).

**Alive but slow / mostly one person**
- **Sacred** — 4,382 stars; last release 0.8.7 (2024-11-26); still gets occasional commits (2025-10-22); `Qwlouse` has 895 commits, next contributor 76 — [github.com/IDSIA/sacred](https://github.com/IDSIA/sacred).
- **Polyaxon** — 3,747 stars; still pushing in Oct 2026 (submodule and release-note commits) — [github.com/polyaxon/polyaxon](https://github.com/polyaxon/polyaxon). I didn't verify how much of the OSS core is still developed versus the commercial product.

**Alive and healthy**
- Aim 6,291 stars, pushed 2026-10-30 — [aimhubio/aim](https://github.com/aimhubio/aim); Mage 8,839 stars — [mage-ai/mage-ai](https://github.com/mage-ai/mage-ai); ClearML 6,926 — [clearml/clearml](https://github.com/clearml/clearml); MLflow 28,602 — [mlflow/mlflow](https://github.com/mlflow/mlflow); W&B client 11,310 — [wandb/wandb](https://github.com/wandb/wandb).
- **Flyte vs Prefect** — Prefect 24,102 stars against Flyte 7,835; both active as of Oct 2026 — [PrefectHQ/prefect](https://github.com/PrefectHQ/prefect), [flyteorg/flyte](https://github.com/flyteorg/flyte). Flyte needs Kubernetes; Prefect runs as plain Python. The star gap fits the "low setup friction wins" thesis, but that is my inference.
- **Kubeflow** — 15,941 stars — [kubeflow/kubeflow](https://github.com/kubeflow/kubeflow). Its own 2022 user survey says respondents preferred standalone components because Kubeflow is "too complex to set up and manage for a small team", and named lack of docs as the biggest adoption challenge — [Kubeflow 2022 survey](https://blog.kubeflow.org/kubeflow-user-survey-2022/). The 2023 survey cites installation complexity and painful upgrades — [Kubeflow 2023 survey](https://blog.kubeflow.org/kubeflow-user-survey-2023/). The 2026 SDK survey (30+ practitioners) still reports GPU configuration, scheduling and Kubernetes setup as heavy manual work — [Kubeflow SDK survey 2026](https://blog.kubeflow.org/kubeflow-sdk-user-survey-insights/).

**Maintainer economics**
- Tidelift's 2024 maintainer survey, as reported: about 60% of maintainers are unpaid, almost 60% have quit or considered quitting a project, and almost half are solo maintainers — [SD Times](https://sdtimes.com/open-source/the-state-of-open-source-maintainers/); [TFIR](https://tfir.io/almost-half-of-open-source-maintainers-are-unpaid-volunteers-survey/) (secondary write-ups; reported sample sizes differ, 339 vs 400).

### Inferences
- **Three failure modes, all relevant:** (1) *no business* (Orchest says so explicitly; Ploomber pivoted); (2) *acquisition freeze* (Sematic → W&B, Determined → HPE: the open-source project stops within 1–4 years even after promises to continue); (3) *bus factor = 1 combined with broad scope* (Guild AI). A solo developer building Flowkestra is most exposed to mode 3. Guild AI is the closest precedent: a capable, agentless, CLI-first experiment runner with almost 1k stars that still stalled.
- **Platform scope is what kills these tools.** Every dead tool tried to cover the whole pipeline or the whole platform (DAGs, UI, tracking, scheduling, deployment). MLflow, with Databricks behind it, couldn't justify keeping an opinionated pipeline layer either. Argument against the plan: an "orchestrator + tracker + agent interface" combination repeats this scope.
- Even ~4k stars (Orchest, Ploomber, Sacred) doesn't sustain a project. Stars are not a survival metric.

### Gaps
- No official post-mortem from Ploomber, Sematic, Determined, or Guild AI was found. The reasons given above are inferred from repo signals.
- I didn't verify the Sematic acquisition through a primary W&B announcement (only PitchBook and a secondary timeline).
- I didn't check whether Polyaxon's open-source core is still developed meaningfully. I also didn't verify Mage's company or funding status.

## Q2. Counter-examples: small single-maintainer infra tools that won

### Takeaway
The survivors each do **one job** on **infrastructure the user already has**, need **no server and no config**, and compose with existing tools (shell, SSH, tmux, Slurm). Several are effectively one person's project, but their scope is small enough that one person can keep up.

### Cited Findings
- **pueue** (shell command queue) — 6,403 stars; v4.0.4 released 2026-03-02; still committing in Oct 2026; `Nukesor` has 1,806 commits, against 46 for the next contributor (one dominant maintainer). The README has an explicit "the following features weren't included as they're out of scope" section — [github.com/Nukesor/pueue](https://github.com/Nukesor/pueue).
- **nvitop** — 7,232 stars, active Oct 2026 — [XuehaiPan/nvitop](https://github.com/XuehaiPan/nvitop). **gpustat** — 4,390 stars, active Oct 2026 — [wookayin/gpustat](https://github.com/wookayin/gpustat). Both are read-only GPU viewers: one job, zero config.
- **submitit** (Meta; submits Python functions to Slurm) — 1,643 stars, last push 2026-01-14 — [facebookincubator/submitit](https://github.com/facebookincubator/submitit). It wraps the existing scheduler instead of replacing it.
- **task-spooler** (GPU fork) — 446 stars, last push 2024-03 — [justanhduc/task-spooler](https://github.com/justanhduc/task-spooler). Small, but useful indefinitely while barely maintained, because it's a tiny daemon with a fixed scope.
- **uv** — 91,377 stars — [astral-sh/uv](https://github.com/astral-sh/uv). It isn't a single-maintainer tool: Astral is VC-backed, so it's a weak counter-example for a solo developer. It shows that speed plus being a drop-in for existing workflows (pip, venv) drives adoption.
- **Fabric** (Python SSH execution) — 15,520 stars, last push 2026-04 — [fabric/fabric](https://github.com/fabric/fabric). Generic agentless SSH has long been a solved, durable library.

### Inferences
- Common traits: (a) one verb (queue, view, submit); (b) works on day 1 against SSH, Slurm or NVML without a server; (c) a short, public non-goals list; (d) the tool can be "finished" so maintenance stays light; (e) composable output (CLI/JSON).
- Devil's advocate: users of pueue, task-spooler and tmux already have a working "SSH + queue + nvitop" toolkit. A new runner has to beat a free, zero-learning combination of tools they already have.

### Gaps
- No primary source found on *why* nvitop/gpustat/pueue grew (no maintainer interviews). The traits above are inferred.
- I couldn't locate a canonical "simple-gpu-scheduler" repo (the names I guessed returned 404), so I have no stats for it.

## Q3. Has the "agentless SSH orchestration for ML" category been tried and failed?

### Takeaway
The **hosted** versions (Spell, Grid.ai, Paperspace Gradient) were absorbed or retired. The **SSH-to-your-own-machines** niche didn't fail: **well-funded incumbents already took it** (SkyPilot SSH Node Pools, dstack SSH fleets). This is the strongest argument against the plan.

### Cited Findings
- **Spell** (run ML experiments remotely) — acquired by Reddit in June 2022 and closed to new commercial customers — [TechCrunch](https://techcrunch.com/2022/06/16/reddit-spell-machine-learning/).
- **Grid.ai** — rebranded as Lightning AI in June 2022 and raised $40M; the product was folded into the Lightning platform — [TechCrunch](https://techcrunch.com/2022/06/16/grid-ai-rebrands-as-lightning-ai-raises-40m-to-build-ai-dev-tools).
- **Paperspace Gradient** — after DigitalOcean acquired Paperspace, the Gradient and Core API endpoints were retired as of 15 July 2024 and the Gradient CLI/SDK deprecated — [DigitalOcean docs](https://docs.digitalocean.com/reference/paperspace/gradient/install/index.html.md); [Paperspace is now DigitalOcean](https://www.paperspace.com/paperspace-is-now-digitalocean-ai-gpu-cloud-platform). The `gradient` Python SDK was slated for retirement on 2026-08-15 (search-result summary of a DigitalOcean GitHub repo; not opened).
- **SkyPilot SSH Node Pools** — list hosts in `~/.sky/ssh_node_pools.yaml`, run `sky ssh up`, then `sky launch --infra ssh/<pool>`. Each host only needs to work with plain `ssh` — [SkyPilot docs](https://docs.skypilot.co/en/stable/reservations/existing-machines.html). SkyPilot has 10,770 stars and is active — [skypilot-org/skypilot](https://github.com/skypilot-org/skypilot). Caveat: it installs a runtime on the hosts (older versions deployed k3s), so it isn't strictly "agentless".
- **dstack SSH fleets** — given an SSH user and key, dstack "automatically connects to on-prem servers, installs the required dependencies"; requires Linux + Docker on the hosts — [dstack fleets docs](https://dstack.ai/docs/concepts/fleets/). Added in 0.18.7 (2024) — [HF blog](https://huggingface.co/blog/chansung/dstack-ssh-fleet). dstack has 2,298 stars and is active — [dstackai/dstack](https://github.com/dstackai/dstack).

### Inferences
- Hosted runners died because GPU compute became a commodity and the runner was a thin layer on top of it. The value went to clouds and big platforms.
- Real gaps the incumbents leave: they install daemons, Docker or k3s on the hosts and run an API server. A **truly agentless** tool (nothing installed on the host beyond SSH, Python and tmux) that is **agent-native** is the only wedge left. But SkyPilot or dstack could add an MCP server or agent skill in a sprint, so the edge is narrow and easy to copy.

### Gaps
- I found no documented failed *solo* project specifically doing "Fabric-based ML runner". Absence of evidence isn't evidence it was never tried.
- I didn't verify whether SkyPilot or dstack already ship an MCP server or agent skill as of Oct 2026. This should be checked; it would be decisive.

## Q4. How do OSS tools in this space get discovered in 2025–2026?

### Takeaway
Hacker News gives a measurable but **small, one-time** bump. Agent-skill registries are a new channel, but they are flooded with listings and their numbers are unreliable.

### Cited Findings
- Across 138 AI/LLM-tool launches in 2024–2025, repos gained on average **121 stars within 24h, 189 within 48h, and 289 within a week** of HN exposure. Posting time was a key factor, and the "Show HN" tag showed **no statistical advantage** after controls — [Kraishan, arXiv:2511.04453 (Nov 2025)](https://arxiv.org/abs/2511.04453v1).
- The SKILL.md / Agent Skills format is supported by many coding agents (Claude Code, Codex, Gemini CLI, Cursor, Copilot, etc.) — [noqta.tn (Apr 2026)](https://noqta.tn/en/news/skill-md-open-standard-30-ai-coding-tools-adoption-2026). The skills.sh registry was reported at about 2,600 community skills in April 2026, and another source claims over 90,000 listings by July. These figures conflict and are unverified — [businesstechnavigator](https://businesstechnavigator.com/news/agent-skills-open-standard-skills-sh-2026). Skill supply-chain security is the main concern cited — [Rubin Lake tech radar](https://rubinlake.com/en/technology-radar/developer-ai-and-delivery/agent-skills).
- Older academic work on what drives stars: [Borges et al. 2016](https://ar5iv.labs.arxiv.org/html/1606.04984); [What's in a GitHub Star? (2018)](https://ar5iv.labs.arxiv.org/html/1811.07643).

### Inferences
- About 300 stars a week from a good HN launch is far from the ~4k stars that still didn't save Orchest or Ploomber. Discovery is not the bottleneck; retention and maintenance are.
- A skill registry with tens of thousands of entries makes discovery there a lottery. Being bundled into an incumbent's workflow (for example, listed in SkyPilot or vLLM docs) probably matters more.

### Gaps
- No 2025–2026 data found specifically on awesome-list inclusion effects or on registry-driven installs for infra tools.

## Q5. Risks specific to building for AI agents

### Takeaway
Agent protocols are still changing in breaking ways (the MCP 2026-07-28 revision removed the initialize handshake and sessions). Coding agents already run SSH and tmux themselves, which weakens the need for a dedicated tool.

### Cited Findings
- The **MCP spec release of 2026-07-28** removed the `initialize`/`initialized` handshake and the `Mcp-Session-Id` header, and replaced server-initiated sampling, elicitation and roots requests with Multi Round-Trip Requests. It made new `Mcp-Method`/`Mcp-Name` headers mandatory for Streamable HTTP, deprecated Roots, Sampling, Logging and the legacy HTTP+SSE transport, and introduced a formal deprecation policy with a 12-month minimum window — [MCP blog, 2026-07-28](https://blog.modelcontextprotocol.io/posts/2026-07-28/).
- SSE had already been removed from SDKs in late 2025, e.g. the rust-sdk PRs — [modelcontextprotocol/rust-sdk#562](https://github.com/modelcontextprotocol/rust-sdk/pull/562). Third-party analysis says servers built on the 2025 Streamable HTTP spec "have the most work to do" — [calmara.app](https://calmara.app/blog/mcp-2026-07-28-spec-change) (opinion).
- Agent integration surfaces now include both MCP and SKILL.md skills — [noqta.tn](https://noqta.tn/en/news/skill-md-open-standard-30-ai-coding-tools-adoption-2026). A tool may need to support several of them.

### Inferences
- Make the core a plain CLI with `--json` output, and keep MCP and skill layers thin and swappable. Coding agents use CLIs well, so a good CLI plus a SKILL.md may cover most of the "agent-usable" value without carrying MCP churn.
- Incumbent risk: SkyPilot or dstack already handle SSH hosts, and adding an MCP wrapper is cheap for them. Agents can also just run `ssh host 'tmux new -d ...'` plus `nvidia-smi` directly. Argument against the plan: the agent may not need the tool at all.

### Gaps
- I didn't verify whether SkyPilot, dstack, W&B or MLflow ship official MCP servers or agent skills as of Oct 2026.
- No data found on how often agents choose dedicated tools over raw shell commands.
