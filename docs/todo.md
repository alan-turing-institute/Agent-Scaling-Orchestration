# To do

Code changes and experiments still needed, by section of [`plan.md`](plan.md) and by importance:
- **P0:** needed for the paper's core claim.
- **P1:** strengthens it.
- **P2:** optional.

Tick items off and add new ones as work moves. Done items go to the bottom with the date.

## 1. Housekeeping

- [ ] **P2** Delete the merged remote branches `orchestrator/small-agents`, `experiments/x1-teams` and `experiments/frontier-sonnet`. The permission check blocks `git push --delete` from Claude, so the user runs it.
- [ ] **P2** Decide on the stale `refactor/benchmark-registry` branch, PR #2 (repo layout) and PR #1 (Carolyn's).

## 2. Matched-compute baselines (plan §3)

Code:
- [ ] **P0** maj@k scorer: majority over k samples per question, estimated by averaging over random k-subsets of a larger pool; accuracy against mean total tokens; thinking off and on.
- [ ] **P0** A run config for extra 35B samples, via `X1_CONFIG` like L2: thinking off s3–s7 at X1 settings, thinking on s1–s2 at the long limit, hard and tool pools.
- [ ] **P0** CV-selected best worker: pick per benchmark on the training folds, score on the held-out fold. Analysis only.
- [ ] **P0** `x1_hub.py --aggregate select`: Universal Self-Consistency / judge-pick prompt, where the orchestrator must return one worker's answer.
- [ ] **P0** `x1_hub.py --workers-from`: let the orchestrator read another set of stage-1 rows (the 35B's own samples) for Self-MoA.
- [ ] **P1** Cascade and kNN router baselines over the answer matrix (#22, #23); X4's replay scorer may already cover the cascade.
- [ ] **P2** Symbolic-MoE baseline, the closest published training-free method: adapt its expert recruitment to our pool and run live.

Experiments:
- [ ] **P0** The 35B thinking on, samples s1–s2, long limit, hard and tool pools (k = 3 to match the orchestrator).
- [ ] **P0** The 35B thinking off, samples s3–s7 (with L2's s0–s2, k up to 8).
- [ ] **P0** USC / judge-pick over the X1 summaries, headline modes, 5 folds × 3.
- [ ] **P0** Self-MoA: the 35B orchestrator over 4 of its own samples.
- [ ] **P1** Cascade and router curves on the same accuracy–cost axes.

## 3. Ablations (plan §4)

Code (most as `x1_hub.py` options that change the orchestrator's inputs, replaying saved stage-1 rows):
- [ ] **P0** `--inputs blank|other-question|all-wrong|dup-minority` (A3).
- [ ] **P0** `--order reverse|random|specialists-first`, with seeded permutations (A4).
- [ ] **P0** `--workers` subset selection: leave-one-out, personas-only, specialists-only, strongest removed (A7).
- [ ] **P0** `--personas-from x0`: replace persona workers with X0 neutral 4B samples s0–s2 (A8).
- [ ] **P0** Orchestrator model and thinking from the command line, or via config variants: 4B orchestrator (A6/L1), thinking on (B5).
- [ ] **P1** `--handoff answer|summary|full|last-n` with a 131k serve (A5/L4).
- [ ] **P1** Analysis scripts: gain by vote margin (A1); error correlation (B1); gain against difficulty (B2); which worker is copied (B6).
- [ ] **P2** K\* of each team against its gain (the predecessor paper's diversity measure); analysis only.
- [ ] **P2** `--modes choose`: the orchestrator picks each specialist's thinking mode per task (L10), replayed from stage 1's two modes.
- [ ] **P0** Statistics: question-clustered paired bootstrap with Holm (extend `scripts/compare_runs.py`), plus Pareto plots (A10).

Experiments (orchestrator calls over saved answers; each about one stage-2 pass):
- [ ] **P0** A3 blank / other-question / all-wrong inputs.
- [ ] **P0** A4 order permutations.
- [ ] **P0** A7 leave-one-out and subsets (size 3 and singles first).
- [ ] **P0** A8 personas → neutral samples.
- [ ] **P0** A6 4B orchestrator (L1).
- [ ] **P1** B5 orchestrator thinking on.
- [ ] **P1** B3 select vs synthesise (shares code with USC).
- [ ] **P1** A5 handoff format (131k serve).
- [ ] **P1** B7 tool tasks: orchestrator episode with no reports, and with reports from another task.
- [ ] **P1** L3 solo runs of the specialists' other bases, Gemma-E4B and Ministral-3B-Instruct, at X1 settings (new worker runs).
- [ ] **P2** L10 run, once `--modes choose` exists.
- [ ] **P2** L9 specialists with a larger thinking cap (new worker runs).
- [ ] **P2** B4 adversarial or weak member (Qwen3.5-0.8B answers need new runs).
- [ ] **P2** B10 style anonymisation; B9 extra rounds.

## 4. Division of labour (plan §5)

Code:
- [ ] **P1** Per-step skill pilot:
  - execution given a correct plan, per agent (needs reference plans for WorkBench and Plancraft);
  - plan quality scored with a fixed executor.
- [ ] **P1** An X1-style runner for `team_config.delegated` with real specialists on separate endpoints (two Sparks); log tokens per agent and every delegation.
- [ ] **P1** An `ask_expert` tool and a transcript hand-off for a small driver calling the 35B (D3; absorbs the old X5).
- [ ] **P2** A fixed pipeline arm, plan → execute → check (D4), via `team_config.pipeline`.

Experiments (tool pool, fold 0 pilot first):
- [ ] **P1** The per-step skill pilot.
- [ ] **P1** D2: the 35B manager delegating to specialists, with controls C1 (35B to itself) and C2 (random worker).
- [ ] **P1** D3: small driver with `ask_expert`.
- [ ] **P2** D4: fixed pipeline.

## 5. Escalation (plan §6)

- [ ] **P1** X4 solve first: in progress in the `-x4` worktree (another session). Fold its results into the plan when it reports.

## 6. Evaluation fixes (plan §7)

- [ ] **P0** Independent judge for BrowseComp-Plus and Finance-Agent. Add a judge-model option to `x1_regrade.py` and regrade every arm; add a human-labelled subset for agreement.
- [ ] **P0** Team-choice provenance: confirm the 2026-10-22 smoke-test questions are not in the test folds, and write down how each team was chosen.
- [ ] **P0** Cost axes: list-price $ and active-parameter FLOPs per arm (price table in `notes/paper-evaluation-norms.md` §2); wall-clock from logs.
- [ ] **P1** Contamination: AIME by year (needs year metadata); a post-release slice (AIME 2026 / HMMT Feb 2026).
- [ ] **P1** Protocol deviations table for the four tool benchmarks; WorkBench side-effect rate.
- [ ] **P2** Check MATH-500's count: the loader gives 262, the brief says 264.

## 7. Frontier (plan §9; needs the user's go-ahead)

- [ ] **P2** Sonnet on the hard set for more folds (about $1.60 a fold of list-price usage).
- [ ] **P2** A local open frontier-class model (e.g. gpt-oss-120b) on fold 0 as a bridge.
- [ ] **P2** Sonnet on the tool set beyond fold 0 (about $70 a fold, mostly BrowseComp-Plus).

## 8. Results page

- [ ] **P0** An X1 section comparing against the 35B thinking on and at matched compute, as §2 baselines land.
- [ ] **P1** The ablations tab, as items in §3 land.

## Done

- 2026-11-13: earlier plans triaged (plan §10). Dropped: X3 per-task selection, agent cards, memory, cost budgets, pool expansion and E10 on single questions. X5 was folded into D3.
- 2026-11-13: housekeeping. PR #38 merged the frontier harness, the L2 scripts, orchestrator-52's by-dataset builders and these docs. The main checkout was fast-forwarded. The `-hard`, `-x0`, `-x1`, `-docs` and `-frontier` worktrees and their local branches were removed; only `-x4` (X4, running) remains.
- 2026-11-13: Sonnet frontier baseline, fold 0 (results page "Frontier model" tab).
- 2026-11-13: matched-compute check from existing data: the 35B thinking on vs orchestrator + team (plan §1).
- 2026-11-13: literature notes (`notes/paper-*.md`) and this plan.
- 2026-11-06: X1 complete, including T8; L2 (35B alone, thinking off), including T8.
