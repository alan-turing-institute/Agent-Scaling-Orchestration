# Agents are checkpoints with a thinking mode, not personas

The orchestrator's pool used to be 50 role-prompt personas on one model. Across two E2 sweeps
(Qwen3.5-0.8B and 4B), choosing personas never beat choosing them at random. On one model, persona
differences were mostly run-to-run noise: at 0.8B, under 1% of the variance and 29% correctness flips
on identical requests. So an agent is now defined by its checkpoint (base model or published
fine-tune), its thinking mode and its prompt, plus tools later. Two agents count as different only
if they differ in capability or cost. A persona survives only where it beats the same model's
neutral prompt on training folds; that is kept as an ablation, not as the pool.

## Consequences

The code's pool (`build_agent_pool`) lists personas, and a team shares one model
(`team_config.for_team(..., model="default")`). Both have to change so that teams can span
several servers.
