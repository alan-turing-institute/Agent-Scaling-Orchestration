# Use published fine-tunes; do not train agents ourselves

Prompt personas gave no real differences between agents, so the pool needs agents that differ in
capability. We take existing fine-tuned checkpoints from Hugging Face, on base models we already
serve, rather than fine-tuning or training our own. They must pass a smoke test and a paired gate
against their base before joining the pool. This keeps the project about orchestration rather than
training. The price is that the pool is limited to what exists: there are no credible specialists
for law, truthfulness or commonsense on our bases, and most candidates come from individuals with
self-reported numbers.
