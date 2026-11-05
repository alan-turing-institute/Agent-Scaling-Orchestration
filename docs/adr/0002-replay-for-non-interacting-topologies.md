# Score non-interacting configurations by replay from an answer matrix

Each agent in the pool answers every question k times, on its own, once. Any configuration whose
agents do not see each other's answers (single agent, vote, cascade, self-consistency, and the
learned-router baselines) is then scored offline from that matrix, rather than by running it again.
Only interacting topologies (sequential, synthesis, debate) and the orchestrator's own calls run
live.

The alternative was running every arm live. Replay makes thousands of configurations and baselines
free to score, and serves one model at a time, which suits a single Spark. The cost is that replay
assumes an agent's answer does not depend on what else the server is batching. That is false on our
vLLM servers: about 20% of answers flip with batch composition. The k repeats sample that noise
rather than hide it, which is why they are kept.
