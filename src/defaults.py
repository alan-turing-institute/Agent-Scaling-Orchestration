"""Default values for the generation settings exposed as command-line flags.

These exist to be argparse defaults. Read the value from `args` (or from a
function parameter) at the point of use rather than importing a constant here,
so that passing a flag actually changes behaviour.
"""

import os

# OpenAICompatChatWrapper.complete forces max_tokens to 4096, so 4096 is what
# every vLLM run has used whatever this said. Defaulting to it means the flag
# keeps meaning the same thing once the wrapper honours it.
MAX_NEW_TOKENS = 4096
TEMPERATURE = 1.0
TOP_P = 0.9

# Reasoning budget for a thinking model. Off unless set, because vLLM rejects
# the field on a server started without --reasoning-config, which is every
# non-thinking model. THINKING_TOKEN_BUDGET in the environment opts in, so a
# sweep can set it once rather than on every command line.
_thinking_token_budget = os.environ.get("THINKING_TOKEN_BUDGET")
THINKING_TOKEN_BUDGET = int(_thinking_token_budget) if _thinking_token_budget else None

# Tagging asks for a short list of tags, and has always been capped at 512.
TAGGING_MAX_NEW_TOKENS = 512

ORCHESTRATOR_MAX_TOKENS = 4096
ORCHESTRATOR_TEMPERATURE = 0.1
ORCHESTRATOR_TOP_P = 0.5

SUMMARISER_MAX_TOKENS = 8192
SUMMARISER_TEMPERATURE = 0.5
SUMMARISER_MAX_ATTEMPTS = 3

ORCHESTRATOR_ITERATIONS = 10

POOL_SWEEP_REPEATS = 10
POOL_SWEEP_MAX_COMPLETION_TOKENS = 1024
