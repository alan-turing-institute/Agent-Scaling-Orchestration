"""Default values for the generation settings exposed as command-line flags.

These exist to be argparse defaults. Read the value from `args` (or from a
function parameter) at the point of use rather than importing a constant here,
so that passing a flag actually changes behaviour.
"""

MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
TOP_P = 0.9

# Reasoning budget for models served with an OpenAI-compatible API that accept it.
# Sent as extra_body; endpoints that do not understand the field reject the request.
THINKING_TOKEN_BUDGET = 1024

ORCHESTRATOR_MAX_TOKENS = 4096
ORCHESTRATOR_TEMPERATURE = 0.1
ORCHESTRATOR_TOP_P = 0.5

SUMMARISER_MAX_TOKENS = 8192
SUMMARISER_TEMPERATURE = 0.5
SUMMARISER_MAX_ATTEMPTS = 3

ORCHESTRATOR_ITERATIONS = 10

POOL_SWEEP_REPEATS = 10
POOL_SWEEP_MAX_COMPLETION_TOKENS = 1024
