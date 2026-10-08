"""Plancraft: craft a Minecraft item from an inventory, through move, smelt and search tools.

Dagan et al., 2024 (`plancraft` on PyPI, MIT), one of the six benchmarks of
Kim et al. (arXiv:2512.08296), where every multi-agent setup loses to a single
agent. The environment is the package's own, text-only: observations are the
target and the inventory as text, never images.

Success is the package's rule: the target item sits in any slot other than the
output slot [0]. A task can be impossible (the inventory cannot make the
target); calling `impossible` ends the episode and succeeds only on those.

By default the set is the first 100 test examples in file order, which is what
the paper's runs used (its harness takes `instances[:100]` unshuffled): 41
hard, 30 easy, 15 impossible, 14 medium. `--sub_data all` takes all 580.

The task description and tool descriptions follow the paper harness
(`ybkim95/agent-scaling`, MIT: prompts/dataset-shared/plancraft.yaml and
agent_scaling/env/plancraft.py). Needs `plancraft==0.4.9` installed without its
heavy extras (requirements-agentic.txt); only its text environment is imported.
"""
NAME = 'plancraft'
ANSWER_TYPE = 'outcome'
# The paper assigned no persona set to this benchmark; see gpqa_diamond.py.
PERSONA_SET = []
# A single agent's cap on turns. The package's own wrapper allows 30 actions;
# optimal plans for the first 100 average 8 and never exceed 29.
MAX_STEPS = 30
DEFAULT_COUNT = 100

import copy
import hashlib
import json
import random
import threading

from benchmarks.base import Instance
from benchmarks.environment import Outcome, function_tool

TASK_DESCRIPTION = """You are crafting in Minecraft. The task is to craft the target item using the tools provided.
Crafting Grid: The crafting table is organized into a 3x3 grid. Each slot in the grid has a unique identifier:
  - Top row: [A1] [A2] [A3]
  - Middle row: [B1] [B2] [B3]
  - Bottom row: [C1] [C2] [C3]
The output of the crafting process is placed in a designated output slot labeled [0]. You cannot move or smelt items directly into slot [0].
Inventory Slots: The remaining inventory slots (outside of the crafting grid) are used for storing items. These slots are labeled as [I1] to [I36]

Constraints:
- You cannot move or smelt items into [0]
- If an item is not in slot [0] then the recipe is incorrect
- You need to move items from [0] to a free inventory slot to complete the crafting process
- If the target cannot be crafted from this inventory, call impossible"""

SLOT = {"type": "string", "description": "A slot such as [I2], [A1] or [0]"}
TOOLS = [
    function_tool("search", "Search for recipes to craft a specific item.",
                  {"recipe_name": {"type": "string"}}, ["recipe_name"]),
    function_tool("move", "Transfer a specific quantity of items from one slot to another. Specifically, move "
                          "from [slot_from] to [slot_to] with target quantity [quantity].\n\nExample:\n- move("
                          "slot_from=\"[I2]\", slot_to=\"[A1]\", quantity=3) to move 3 items from slot I2 to A1",
                  {"slot_from": SLOT, "slot_to": SLOT, "quantity": {"type": "integer"}},
                  ["slot_from", "slot_to", "quantity"]),
    function_tool("smelt", "Smelt an item in a furnace and move the output to a specific slot. Specifically, "
                           "smelt from [slot_from] to [slot_to] with target quantity [quantity].\n\nExample:\n- "
                           "smelt(slot_from=\"[I5]\", slot_to=\"[I6]\", quantity=1)",
                  {"slot_from": SLOT, "slot_to": SLOT, "quantity": {"type": "integer"}},
                  ["slot_from", "slot_to", "quantity"]),
    function_tool("impossible", "Stop the task if it is certain that it is impossible with the given inventory. "
                                "Specifically, indicate the task is impossible with reason [reason].",
                  {"reason": {"type": "string"}}, ["reason"]),
]

# `gold_search_recipe` lays a recipe out on a grid drawn from the global
# `random`; the paper harness seeds it with 42 before every search. Done under a
# lock, with the global state put back, so concurrent episodes neither race nor
# disturb anything else that draws from `random`.
_SEARCH_LOCK = threading.Lock()


def _search(recipe_name):
    from plancraft.environment.search import gold_search_recipe
    with _SEARCH_LOCK:
        state = random.getstate()
        random.seed(42)
        try:
            return gold_search_recipe(str(recipe_name))
        finally:
            random.setstate(state)


class _NullTable:
    """Stands in for the package's PIL crafting-table renderer, which text tasks never draw."""

    frame = None

    def __init__(self, *args, **kwargs):
        pass

    def clear(self):
        pass

    def add_item_to_slot(self, *args, **kwargs):
        pass

    def remove_item_from_slot(self, *args, **kwargs):
        pass


def _text_env(inventory):
    """The package's environment without its image renderer: 0.2 ms to build instead of 0.3 s."""
    import plancraft.environment.env as env_module
    original = env_module.CraftingTableGUI
    env_module.CraftingTableGUI = _NullTable
    try:
        env = env_module.PlancraftEnvironment(inventory=copy.deepcopy(inventory), resolution="low")
    finally:
        env_module.CraftingTableGUI = original
    return env


class PlancraftTask:
    """One Plancraft example as an `Environment` (see benchmarks.environment)."""

    def __init__(self, metadata, _env=None):
        self.target = metadata["target"]
        self.impossible_task = bool(metadata["impossible"])
        self.metadata = metadata
        if _env is None:
            inventory = {int(slot): dict(item) for slot, item in metadata["slotted_inventory"].items()}
            _env = _text_env(inventory)
        self._env = _env
        self.done = False
        self.called_impossible = False
        self.crafted = False

    # -- the Environment protocol ---------------------------------------------

    def task_prompt(self):
        return f"{TASK_DESCRIPTION}\n\n{self.observation()}"

    def tools(self):
        return TOOLS

    def call(self, name, arguments):
        from plancraft.environment.actions import MoveAction, SmeltAction
        if self.done:
            return "The task is over; no further actions are taken."
        try:
            if name == "search":
                return _search(arguments.get("recipe_name", ""))
            if name == "impossible":
                self.called_impossible, self.done = True, True
                return "Stopped: you declared the task impossible."
            if name in ("move", "smelt"):
                action = (MoveAction if name == "move" else SmeltAction)(
                    slot_from=arguments.get("slot_from"), slot_to=arguments.get("slot_to"),
                    quantity=arguments.get("quantity"))
                inventory = self._env.step(action)["inventory"]
                if any(item["type"] == self.target and slot != 0 for slot, item in inventory.items()):
                    self.crafted, self.done = True, True
                return self.observation(inventory)
            return f"ERROR: unknown tool {name!r}; use search, move, smelt or impossible"
        except Exception as error:  # the package raises on malformed slots and quantities
            return f"ERROR: {error}"

    def outcome(self):
        if self.called_impossible:
            return Outcome(success=self.impossible_task, fingerprint="impossible",
                           detail={"declared_impossible": True, "impossible_task": self.impossible_task})
        if self.crafted:
            return Outcome(success=True, fingerprint=f"crafted:{self.target}", detail={"crafted": True})
        state = json.dumps(self._inventory(), sort_keys=True)
        return Outcome(success=False, fingerprint="inventory:" + hashlib.sha1(state.encode()).hexdigest()[:10],
                       detail={"crafted": False, "impossible_task": self.impossible_task})

    def resume(self):
        """Let the next agent overrule an `impossible` declaration; a crafted item stays crafted."""
        self.called_impossible = False
        self.done = self.crafted

    def fork(self):
        env = object.__new__(type(self._env))
        env.__dict__.update(self._env.__dict__)
        # Only the inventory and the pending recipe's ingredients change as the
        # agent acts; the recipe tables are shared.
        env.state = copy.deepcopy(self._env.state)
        env.ingredients_idxs = list(getattr(self._env, "ingredients_idxs", []))
        twin = PlancraftTask(self.metadata, _env=env)
        twin.done, twin.called_impossible, twin.crafted = self.done, self.called_impossible, self.crafted
        return twin

    # -- helpers -----------------------------------------------------------------

    def _inventory(self):
        return self._env.step()["inventory"]

    def observation(self, inventory=None):
        from plancraft.environment.env import target_and_inventory_to_text_obs
        return target_and_inventory_to_text_obs(self.target, inventory or self._inventory())


def ENVIRONMENT(instance):
    return PlancraftTask(instance.metadata)


def load_instances(args, split='test'):
    from plancraft.environment.env import target_and_inventory_to_text_obs
    from plancraft.simple import get_plancraft_examples
    # The package ships train/val/test; the paper's runs are on test.
    examples = get_plancraft_examples('val' if split == 'train' else 'test')
    if (getattr(args, 'sub_data', '') or '') != 'all':
        examples = examples[:args.data_size if args.data_size and args.data_size > 0 else DEFAULT_COUNT]
    elif args.data_size and args.data_size > 0:
        examples = examples[:args.data_size]

    instances = []
    for example in examples:
        inventory = {str(slot): {"type": item["type"], "quantity": int(item["quantity"])}
                     for slot, item in example.slotted_inventory.items()}
        difficulty = example.complexity_split or ("impossible" if example.impossible else "unknown")
        instances.append(Instance(
            question=target_and_inventory_to_text_obs(example.target, example.slotted_inventory),
            answer="impossible" if example.impossible else example.target,
            id=f"{NAME}:{example.id}",
            tags=[f"benchmark: {NAME}", f"difficulty: {difficulty}", "tools: 4"],
            metadata={"example_id": example.id, "target": example.target, "impossible": bool(example.impossible),
                      "slotted_inventory": inventory, "complexity_split": difficulty,
                      "optimal_path_length": example.optimal_path_length},
        ))
    return instances
