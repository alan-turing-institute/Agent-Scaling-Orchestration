"""A team as a small graph of calls.

A team used to mean an unordered list of persona names that each answered the
question once and voted. That cannot describe a debate round, a hub reading its
workers, or a pipeline where a critic reads a solver. A `TeamConfig` can: each
`Stage` names its persona, role and model, and lists the earlier stages it reads
and in what form. One runner (`runner.py`) executes any config.

    vote        stages that read nothing, then a majority vote
    debate      each round's stages read all of the previous round, vote at the end
    centralized workers read nothing, a hub reads them all and decides
    pipeline    each stage reads every stage before it, the last one decides
    synthesis   workers read nothing, a synthesiser reads them all and decides
    parallel    several configs side by side, voting over their final answers

Stages are listed in execution order and may only read stages listed before
them, so a config cannot contain a cycle. `config_id` is a stable hash of
everything that changes behaviour, including the role-template version, so two
records with the same id were produced by the same prompts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional, Sequence

from roles import HANDOFFS, ROLE_TEMPLATES_VERSION, ROLES

AGGREGATES = ("vote", "stage")


@dataclass(frozen=True)
class Input:
    source: str
    handoff: str = "full"

    def to_dict(self):
        return {"source": self.source, "handoff": self.handoff}


@dataclass(frozen=True)
class Stage:
    id: str
    persona: Optional[str]
    role: str = "solver"
    model: str = "default"
    inputs: tuple = ()
    max_tokens: Optional[int] = None

    def to_dict(self):
        record = {
            "id": self.id,
            "persona": self.persona,
            "role": self.role,
            "model": self.model,
            "inputs": [i.to_dict() for i in self.inputs],
        }
        if self.max_tokens is not None:
            record["max_tokens"] = self.max_tokens
        return record


@dataclass(frozen=True)
class TeamConfig:
    stages: tuple
    aggregate: str = "vote"
    final: Optional[str] = None
    vote_over: Optional[tuple] = None
    max_tokens: Optional[int] = None
    name: str = field(default="", compare=False)

    def __post_init__(self):
        self.validate()

    # -- structure ---------------------------------------------------------

    def validate(self):
        if not self.stages:
            raise ValueError("a team needs at least one stage")
        seen = set()
        for stage in self.stages:
            if stage.id in seen:
                raise ValueError(f"duplicate stage id {stage.id!r}")
            if stage.role not in ROLES:
                raise ValueError(f"stage {stage.id!r}: unknown role {stage.role!r}; known: {sorted(ROLES)}")
            for item in stage.inputs:
                if item.source not in seen:
                    raise ValueError(f"stage {stage.id!r} reads {item.source!r}, which is not an earlier stage")
                if item.handoff not in HANDOFFS:
                    raise ValueError(f"stage {stage.id!r}: unknown handoff {item.handoff!r}; known: {HANDOFFS}")
            seen.add(stage.id)
        if self.aggregate not in AGGREGATES:
            raise ValueError(f"unknown aggregate {self.aggregate!r}; known: {AGGREGATES}")
        if self.aggregate == "stage" and self.final not in seen:
            raise ValueError(f"aggregate 'stage' needs final to name a stage, got {self.final!r}")
        for stage_id in self.vote_over or ():
            if stage_id not in seen:
                raise ValueError(f"vote_over names {stage_id!r}, which is not a stage")

    @property
    def stage_ids(self):
        return [s.id for s in self.stages]

    def stage(self, stage_id):
        for s in self.stages:
            if s.id == stage_id:
                return s
        raise KeyError(stage_id)

    def voters(self):
        """The stages a vote counts: `vote_over`, else every stage nothing else reads."""
        if self.vote_over:
            return list(self.vote_over)
        read = {i.source for s in self.stages for i in s.inputs}
        return [s.id for s in self.stages if s.id not in read]

    def answer_stages(self):
        """The stages whose answers decide the team's answer."""
        return [self.final] if self.aggregate == "stage" else self.voters()

    def layers(self):
        """Stage ids grouped by dependency depth; stages in one layer can run together."""
        depth = {}
        for s in self.stages:
            depth[s.id] = 1 + max((depth[i.source] for i in s.inputs), default=-1)
        out = [[] for _ in range(max(depth.values()) + 1)]
        for s in self.stages:
            out[depth[s.id]].append(s.id)
        return out

    def personas(self):
        return sorted({s.persona for s in self.stages if s.persona})

    def models(self):
        return sorted({s.model for s in self.stages})

    # -- identity and serialisation ----------------------------------------

    def to_dict(self):
        record = {
            "stages": [s.to_dict() for s in self.stages],
            "aggregate": self.aggregate,
            "final": self.final,
            "vote_over": list(self.vote_over) if self.vote_over else None,
            "max_tokens": self.max_tokens,
        }
        if self.name:
            record["name"] = self.name
        return record

    @property
    def config_id(self):
        behaviour = self.to_dict()
        behaviour.pop("name", None)
        behaviour["role_templates_version"] = ROLE_TEMPLATES_VERSION
        blob = json.dumps(behaviour, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def from_dict(cls, raw):
        """Build a config from its JSON shape, refusing keys it does not know.

        A typo such as `"aggregation": "stage"` would otherwise be dropped and
        the config would run as a vote without saying so.
        """
        _check_keys("config", raw, _CONFIG_KEYS, required={"stages"})
        stages = []
        for s in raw["stages"]:
            _check_keys("stage", s, _STAGE_KEYS, required={"id"})
            inputs = []
            for i in s.get("inputs", []):
                if not isinstance(i, dict):
                    raise ValueError(f"stage {s['id']!r}: each input must be an object like "
                                     f'{{"source": "<stage id>", "handoff": "full"}}, got {i!r}')
                _check_keys(f"input of stage {s['id']!r}", i, _INPUT_KEYS, required={"source"})
                inputs.append(Input(i["source"], i.get("handoff", "full")))
            stages.append(Stage(
                id=s["id"],
                persona=s.get("persona"),
                role=s.get("role", "solver"),
                model=s.get("model", "default"),
                inputs=tuple(inputs),
                max_tokens=s.get("max_tokens"),
            ))
        stages = tuple(stages)
        vote_over = raw.get("vote_over")
        return cls(
            stages=stages,
            aggregate=raw.get("aggregate", "vote"),
            final=raw.get("final"),
            vote_over=tuple(vote_over) if vote_over else None,
            max_tokens=raw.get("max_tokens"),
            name=raw.get("name", ""),
        )


_CONFIG_KEYS = {"stages", "aggregate", "final", "vote_over", "max_tokens", "name"}
_STAGE_KEYS = {"id", "persona", "role", "model", "inputs", "max_tokens"}
_INPUT_KEYS = {"source", "handoff"}


def _check_keys(what, raw, allowed, required=()):
    if not isinstance(raw, dict):
        raise ValueError(f"{what} must be an object, got {raw!r}")
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{what}: unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")
    missing = set(required) - set(raw)
    if missing:
        raise ValueError(f"{what}: missing {sorted(missing)}")


def _require(value, what):
    if not value:
        raise ValueError(f"{what} needs a persona")
    return value


# -- builders ----------------------------------------------------------------

def _unique_ids(personas):
    """Persona names as stage ids, suffixed only where a persona repeats.

    A vote's stage ids are its persona names, so its per-agent counts land under
    the same keys the scoreboard and every saved record already use.
    """
    counts, ids = {}, []
    for persona in personas:
        counts[persona] = counts.get(persona, 0) + 1
        ids.append(persona if counts[persona] == 1 else f"{persona}#{counts[persona]}")
    return ids


def vote(personas: Sequence[str], model="default", max_tokens=None, name="vote"):
    stages = tuple(Stage(id=sid, persona=p, model=model)
                   for sid, p in zip(_unique_ids(personas), personas))
    return TeamConfig(stages=stages, aggregate="vote", max_tokens=max_tokens, name=name)


def debate(personas: Sequence[str], rounds=1, model="default", handoff="full", max_tokens=None,
           name="debate"):
    """Round 0 answers alone; each later round reads all of the round before, then a vote.

    `rounds` counts the exchange rounds after the first answer, as `--debate_rounds`
    does in `main.py`. Unlike `main.py`'s decentralized branch, which passes peers
    only the last number in each response, the handoff here is chosen explicitly.
    """
    base = _unique_ids(personas)
    stages, previous = [], []
    for r in range(rounds + 1):
        current = []
        for sid, persona in zip(base, personas):
            stage_id = sid if r == 0 else f"{sid}@r{r}"
            stages.append(Stage(
                id=stage_id, persona=persona, model=model,
                role="solver" if r == 0 else "debater",
                inputs=tuple(Input(src, handoff) for src in previous),
            ))
            current.append(stage_id)
        previous = current
    return TeamConfig(stages=tuple(stages), aggregate="vote", vote_over=tuple(previous),
                      max_tokens=max_tokens, name=name)


def centralized(workers: Sequence[str], hub: str, model="default", hub_model=None, handoff="full",
                max_tokens=None, name="centralized"):
    _require(hub, "centralized: the hub")
    ids = _unique_ids(workers)
    stages = [Stage(id=sid, persona=p, model=model) for sid, p in zip(ids, workers)]
    hub_id = f"hub:{hub}"
    stages.append(Stage(id=hub_id, persona=hub, role="hub", model=hub_model or model,
                        inputs=tuple(Input(sid, handoff) for sid in ids)))
    return TeamConfig(stages=tuple(stages), aggregate="stage", final=hub_id,
                      max_tokens=max_tokens, name=name)


def synthesis(workers: Sequence[str], synthesiser: str, model="default", synthesiser_model=None,
              handoff="full", max_tokens=None, name="synthesis"):
    _require(synthesiser, "synthesis: the synthesiser")
    ids = _unique_ids(workers)
    stages = [Stage(id=sid, persona=p, model=model) for sid, p in zip(ids, workers)]
    final = f"synth:{synthesiser}"
    stages.append(Stage(id=final, persona=synthesiser, role="synthesiser",
                        model=synthesiser_model or model,
                        inputs=tuple(Input(sid, handoff) for sid in ids)))
    return TeamConfig(stages=tuple(stages), aggregate="stage", final=final,
                      max_tokens=max_tokens, name=name)


def pipeline(steps: Sequence, model="default", handoff="full", max_tokens=None, name="pipeline"):
    """`steps` is a sequence of `(role, persona)` or `(role, persona, model)`.

    Each stage reads every stage before it, so a reviser sees both the solution
    and the review of it. The last stage's answer is the team's.
    """
    stages = []
    for k, step in enumerate(steps, start=1):
        role, persona = step[0], step[1]
        step_model = step[2] if len(step) > 2 and step[2] else model
        stages.append(Stage(
            id=f"{k}:{role}:{persona}", persona=persona, role=role, model=step_model,
            inputs=tuple(Input(s.id, handoff) for s in stages),
        ))
    return TeamConfig(stages=tuple(stages), aggregate="stage", final=stages[-1].id,
                      max_tokens=max_tokens, name=name)


def parallel(*configs: TeamConfig, name="parallel", max_tokens=None):
    """Several configs side by side; the team votes over each one's answer.

    Stage ids are prefixed with the branch number so branches cannot collide,
    e.g. two `solve -> check` pipelines voted on.
    """
    stages, finals = [], []
    for b, config in enumerate(configs, start=1):
        prefix = f"b{b}/"
        for s in config.stages:
            stages.append(Stage(
                id=prefix + s.id, persona=s.persona, role=s.role, model=s.model,
                inputs=tuple(Input(prefix + i.source, i.handoff) for i in s.inputs),
                max_tokens=s.max_tokens,
            ))
        finals.extend(prefix + sid for sid in config.answer_stages())
    if len(finals) != len(configs):
        raise ValueError("parallel() needs each branch to end in one answer stage (a pipeline, hub or synthesiser)")
    return TeamConfig(stages=tuple(stages), aggregate="vote", vote_over=tuple(finals),
                      max_tokens=max_tokens, name=name)


def split_budget(config: TeamConfig, total_tokens: int) -> TeamConfig:
    """Give every stage an equal share of a per-question token budget.

    The crude version of matched compute: n stages get total/n each, so a vote of
    four and a four-stage pipeline spend the same. Records keep `config_id`,
    which changes with the budget, so budgets cannot be mixed up afterwards.
    """
    share = max(1, total_tokens // len(config.stages))
    return TeamConfig(stages=config.stages, aggregate=config.aggregate, final=config.final,
                      vote_over=config.vote_over, max_tokens=share,
                      name=f"{config.name}@{total_tokens}" if config.name else "")
