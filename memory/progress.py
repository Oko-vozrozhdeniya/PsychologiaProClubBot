"""Progress checkpoints: validated state only, no database or AI calls."""
from dataclasses import dataclass, replace
from datetime import datetime

STATES = {"planned", "active", "paused", "completed"}
ACTION_STATES = {"planned", "done", "deferred"}
EVIDENCE = {"reported", "confirmed"}


def valid_text(value, optional=False):
    if not isinstance(value, str) or len(value) > 1000 or (not optional and not value.strip()):
        raise ValueError("Invalid text")
    return value.strip()


def valid_time(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("Timezone required")
        return parsed.isoformat()
    except (AttributeError, ValueError) as exc:
        raise ValueError("Invalid timestamp") from exc


@dataclass(frozen=True)
class Action:
    key: str
    description: str
    status: str
    source: str

    def __post_init__(self):
        for value in (self.key, self.description, self.source):
            valid_text(value)
        if self.status not in ACTION_STATES:
            raise ValueError("Invalid action status")


@dataclass(frozen=True)
class Result:
    description: str
    evidence: str
    source: str

    def __post_init__(self):
        valid_text(self.description)
        valid_text(self.source)
        if self.evidence not in EVIDENCE:
            raise ValueError("Invalid evidence")


@dataclass(frozen=True)
class Checkpoint:
    user_id: int
    program: str
    goal: str
    stage: str
    actions: tuple
    results: tuple
    next_step: str
    source: str
    updated_at: str
    version: int = 1

    def __post_init__(self):
        if type(self.user_id) is not int or self.user_id <= 0:
            raise ValueError("Invalid user")
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("Invalid version")
        for value in (self.program, self.goal, self.source):
            valid_text(value)
        valid_text(self.next_step, optional=True)
        valid_time(self.updated_at)
        if self.stage not in STATES:
            raise ValueError("Invalid stage")
        if type(self.actions) is not tuple or type(self.results) is not tuple:
            raise ValueError("Invalid collections")
        if len(self.actions) > 100 or len(self.results) > 100:
            raise ValueError("Too many records")
        if any(not isinstance(a, Action) for a in self.actions):
            raise ValueError("Invalid actions")
        if any(not isinstance(r, Result) for r in self.results):
            raise ValueError("Invalid results")
        if len({a.key for a in self.actions}) != len(self.actions):
            raise ValueError("Duplicate actions")


def update(checkpoint, user_id, source, updated_at, *, action=None, result=None,
           stage=None, next_step=None, expected_version=None):
    if not isinstance(checkpoint, Checkpoint) or type(user_id) is not int or user_id != checkpoint.user_id:
        raise ValueError("Invalid owner")
    valid_text(source)
    if expected_version is not None and expected_version != checkpoint.version:
        raise ValueError("Version conflict")
    if valid_time(updated_at) < valid_time(checkpoint.updated_at):
        raise ValueError("Old event")
    if action is not None and not isinstance(action, Action):
        raise ValueError("Invalid action")
    if result is not None and not isinstance(result, Result):
        raise ValueError("Invalid result")
    if stage is not None and stage not in STATES:
        raise ValueError("Invalid stage")
    if next_step is not None:
        valid_text(next_step, optional=True)
    actions = list(checkpoint.actions)
    if action is not None:
        actions = [a for a in actions if a.key != action.key] + [action]
    results = list(checkpoint.results)
    if result is not None and result not in results:
        results.append(result)
    if len(actions) > 100 or len(results) > 100:
        raise ValueError("Too many records")
    changes = dict(actions=tuple(actions), results=tuple(results),
                   stage=checkpoint.stage if stage is None else stage,
                   next_step=checkpoint.next_step if next_step is None else next_step)
    if all(getattr(checkpoint, key) == value for key, value in changes.items()):
        return checkpoint
    return replace(checkpoint, **changes, source=source,
                   updated_at=updated_at, version=checkpoint.version + 1)


def summary(checkpoint, user_id):
    if not isinstance(checkpoint, Checkpoint) or type(user_id) is not int or checkpoint.user_id != user_id:
        raise ValueError("Invalid owner")
    return {
        "goal": checkpoint.goal,
        "stage": checkpoint.stage,
        "planned": [a.description for a in checkpoint.actions if a.status == "planned"],
        "completed": [a.description for a in checkpoint.actions if a.status == "done"],
        "deferred": [a.description for a in checkpoint.actions if a.status == "deferred"],
        "reported": [r.description for r in checkpoint.results if r.evidence == "reported"],
        "confirmed": [r.description for r in checkpoint.results if r.evidence == "confirmed"],
        "next_step": checkpoint.next_step,
        "version": checkpoint.version,
    }
