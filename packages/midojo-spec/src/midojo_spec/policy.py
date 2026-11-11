"""The tool policy: which of the upstream's tools an agent may call, and what happens to its writes.

A driver states the policy as data (the injection spec document's ``tools``, and the
``bind`` message's ``policy``), and the interceptor decodes it and enforces it. Both read
this one implementation, on the standard library alone. :meth:`ToolPolicy.to_data` and
:meth:`ToolPolicy.from_data` are the document form; ``vectors/policy/policy.json`` pins
what each accepts and refuses.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

WRITE_DISPOSITIONS = frozenset({"forward", "block"})
"""What the interception layer does with a write: forward it upstream, or block it."""


def _exact_str(value: Any, what: str) -> str:
    """``value`` when it is exactly a ``str``; a subclass could carry mutable state, so it raises TypeError."""
    if type(value) is not str:
        raise TypeError(f"{what} must be a str, not {type(value).__name__}")
    return value


def _disposition(value: object, what: str) -> str:
    """``value`` when it is a write disposition; ValueError naming ``what`` otherwise."""
    if type(value) is not str or value not in WRITE_DISPOSITIONS:
        raise ValueError(f"{what} must be 'forward' or 'block', not {value!r}")
    return value


@dataclass(frozen=True)
class ToolPolicy:
    """Which of the upstream's tools the agent may call, and what happens to its writes.

    ``allowed`` names the tools the interception layer exposes. An empty tuple
    exposes none, which is not the same as no policy at all (None mirrors every
    upstream tool). ``write_disposition_default`` applies to a write the policy
    does not name, and ``write_disposition`` overrides it per tool.
    """

    allowed: tuple[str, ...]
    write_disposition_default: Literal["forward", "block"] = "forward"
    write_disposition: Mapping[str, Literal["forward", "block"]] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.allowed, str) or not isinstance(self.allowed, Sequence):
            raise TypeError("allowed must be a sequence of tool names, not a string")
        allowed = tuple(_exact_str(name, "an allowed tool name") for name in self.allowed)
        if len(set(allowed)) != len(allowed):
            raise ValueError("allowed tool names must not repeat")
        object.__setattr__(self, "allowed", allowed)
        _disposition(self.write_disposition_default, "the write disposition default")
        overrides = {
            _exact_str(name, "a tool name"): _disposition(value, f"the write disposition of {name!r}")
            for name, value in self.write_disposition.items()
        }
        object.__setattr__(self, "write_disposition", MappingProxyType(overrides))

    def to_data(self) -> dict[str, Any]:
        """The policy in the injection spec's document form, the way a binding's ``policy`` message carries it."""
        return {
            "allowed": list(self.allowed),
            "write_disposition": {
                "default": self.write_disposition_default,
                "tools": dict(self.write_disposition),
            },
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> ToolPolicy:
        """Decode :meth:`to_data`'s form, the injection spec document's ``tools`` value."""
        if not isinstance(data, Mapping):
            raise ValueError(f"a tool policy is an object, not {type(data).__name__}")
        unknown = set(data) - {"allowed", "write_disposition"}
        if unknown:
            raise ValueError(f"a tool policy has unknown fields: {sorted(unknown)}")
        allowed = data.get("allowed")
        if not isinstance(allowed, list):
            raise ValueError("a tool policy names the tools it allows, as a list")
        disposition = data.get("write_disposition", {})
        if not isinstance(disposition, Mapping):
            raise ValueError("a tool policy's write_disposition is an object")
        unknown = set(disposition) - {"default", "tools"}
        if unknown:
            raise ValueError(f"a tool policy's write_disposition has unknown fields: {sorted(unknown)}")
        overrides = disposition.get("tools", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("a tool policy's write_disposition.tools is an object")
        try:
            return cls(
                tuple(allowed),
                write_disposition_default=disposition.get("default", "forward"),
                write_disposition=dict(overrides),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid tool policy data: {exc}") from None


def tool_policy(tools: ToolPolicy | None) -> tuple[frozenset[str] | None, tuple[str, ...]]:
    """The interceptor's view of a policy: the tools to mirror, and the tools to block.

    ``None`` mirrors every upstream tool and blocks nothing. An empty
    ``allowed`` mirrors no tool, which is the policy's own meaning.
    """
    if tools is None:
        return None, ()
    names = {*tools.allowed, *tools.write_disposition}
    default = tools.write_disposition_default
    blocked = tuple(sorted(name for name in names if tools.write_disposition.get(name, default) == "block"))
    return frozenset(tools.allowed), blocked


__all__ = ["WRITE_DISPOSITIONS", "ToolPolicy", "tool_policy"]
