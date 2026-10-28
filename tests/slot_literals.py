"""A placement's digests and slot, written from literal payloads and targets, apart from MiDojo's helpers.

As documented: a payload digest is ``sha256:<hex>`` of the text's UTF-8
bytes; a slot is ``{"id", "target", "mode"}``, its target in the plan codec's
form and its id the digest of that target's sorted-key compact UTF-8 JSON.
Only ``hashlib`` and ``json`` compute anything here.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

NOT_PLACED = {"payload_digest": None, "placed_digest": None, "slot": None}
"""An outcome that placed nothing names no payload and no slot."""


def digest(text: str) -> str:
    """``sha256:`` and the lowercase hex SHA-256 of ``text``'s UTF-8 bytes."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def codec_target(**fields: Any) -> dict[str, Any]:
    """A tool-output target in the plan codec's form: every field, those not given at their defaults.

    ``call_arguments`` and ``result_record`` appear only when given, as the codec leaves them out when empty.
    """
    defaults = {
        "tool": None,
        "surface": "content_block",
        "content_block_index": None,
        "content_block_type": None,
        "structured_content_path": [],
        "template_path": [],
        "template_key": None,
    }
    return defaults | fields


def slot_id(target: dict[str, Any]) -> str:
    """The digest of the target's JSON with sorted keys and no whitespace, as UTF-8 (non-ASCII unescaped)."""
    return digest(json.dumps(target, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def placed(payload: str, written: str, target: dict[str, Any], mode: str = "append") -> dict[str, Any]:
    """The fields a placement adds: the payload's digest before its transformation, the written text's, the slot."""
    return {
        "payload_digest": digest(payload),
        "placed_digest": digest(written),
        "slot": {"id": slot_id(target), "target": target, "mode": mode},
    }
