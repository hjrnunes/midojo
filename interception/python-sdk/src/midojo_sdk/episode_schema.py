"""The episode document's version and its closed JSON Schema, shipped as package data.

The schema is the language-neutral half of the contract: every repository that
reads or writes an episode document validates it with the schema, so this
package owns the file. MiDojo itself reads and writes documents through
:mod:`midojo.episode_document`, and a test holds the two in agreement over the
golden vectors.

The plan is ``$defs/plan`` inside the schema. Its version is the enclosing
document's ``schema_version``; :data:`PLAN_CODEC_VERSION` names the codec that
writes it, so a change to the plan's bytes needs a new document version.

The module imports only the standard library, so a tool can read the schema
with no MiDojo, no HTTP client and no MCP server installed.
"""

from __future__ import annotations

import json
from importlib import resources

from midojo_sdk.injection import PLAN_CODEC_VERSION

__all__ = ["EPISODE_DOCUMENT_V1", "PLAN_CODEC_VERSION", "SCHEMA_FILE", "load_schema"]

EPISODE_DOCUMENT_V1 = "midojo-episode-v1"
"""The document version MiDojo writes; the only version it accepts besides the read-only v0 wrapper."""

SCHEMA_FILE = "schemas/midojo-episode-v1.schema.json"
"""The schema's path inside this package."""


def load_schema() -> dict[str, object]:
    """A fresh copy of the episode document v1 JSON Schema, decoded from the package data.

    The copy is the caller's: changing it changes no later call and no other
    caller's schema.
    """
    text = resources.files(__package__).joinpath(SCHEMA_FILE).read_text(encoding="utf-8")
    schema: dict[str, object] = json.loads(text)
    return schema
