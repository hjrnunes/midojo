"""Deprecated alias for :mod:`midojo_sdk.injection_spec_schema`, named after the document's old name.

Every name here is the same object as the new module's. New code imports
:mod:`midojo_sdk.injection_spec_schema`.
"""

from __future__ import annotations

from midojo_sdk.injection_spec_schema import (
    EPISODE_DOCUMENT_V1,
    INJECTION_SPEC_DOCUMENT_V1,
    LEGACY_INJECTION_SPEC_DOCUMENT_V1,
    LEGACY_SCHEMA_FILE,
    PLAN_CODEC_VERSION,
    SCHEMA_FILE,
    load_legacy_schema,
    load_schema,
)

__all__ = [
    "EPISODE_DOCUMENT_V1",
    "INJECTION_SPEC_DOCUMENT_V1",
    "LEGACY_INJECTION_SPEC_DOCUMENT_V1",
    "LEGACY_SCHEMA_FILE",
    "PLAN_CODEC_VERSION",
    "SCHEMA_FILE",
    "load_legacy_schema",
    "load_schema",
]
