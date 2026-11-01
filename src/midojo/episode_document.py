"""Deprecated alias for :mod:`midojo.injection_spec_document`, kept until the aliases are removed (D24).

``from midojo.episode_document import EpisodeDocumentError`` gives you
:class:`~midojo.injection_spec_document.InjectionSpecDocumentError`; every other
name here is the same object as the new module's. New code imports
:mod:`midojo.injection_spec_document`.
"""

from __future__ import annotations

from midojo.injection_spec import InjectionSpec as Episode
from midojo.injection_spec_document import (
    LEGACY_DOCUMENT_VERSIONS,
    SUPPORTED_DOCUMENT_VERSIONS,
    UnsupportedDocumentVersion,
    dump_document,
    from_document,
    load_document,
    to_document,
)
from midojo.injection_spec_document import (
    InjectionSpecDocumentError as EpisodeDocumentError,
)

__all__ = [
    "LEGACY_DOCUMENT_VERSIONS",
    "SUPPORTED_DOCUMENT_VERSIONS",
    "Episode",
    "EpisodeDocumentError",
    "UnsupportedDocumentVersion",
    "dump_document",
    "from_document",
    "load_document",
    "to_document",
]
