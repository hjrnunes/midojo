"""The MCP interceptor's declared capability is one value, kept with the plan it is checked against."""

from __future__ import annotations

import dataclasses

import pytest

from midojo_spec.injection import (
    MIDOJO_MCP_CAPABILITY,
    Channel,
    InjectionMode,
    InterceptorCapability,
    MismatchPolicy,
    SelectorSurface,
    Transformation,
)


def test_the_mcp_interceptor_declares_exactly_what_it_places() -> None:
    assert MIDOJO_MCP_CAPABILITY == InterceptorCapability(
        channels=frozenset({Channel.TOOL_OUTPUT}),
        selector_surfaces=frozenset({SelectorSurface.CONTENT_BLOCK, SelectorSurface.STRUCTURED_CONTENT}),
        content_block_types=frozenset({"text"}),
        modes=frozenset(InjectionMode),
        transformations=frozenset(Transformation),
        mismatch_policies=frozenset(MismatchPolicy),
        call_arguments=True,
        result_record=True,
    )


def test_the_declaration_cannot_be_changed_in_place() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        MIDOJO_MCP_CAPABILITY.call_arguments = False  # type: ignore[misc]
