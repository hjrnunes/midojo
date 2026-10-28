"""Three-valued (Kleene) semantics of the verifier combinators.

A check result is True, False or None (unknown). The combinators follow Kleene's
strong three-valued logic and evaluate every branch, so the verdict does not
depend on the order the branches are listed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import pytest

from midojo.types import Environment
from midojo.verifiers import AllOf, AnyOf, Not, VerificationContext, VerificationResult

VALUES = (True, False, None)


@dataclass
class _Fixed:
    """A leaf with a fixed verdict that counts how often it is assessed."""

    passed: bool | None
    name: str = "leaf"
    calls: int = 0

    def assess(self, ctx: VerificationContext) -> VerificationResult:
        self.calls += 1
        if self.passed is None:
            return VerificationResult.unknown(self.name, "evidence missing")
        return VerificationResult(self.passed, self.name)

    def evaluate(self, ctx: VerificationContext) -> bool | None:
        return self.assess(ctx).passed


def _ctx() -> VerificationContext:
    return VerificationContext(agent_output="", pre_environment=Environment(), post_environment=Environment())


def _kleene_and(a: bool | None, b: bool | None) -> bool | None:
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return True


def _kleene_or(a: bool | None, b: bool | None) -> bool | None:
    if a is True or b is True:
        return True
    if a is None or b is None:
        return None
    return False


@pytest.mark.parametrize(("value", "expected"), [(True, False), (False, True), (None, None)])
def test_not_truth_table(value, expected):
    assert Not(_Fixed(value)).assess(_ctx()).passed is expected


@pytest.mark.parametrize(("a", "b"), list(product(VALUES, repeat=2)))
def test_all_of_truth_table(a, b):
    assert AllOf([_Fixed(a), _Fixed(b)]).assess(_ctx()).passed is _kleene_and(a, b)


@pytest.mark.parametrize(("a", "b"), list(product(VALUES, repeat=2)))
def test_any_of_truth_table(a, b):
    assert AnyOf([_Fixed(a), _Fixed(b)]).assess(_ctx()).passed is _kleene_or(a, b)


def test_empty_combinators_keep_their_identity():
    assert AllOf([]).assess(_ctx()).passed is True
    assert AnyOf([]).assess(_ctx()).passed is False


@pytest.mark.parametrize("combinator", [AllOf, AnyOf])
@pytest.mark.parametrize("values", list(product(VALUES, repeat=3)))
def test_verdict_and_unknowns_do_not_depend_on_branch_order(combinator, values):
    results = set()
    for order in permutations(values):
        result = combinator([_Fixed(v, name=f"leaf-{v}") for v in order]).assess(_ctx())
        results.add((result.passed, frozenset(u.check for u in result.unknowns)))
    assert len(results) == 1


@pytest.mark.parametrize("combinator", [AllOf, AnyOf])
def test_every_branch_is_assessed(combinator):
    # No short-circuit: a decisive first branch still lets the rest be assessed,
    # so an unknown later branch is always seen.
    leaves = [_Fixed(combinator is AnyOf), _Fixed(None), _Fixed(not (combinator is AnyOf))]
    combinator(leaves).assess(_ctx())
    assert [leaf.calls for leaf in leaves] == [1, 1, 1]


def test_unknown_result_carries_the_unknown_leaves():
    result = AllOf([_Fixed(True), _Fixed(None, name="a"), Not(_Fixed(None, name="b"))]).assess(_ctx())
    assert result.passed is None
    assert [(u.check, u.reason) for u in result.unknowns] == [("a", "evidence missing"), ("b", "evidence missing")]
    assert "unknown" in result.reason


def test_decided_result_drops_unknowns():
    # A False conjunct decides AllOf whatever the unknown branch would have been.
    result = AllOf([_Fixed(None, name="a"), _Fixed(False, name="decisive")]).assess(_ctx())
    assert result.passed is False
    assert result.unknowns == ()
    assert result.reason == "decisive"
    # A True branch decides AnyOf.
    result = AnyOf([_Fixed(None, name="a"), _Fixed(True, name="fired")]).assess(_ctx())
    assert result.passed is True
    assert result.unknowns == ()
    assert result.reason == "fired"


def test_not_of_unknown_keeps_the_unknown_leaf():
    result = Not(_Fixed(None, name="a")).assess(_ctx())
    assert result.passed is None
    assert [u.check for u in result.unknowns] == ["a"]


def _kleene_not(a: bool | None) -> bool | None:
    return None if a is None else not a


# Nested shapes, each with the reference Kleene evaluation over three leaves.
NESTED = [
    (lambda a, b, c: Not(AnyOf([a, b, c])), lambda a, b, c: _kleene_not(_kleene_or(_kleene_or(a, b), c))),
    (lambda a, b, c: Not(AllOf([a, b, c])), lambda a, b, c: _kleene_not(_kleene_and(_kleene_and(a, b), c))),
    (lambda a, b, c: AllOf([AnyOf([a, b]), Not(c)]), lambda a, b, c: _kleene_and(_kleene_or(a, b), _kleene_not(c))),
    (lambda a, b, c: AnyOf([AllOf([a, b]), Not(c)]), lambda a, b, c: _kleene_or(_kleene_and(a, b), _kleene_not(c))),
    (lambda a, b, c: Not(Not(AnyOf([a, Not(b), c]))), lambda a, b, c: _kleene_or(_kleene_or(a, _kleene_not(b)), c)),
]


@pytest.mark.parametrize("shape", range(len(NESTED)))
@pytest.mark.parametrize(("a", "b", "c"), list(product(VALUES, repeat=3)))
def test_nested_combinators_follow_kleene_logic(shape, a, b, c):
    build, reference = NESTED[shape]
    result = build(_Fixed(a, name="a"), _Fixed(b, name="b"), _Fixed(c, name="c")).assess(_ctx())
    assert result.passed is reference(a, b, c)
    # An undecided result names the unknown leaves; a decided one names none.
    assert bool(result.unknowns) is (result.passed is None)


@pytest.mark.parametrize("value", VALUES)
def test_double_negation_returns_the_verdict(value):
    assert Not(Not(_Fixed(value))).assess(_ctx()).passed is value
