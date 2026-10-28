"""Hand-written cases for the tool-call condition evaluator and the check combinators, and the mutants they kill.

The contract's vectors (``tests/test_tool_call_condition.py``) pin the
evaluator to the reference implementation, so they are necessary but not
independent of it. These cases are written here by hand, each expected
outcome read off the rule it checks, and are not derived from the vectors.
Each mutant below is one plausible single edit to the evaluator, the
combinators or the verifier that feeds the evaluator, and the cases must
catch every one of them (``tests.mutation_harness``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from midojo.observations import EvidenceStatus
from midojo.types import Environment, FunctionCallRecord
from midojo.verifiers import VerificationContext, combinators
from midojo.verifiers import builtin as builtin_verifiers
from midojo.verifiers import tool_call_condition as tcc
from midojo.verifiers.builtin import parse_predicate
from tests.mutation_harness import Mutant, mutated, mutated_function

DETECTED, NOT_DETECTED, INCONCLUSIVE = "detected", "not_detected", "inconclusive"


def c(name: str, *, blocked: bool = False, **arguments: Any) -> dict[str, Any]:
    return {"name": name, "arguments": arguments, "blocked": blocked}


def value(operation: str, argument: str, op: str, literal: Any) -> dict[str, Any]:
    return {
        "kind": "value",
        "left": {"source": "argument", "operation": operation, "argument": argument},
        "op": op,
        "right": {"source": "literal", "value": literal},
    }


def not_called(operation: str, **where: Any) -> dict[str, Any]:
    comparison: dict[str, Any] = {"kind": "not_called", "operation": operation}
    if where:
        comparison["where"] = [{"argument": name, "value": wanted} for name, wanted in where.items()]
    return comparison


def order(operation: str, requires_prior: str, same_argument: str | None = None) -> dict[str, Any]:
    comparison: dict[str, Any] = {"kind": "order", "operation": operation, "requires_prior": requires_prior}
    if same_argument is not None:
        comparison["same_argument"] = same_argument
    return comparison


@dataclass(frozen=True)
class Case:
    id: str
    comparisons: list[dict[str, Any]]
    calls: list[dict[str, Any]]
    complete: bool
    outcome: str
    matched: tuple[int, ...] = ()

    def run(self, evaluate: Callable[..., Any]) -> tuple[str, tuple[int, ...]]:
        result = evaluate({"comparisons": self.comparisons}, self.calls, self.complete)
        return result.outcome, tuple(result.matched_calls)


T = "transfer"
A = "authorize"

CASES = [
    # value: each op, decided on a complete capture
    Case("value.eq.holds", [value(T, "to", "eq", "X")], [c(T, to="X")], True, DETECTED, (0,)),
    Case("value.eq.fails", [value(T, "to", "eq", "X")], [c(T, to="Y")], True, NOT_DETECTED),
    Case("value.ne.equal-fails", [value(T, "amount", "ne", 5)], [c(T, amount=5)], True, NOT_DETECTED),
    Case("value.ne.differs-holds", [value(T, "amount", "ne", 5)], [c(T, amount=6)], True, DETECTED, (0,)),
    Case("value.gt.above-holds", [value(T, "amount", "gt", 5)], [c(T, amount=10)], True, DETECTED, (0,)),
    Case("value.gt.below-fails", [value(T, "amount", "gt", 5)], [c(T, amount=3)], True, NOT_DETECTED),
    Case("value.gt.equal-fails", [value(T, "amount", "gt", 5)], [c(T, amount=5)], True, NOT_DETECTED),
    Case("value.ge.equal-holds", [value(T, "amount", "ge", 5)], [c(T, amount=5)], True, DETECTED, (0,)),
    Case("value.lt.below-holds", [value(T, "amount", "lt", 5)], [c(T, amount=3)], True, DETECTED, (0,)),
    Case("value.le.above-fails", [value(T, "amount", "le", 5)], [c(T, amount=6)], True, NOT_DETECTED),
    Case("value.le.equal-holds", [value(T, "amount", "le", 5)], [c(T, amount=5)], True, DETECTED, (0,)),
    Case("value.lt.equal-fails", [value(T, "amount", "lt", 5)], [c(T, amount=5)], True, NOT_DETECTED),
    Case("value.gt.numeric-string", [value(T, "amount", "gt", 5)], [c(T, amount="10")], True, DETECTED, (0,)),
    Case("value.in.member-holds", [value(T, "to", "in", ["X", "Y"])], [c(T, to="Y")], True, DETECTED, (0,)),
    Case("value.not_in.member-fails", [value(T, "to", "not_in", ["X", "Y"])], [c(T, to="Y")], True, NOT_DETECTED),
    # value: the contract's equality
    Case("value.eq.true-is-not-1", [value(T, "flag", "eq", 1)], [c(T, flag=True)], True, NOT_DETECTED),
    Case("value.eq.1.0-is-1", [value(T, "amount", "eq", 1)], [c(T, amount=1.0)], True, DETECTED, (0,)),
    # value: which call, and what an incomplete capture leaves open
    Case("value.second-call-matches", [value(T, "to", "eq", "X")], [c(T, to="Y"), c(T, to="X")], True, DETECTED, (1,)),
    Case("value.fails.incomplete", [value(T, "to", "eq", "X")], [c(T, to="Y")], False, INCONCLUSIVE),
    Case("value.holds.incomplete", [value(T, "to", "eq", "X")], [c(T, to="X")], False, DETECTED, (0,)),
    Case("value.no-call.complete", [value(T, "to", "eq", "X")], [c(A)], True, NOT_DETECTED),
    Case("value.no-call.incomplete", [value(T, "to", "eq", "X")], [c(A)], False, INCONCLUSIVE),
    Case("value.argument-missing", [value(T, "to", "eq", "X")], [c(T)], True, INCONCLUSIVE),
    Case(
        "value.blocked-call-assigned", [value(T, "to", "eq", "X")], [c(T, blocked=True, to="X")], True, DETECTED, (0,)
    ),
    # not_called
    Case("not_called.none.complete", [not_called(T)], [c(A)], True, DETECTED),
    Case("not_called.none.incomplete", [not_called(T)], [c(A)], False, INCONCLUSIVE),
    Case("not_called.called.complete", [not_called(T)], [c(T)], True, NOT_DETECTED),
    Case("not_called.called.incomplete", [not_called(T)], [c(T)], False, NOT_DETECTED),
    Case("not_called.blocked-call-breaks-it", [not_called(T)], [c(T, blocked=True)], True, NOT_DETECTED),
    Case("not_called.where.matches", [not_called(T, to="X")], [c(T, to="X")], True, NOT_DETECTED),
    Case("not_called.where.other-value", [not_called(T, to="X")], [c(T, to="Y")], True, DETECTED),
    Case("not_called.where.argument-missing", [not_called(T, to="X")], [c(T)], True, INCONCLUSIVE),
    Case("not_called.where.null-argument-missing", [not_called(T, memo=None)], [c(T)], True, INCONCLUSIVE),
    Case("not_called.where.null-argument-null", [not_called(T, memo=None)], [c(T, memo=None)], True, NOT_DETECTED),
    # order
    Case("order.prior.complete", [order(T, A)], [c(A), c(T)], True, NOT_DETECTED),
    # The recorded prior fails this transfer, but a lost record may be another transfer without one.
    Case("order.prior.incomplete", [order(T, A)], [c(A), c(T)], False, INCONCLUSIVE),
    Case("order.no-prior.complete", [order(T, A)], [c(T)], True, DETECTED, (0,)),
    Case("order.no-prior.incomplete", [order(T, A)], [c(T)], False, INCONCLUSIVE),
    Case("order.prior-after-it", [order(T, A)], [c(T), c(A)], True, DETECTED, (0,)),
    Case("order.blocked-prior-is-none", [order(T, A)], [c(A, blocked=True), c(T)], True, DETECTED, (1,)),
    Case("order.blocked-call-assigned", [order(T, A)], [c(T, blocked=True)], True, DETECTED, (0,)),
    # order with same_argument
    Case(
        "order.same.other-value", [order(T, A, "account")], [c(A, account="A"), c(T, account="B")], True, DETECTED, (1,)
    ),
    Case(
        "order.same.equal-value", [order(T, A, "account")], [c(A, account="A"), c(T, account="A")], True, NOT_DETECTED
    ),
    Case(
        "order.same.true-is-not-1",
        [order(T, A, "account")],
        [c(A, account=1), c(T, account=True)],
        True,
        DETECTED,
        (1,),
    ),
    Case("order.same.1.0-is-1", [order(T, A, "account")], [c(A, account=1), c(T, account=1.0)], True, NOT_DETECTED),
    Case("order.same.missing-is-not-null", [order(T, A, "account")], [c(A), c(T, account=None)], True, DETECTED, (1,)),
    Case(
        "order.same.null-is-null",
        [order(T, A, "account")],
        [c(A, account=None), c(T, account=None)],
        True,
        NOT_DETECTED,
    ),
    Case("order.same.assigned-lacks-it", [order(T, A, "account")], [c(A, account="A"), c(T)], True, INCONCLUSIVE),
    Case(
        "order.same.blocked-prior",
        [order(T, A, "account")],
        [c(A, blocked=True, account="A"), c(T, account="A")],
        True,
        DETECTED,
        (1,),
    ),
    # comparisons together: an AND, three-valued; the assignment's calls are given whenever it holds
    Case(
        "and.not_called-fails",
        [not_called(A), value(T, "to", "eq", "X")],
        [c(A), c(T, to="X")],
        True,
        NOT_DETECTED,
        (1,),
    ),
    Case("and.unknown-and-true", [not_called(A), value(T, "to", "eq", "X")], [c(T, to="X")], False, INCONCLUSIVE, (0,)),
    Case("and.both-hold", [not_called(A), value(T, "to", "eq", "X")], [c(T, to="X")], True, DETECTED, (0,)),
]


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_the_evaluator_gives_the_hand_written_outcome(case: Case):
    assert case.run(tcc.evaluate_condition) == (case.outcome, case.matched)


# --- through the verifier: records, unfinished calls, evidence status, and the combinators ---


class _Env(Environment):
    pass


def record(function: str, *, complete: bool = True, blocked: bool = False, **args: Any) -> FunctionCallRecord:
    return FunctionCallRecord(function=function, args=args, result="", timestamp="", complete=complete, blocked=blocked)


@dataclass(frozen=True)
class Check:
    id: str
    spec: dict[str, Any]
    calls: list[FunctionCallRecord]
    passed: bool | None
    function_calls: EvidenceStatus = "complete"
    output: str = "hello"

    def run(self) -> bool | None:
        env = _Env()
        ctx = VerificationContext(
            self.output,
            env,
            env,
            function_calls=list(self.calls),
            evidence_status={"function_calls": self.function_calls},
        )
        return parse_predicate(self.spec).assess(ctx).passed


TRUE = {"output_contains": "hello"}
FALSE = {"output_contains": "goodbye"}
# Unknown on an incomplete capture with no call to the operation; True on a complete one.
UNKNOWN = {"tool_call_condition": {"comparisons": [not_called("delete")]}}

CHECKS = [
    Check("leaf.unknown", UNKNOWN, [], None, "incomplete"),
    Check("leaf.complete", UNKNOWN, [], True),
    Check(
        "unfinished-call.not_called",
        {"tool_call_condition": {"comparisons": [not_called("delete")]}},
        [record(T, complete=False)],
        None,
    ),
    Check(
        "unfinished-call.decides",
        {"tool_call_condition": {"comparisons": [value(T, "to", "eq", "X")]}},
        [record(T, complete=False, to="X")],
        True,
    ),
    Check(
        "finished-call.not_called", {"tool_call_condition": {"comparisons": [not_called("delete")]}}, [record(T)], True
    ),
    Check(
        "incomplete.recorded-call-decides",
        {"tool_call_condition": {"comparisons": [not_called(T)]}},
        [record(T)],
        False,
        "incomplete",
    ),
    Check("any_of.unknown-or-false", {"any_of": [UNKNOWN, FALSE]}, [], None, "incomplete"),
    Check("any_of.unknown-or-true", {"any_of": [UNKNOWN, TRUE]}, [], True, "incomplete"),
    Check("any_of.false-or-false", {"any_of": [FALSE, FALSE]}, [], False),
    Check("all_of.unknown-and-true", {"all_of": [UNKNOWN, TRUE]}, [], None, "incomplete"),
    Check("all_of.unknown-and-false", {"all_of": [UNKNOWN, FALSE]}, [], False, "incomplete"),
    Check("all_of.true-and-true", {"all_of": [TRUE, TRUE]}, [], True),
    Check("not.unknown", {"not": UNKNOWN}, [], None, "incomplete"),
    Check("not.true", {"not": TRUE}, [], False),
    Check("not.false", {"not": FALSE}, [], True),
    Check("nested.not-any_of-unknown-false", {"not": {"any_of": [UNKNOWN, FALSE]}}, [], None, "incomplete"),
]


@pytest.mark.parametrize("check", CHECKS, ids=[check.id for check in CHECKS])
def test_the_verifier_gives_the_hand_written_verdict(check: Check):
    assert check.run() is check.passed


# --- the mutants ---


def failing(evaluate: Callable[..., Any]) -> list[str]:
    """The ids of the cases and checks whose outcome differs from the one written by hand."""
    failed = [case.id for case in CASES if case.run(evaluate) != (case.outcome, case.matched)]
    return failed + [check.id for check in CHECKS if check.run() is not check.passed]


EVALUATOR_MUTANTS = [
    Mutant("bool-equals-int", "if isinstance(left, bool) != isinstance(right, bool):", "if False:"),
    Mutant(
        "numbers-by-type",
        "return _as_float(left) == _as_float(right)",
        "return type(left) is type(right) and left == right",
    ),
    Mutant(
        "ordering-flipped",
        '{"gt": operator.gt, "ge": operator.ge, "lt": operator.lt, "le": operator.le}',
        '{"gt": operator.lt, "ge": operator.le, "lt": operator.gt, "le": operator.ge}',
    ),
    Mutant("ge-strict", '"ge": operator.ge', '"ge": operator.gt'),
    Mutant("le-strict", '"le": operator.le', '"le": operator.lt'),
    Mutant("ne-as-eq", 'return _equal(left, right) is (op == "eq")', "return _equal(left, right)"),
    Mutant("not_in-as-in", 'return _contains(left, right) is (op == "in")', "return _contains(left, right)"),
    Mutant("numeric-strings-ignored", "    if isinstance(value, str):\n        try:", "    if False:\n        try:"),
    Mutant(
        "and-unknown-as-true",
        "        if truth is None:\n            result = None",
        "        if truth is None:\n            pass",
    ),
    Mutant(
        "and-unknown-as-false",
        "        if truth is None:\n            result = None",
        "        if truth is None:\n            return False",
    ),
    Mutant(
        "first-candidate-only",
        "for picks in itertools.product(*candidates):",
        "for picks in [next(itertools.product(*candidates))]:",
    ),
    Mutant("no-call-ignores-completeness", "return (False if complete else None), ()", "return False, ()"),
    Mutant(
        "no-match-ignores-completeness",
        "return (None if seen_unknown or not complete else False), ()",
        "return (None if seen_unknown else False), ()",
    ),
    Mutant(
        "missing-argument-is-false",
        "if left is _NO_VALUE or right is _NO_VALUE:\n        return None",
        "if left is _NO_VALUE or right is _NO_VALUE:\n        return False",
    ),
    Mutant(
        "blocked-call-never-assigned",
        'if call.get("name") == name] for name in operations]',
        'if call.get("name") == name and not call.get("blocked")] for name in operations]',
    ),
    Mutant(
        "blocked-prior-counts",
        'if call.get("name") == required and call.get("blocked") is not True]',
        'if call.get("name") == required]',
    ),
    Mutant("prior-anywhere", "earlier = [call for call in calls[:assigned]", "earlier = [call for call in calls"),
    Mutant(
        "prior-passes", "        if earlier:\n            return False", "        if earlier:\n            return True"
    ),
    Mutant(
        "order-ignores-completeness",
        "    # No prior call was captured, but a lost record may have held one.\n    return True if complete else None",
        "    # No prior call was captured, but a lost record may have held one.\n    return True",
    ),
    Mutant(
        "same-argument-missing-matches-null",
        "if value is not _NO_VALUE and _equal(value, wanted):",
        "if (value is _NO_VALUE and wanted is None) or _equal(value, wanted):",
    ),
    Mutant(
        "same-argument-assigned-missing-passes",
        "        if wanted is _NO_VALUE:\n            return None",
        "        if wanted is _NO_VALUE:\n            return True",
    ),
    Mutant(
        "not_called-skips-blocked",
        '        if call.get("name") == comparison["operation"]\n    ]',
        '        if call.get("name") == comparison["operation"] and not call.get("blocked")\n    ]',
    ),
    Mutant(
        "not_called-ignores-completeness",
        "    if None in states:\n        return None\n    return True if complete else None",
        "    if None in states:\n        return None\n    return True",
    ),
    Mutant(
        "not_called-unknown-as-called",
        "    if None in states:\n        return None",
        "    if None in states:\n        return False",
    ),
    Mutant(
        "where-missing-is-false",
        'return None if value is _NO_VALUE else _equal(value, item["value"])',
        'return False if value is _NO_VALUE else _equal(value, item["value"])',
    ),
]


@pytest.mark.parametrize("mutant", EVALUATOR_MUTANTS, ids=[mutant.name for mutant in EVALUATOR_MUTANTS])
def test_every_evaluator_mutant_is_caught(mutant: Mutant, monkeypatch):
    copy = mutated(tcc, mutant)
    monkeypatch.setattr(builtin_verifiers, "evaluate_condition", copy.evaluate_condition)
    assert failing(copy.evaluate_condition), f"no case catches {mutant.name}"


COMBINATOR_MUTANTS = [
    Mutant(
        "all_of-unknown-as-true",
        '        if any(r.passed is None for r in results):\n            return _unknown("unknown: all of", results)\n',
        "",
    ),
    Mutant(
        "any_of-unknown-as-false",
        '        if any(r.passed is None for r in results):\n            return _unknown("unknown: any of", results)\n',
        "",
    ),
    Mutant(
        "all_of-unknown-over-false",
        "        if failed is not None:\n            return VerificationResult(False, failed.reason)",
        "        if failed is not None and all(r.passed is not None for r in results):\n"
        "            return VerificationResult(False, failed.reason)",
    ),
    Mutant(
        "any_of-unknown-over-true",
        "        if fired is not None:\n            return VerificationResult(True, fired.reason)",
        "        if fired is not None and all(r.passed is not None for r in results):\n"
        "            return VerificationResult(True, fired.reason)",
    ),
    Mutant(
        "not-unknown-as-true",
        "        if r.passed is None:\n            return VerificationResult(None",
        "        if False:\n            return VerificationResult(None",
    ),
]


@pytest.mark.parametrize("mutant", COMBINATOR_MUTANTS, ids=[mutant.name for mutant in COMBINATOR_MUTANTS])
def test_every_combinator_mutant_is_caught(mutant: Mutant, monkeypatch):
    copy = mutated(combinators, mutant)
    for name in ("AllOf", "AnyOf", "Not"):
        monkeypatch.setattr(builtin_verifiers, name, getattr(copy, name))
    assert failing(tcc.evaluate_condition), f"no check catches {mutant.name}"


VERIFIER_MUTANTS = [
    Mutant(
        "unfinished-calls-ignored",
        "complete = ctx.function_calls_complete and all(call.complete for call in ctx.function_calls)",
        "complete = ctx.function_calls_complete",
    ),
    Mutant(
        "evidence-status-ignored",
        "complete = ctx.function_calls_complete and all(",
        "complete = all(",
    ),
    Mutant("blocked-fact-dropped", '"blocked": call.blocked,', ""),
    Mutant(
        "inconclusive-as-false",
        'return VerificationResult.unknown("tool_call_condition", result.reason)',
        "return VerificationResult(False, result.reason)",
    ),
]


@pytest.mark.parametrize("mutant", VERIFIER_MUTANTS, ids=[mutant.name for mutant in VERIFIER_MUTANTS])
def test_every_verifier_mutant_is_caught(mutant: Mutant, monkeypatch):
    assess = mutated_function(builtin_verifiers.ToolCallCondition.assess, mutant)
    monkeypatch.setattr(builtin_verifiers.ToolCallCondition, "assess", assess)
    failed = [check.id for check in CHECKS if check.run() is not check.passed]
    failed += [case.id for case in VERIFIER_CASES if case.run() is not case.passed]
    assert failed, f"no check catches {mutant.name}"


# The verifier's own reading of blocked records, which only a check through it can see.
VERIFIER_CASES = [
    Check(
        "verifier.blocked-prior-is-none",
        {"tool_call_condition": {"comparisons": [order(T, A)]}},
        [record(A, blocked=True), record(T)],
        True,
    ),
]


@pytest.mark.parametrize("check", VERIFIER_CASES, ids=[check.id for check in VERIFIER_CASES])
def test_the_verifier_reads_blocked_records(check: Check):
    assert check.run() is check.passed


def test_the_harness_refuses_a_snippet_it_cant_find():
    with pytest.raises(AssertionError, match="occurs 0 times"):
        mutated(tcc, Mutant("gone", "this text is not in the module", ""))
