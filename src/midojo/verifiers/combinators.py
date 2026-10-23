"""Three-valued boolean combinators over predicates.

These are concrete :class:`~midojo.verifiers.Predicate` implementations that
compose other predicates — the logical glue (``all_of``, ``any_of``, ``not``)
that suites use to build richer checks. They live alongside the built-in
predicates (:mod:`midojo.verifiers.builtin`) rather than in the framework core,
and are re-exported from :mod:`midojo.verifiers` for convenience.

A branch's verdict is True, False or None (unknown), combined with Kleene's
strong three-valued logic:

- ``Not``: None stays None.
- ``AllOf``: False if any branch is False, else None if any branch is None, else True.
- ``AnyOf``: True if any branch is True, else None if any branch is None, else False.

Every branch is assessed (no short-circuit), so the verdict and the unknown
branches it reports don't depend on the order the branches are listed in.

Each combinator produces a :class:`~midojo.verifiers.VerificationResult` whose
``reason`` names only the criterion that decided the verdict — for an
:class:`AnyOf` the first branch that fired, for an :class:`AllOf` the
conjunction (or its first failing conjunct), for a :class:`Not` the negated
reason. An unknown result names the unknown branches instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from midojo.verifiers import Predicate, VerificationContext, VerificationResult


def _unknown(label: str, results: list[VerificationResult]) -> VerificationResult:
    """The undecided result of a combinator, carrying its unknown branches."""
    undecided = [r for r in results if r.passed is None]
    reason = f"{label} (" + "; ".join(r.reason for r in undecided) + ")"
    return VerificationResult(None, reason, tuple(u for r in undecided for u in r.unknowns))


@dataclass
class AllOf:
    predicates: list[Predicate]

    def assess(self, ctx: VerificationContext) -> VerificationResult:
        results = [p.assess(ctx) for p in self.predicates]
        failed = next((r for r in results if r.passed is False), None)
        if failed is not None:
            return VerificationResult(False, failed.reason)  # a failing conjunct decides, whatever the rest
        if any(r.passed is None for r in results):
            return _unknown("unknown: all of", results)
        return VerificationResult(True, "all of (" + "; ".join(r.reason for r in results) + ")")

    def evaluate(self, ctx: VerificationContext) -> bool | None:
        return self.assess(ctx).passed


@dataclass
class AnyOf:
    predicates: list[Predicate]

    def assess(self, ctx: VerificationContext) -> VerificationResult:
        results = [p.assess(ctx) for p in self.predicates]
        fired = next((r for r in results if r.passed is True), None)
        if fired is not None:
            return VerificationResult(True, fired.reason)  # a branch that fired decides, whatever the rest
        if any(r.passed is None for r in results):
            return _unknown("unknown: any of", results)
        return VerificationResult(False, "any of (" + "; ".join(r.reason for r in results) + ")")

    def evaluate(self, ctx: VerificationContext) -> bool | None:
        return self.assess(ctx).passed


@dataclass
class Not:
    predicate: Predicate

    def assess(self, ctx: VerificationContext) -> VerificationResult:
        r = self.predicate.assess(ctx)
        if r.passed is None:
            return VerificationResult(None, f"not ({r.reason})", r.unknowns)
        return VerificationResult(not r.passed, f"not ({r.reason})")

    def evaluate(self, ctx: VerificationContext) -> bool | None:
        return self.assess(ctx).passed
