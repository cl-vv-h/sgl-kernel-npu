"""Independent FP64 oracle and error gates for experimental AIR Top-P.

Validation deliberately distinguishes cutoff semantics from V1 compatibility.
The boundary tolerance is a probability-mass tolerance, not permission to drop
arbitrary tokens. Every selected row must still be a complete threshold set.
"""

import numpy as np


def reference(probs, ps):
    output = np.zeros_like(probs)
    for row, p in enumerate(np.broadcast_to(ps, (len(probs),))):
        values = probs[row].astype(np.float64)
        total = values.sum()
        if total == 0:
            continue
        ordered = np.sort(values)[::-1]
        index = min(np.searchsorted(np.cumsum(ordered), p * total), len(values) - 1)
        kept = values if p >= 1 else np.where(values >= ordered[index], values, 0)
        output[row] = kept / max(kept.sum(), 1e-20)
    return output


def validate(probs, ps, actual, *, mass_rtol=2e-6, tv_limit=2e-5):
    """Fail on malformed outputs, broken ties or excessive cutoff/norm error.

    FP32 radix reductions can choose an adjacent boundary value relative to FP64.
    Require <=2e-6 relative row mass error AND <=2e-5 total variation. Report even
    accepted support differences. These are experimental gates, not proof of
    model-level equivalence or acceptable acceptance-length changes.
    """
    assert actual.shape == probs.shape
    assert np.isfinite(actual).all() and (actual >= 0).all()
    expected = reference(probs, ps)
    support_diffs = np.count_nonzero((actual > 0) != (expected > 0), axis=1)
    max_under = max_over = 0.0
    for row, p in enumerate(np.broadcast_to(ps, (len(probs),))):
        values = probs[row].astype(np.float64)
        out = actual[row].astype(np.float64)
        total = values.sum()
        if total == 0:
            assert not np.any(out)
            continue
        support = out > 0
        assert support.any(), "Nonzero input produced an empty nucleus"
        cutoff = values[support].min()
        np.testing.assert_array_equal(support, (values >= cutoff) & (values > 0))
        kept_mass = values[support].sum()
        strictly_above = values[values > cutoff].sum()
        target = float(p) * total
        under = max(target - kept_mass, 0.0) / total
        over = max(strictly_above - target, 0.0) / total
        max_under, max_over = max(max_under, under), max(max_over, over)
        if p >= 1:
            np.testing.assert_array_equal(support, values > 0)
        else:
            assert under <= mass_rtol, ("Insufficient retained mass", row, under)
            assert over <= mass_rtol, ("Cutoff too low", row, over)
        normalized = np.where(support, values, 0) / max(kept_mass, 1e-20)
        np.testing.assert_allclose(out, normalized, rtol=3e-6, atol=1e-12)
    tv = 0.5 * np.abs(actual.astype(np.float64) - expected).sum(axis=1)
    assert tv.max(initial=0) <= tv_limit, ("Excessive distribution error", tv.max())
    return {
        "support_diff_tokens": int(support_diffs.sum()),
        "max_support_diff_per_row": int(support_diffs.max(initial=0)),
        "max_tv_vs_fp64_air": float(tv.max(initial=0)),
        "max_relative_missing_mass": max_under,
        "max_relative_excess_strict_mass": max_over,
    }
