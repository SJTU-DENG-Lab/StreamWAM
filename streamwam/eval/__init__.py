"""StreamWAM evaluation utilities (benchmark-neutral).

See :mod:`streamwam.eval.policy` for the shared closed-loop policy wrapper.
LIBERO keeps its own ``examples/libero/rollout.py``; new benchmarks build a
thin adapter on top of :class:`streamwam.eval.policy.StreamWAMPolicy`.
"""

from streamwam.eval.policy import StreamWAMPolicy

__all__ = ["StreamWAMPolicy"]
