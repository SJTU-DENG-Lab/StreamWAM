"""Streaming-WAM evaluation utilities (benchmark-neutral).

See :mod:`streamingwam.eval.policy` for the shared closed-loop policy wrapper.
LIBERO keeps its own ``examples/libero/rollout.py``; new benchmarks build a
thin adapter on top of :class:`streamingwam.eval.policy.StreamingWAMPolicy`.
"""

from streamingwam.eval.policy import StreamingWAMPolicy

__all__ = ["StreamingWAMPolicy"]
