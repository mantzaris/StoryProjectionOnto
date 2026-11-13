"""Public offline benchmark API.

The implementation lives in :mod:`synthetic_benchmark`; model-serving code
must import :mod:`benchmark_runtime` directly so it cannot import scorer gold.
"""

from story_projection_onto.synthetic_benchmark import *  # noqa: F403
