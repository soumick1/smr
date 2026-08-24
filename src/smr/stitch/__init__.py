"""Chunk stitching for long sequences: chained baseline, memory-anchored
streaming (SMR), and batch pose-graph optimisation over identical
measurements."""
from .chunks import keyframe_indices, make_chunks, primary_chunk, regauge   # noqa: F401
from .passes import (BackboneRunner, PassCache, SimulatedRunner,             # noqa: F401
                     array_descriptors, image_descriptors)
from .memory_index import (PERIODS, DescriptorIndex, ScaffoldIndex,          # noqa: F401
                           make_index, scaffold_decode)
from .anchored import AnchoredStitcher, stitch_chained                       # noqa: F401
from . import posegraph, probe, sim3                                         # noqa: F401
