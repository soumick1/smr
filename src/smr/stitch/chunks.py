"""Keyframe selection and chunk windows for the long-sequence experiment."""
from __future__ import annotations

import numpy as np


def keyframe_indices(n, stride, max_frames=0):
    """Every `stride`-th frame of a sequence of n frames (optionally only
    the first max_frames of them)."""
    idx = list(range(0, n, max(1, int(stride))))
    if max_frames:
        idx = idx[:max_frames]
    return idx


def make_chunks(n, size, overlap):
    """Overlapping index windows covering [0, n).

    A runt window of fewer than 3 frames is folded into the previous one:
    Sim(3) alignment needs three correspondences and a two-frame chunk is
    not a measurement of anything.
    """
    assert 0 <= overlap < size, "overlap must be smaller than the window"
    stride = size - overlap
    chunks, start = [], 0
    while start < n:
        idx = list(range(start, min(start + size, n)))
        if len(idx) < 3 and chunks:
            chunks[-1] = sorted(set(chunks[-1] + idx))
            break
        chunks.append(idx)
        if start + size >= n:
            break
        start += stride
    return chunks


def primary_chunk(chunks, n):
    """chunk id of each frame: the FIRST chunk that contains it (overlap
    frames belong to the earlier chunk, matching the stitchers' setdefault
    behaviour)."""
    owner = np.full(n, -1, int)
    for k, idx in enumerate(chunks):
        for i in idx:
            if owner[i] < 0:
                owner[i] = k
    return owner


def regauge(poses):
    """Express poses in the pass's own frame with its own scale gauge.

    Every backbone reports relative to its first view and normalises scale
    per call; a simulated pass must do the same or the experiment is
    measuring nothing.
    """
    poses = np.asarray(poses, float).copy()
    T0inv = np.linalg.inv(poses[0])
    poses = np.einsum("ij,kjl->kil", T0inv, poses)
    scale = float(np.median(np.linalg.norm(poses[1:, :3, 3], axis=1))) \
        if len(poses) > 1 else 1.0
    if scale > 1e-9:
        poses[:, :3, 3] /= scale
    return poses


def make_session_chunks(sessions, size, overlap):
    """Chunks that never straddle a session boundary.

    `sessions` is the session id of each keyframe (in order).  A chunk that
    mixes the end of one session with the start of the next contains two
    disjoint places, its pass geometry is garbage, and the chain breaks
    there (7-Scenes office x6: chained ATE 0.55 m for VGGT-Omega with every
    chunk individually fine).  Each session is chunked on its own; the
    first chunk of a session shares no frame with memory and must be
    placed by relocalisation (AnchoredStitcher handles that case).
    """
    sessions = list(sessions)
    chunks, start = [], 0
    for i in range(1, len(sessions) + 1):
        if i == len(sessions) or sessions[i] != sessions[start]:
            for c in make_chunks(i - start, size, overlap):
                chunks.append([start + j for j in c])
            start = i
    return chunks
