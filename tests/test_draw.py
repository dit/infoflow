"""
Drawing: geometry, the decomposition glyphs, every view, and both backends.
"""

import numpy as np
import pytest

from infoflow import datasets
from infoflow.draw import _arrow, _bezier, _trim, draw, node_positions, save, scene
from infoflow.network import infer_multiplex
from infoflow.selection import SkeletonSettings

FAST = SkeletonSettings(
    n_perm_max_stat=100, n_perm_min_stat=100, n_perm_omnibus=200, n_perm_max_seq=200, n_perm_pairs=50
)


@pytest.fixture(scope="module")
def xor_net():
    data = datasets.xor_synergy(4000, noise=0.05, seed=1)
    return infer_multiplex(
        data, max_lag=2, skeleton_settings=FAST, n_boot=50, n_null=50, interpret=False, hyperedges=True, prng=0
    )


def test_bezier_trim_and_arrow():
    pts = _bezier((0, 0), (1, 0), 0.2)
    assert np.allclose(pts[0], (0, 0)) and np.allclose(pts[-1], (1, 0))
    assert pts[:, 1].min() < -0.05  # bulges to the right of travel
    trimmed = _trim(pts, 0.1, 0.2)
    length = lambda p: np.linalg.norm(np.diff(p, axis=0), axis=1).sum()  # noqa: E731
    assert length(trimmed) == pytest.approx(length(pts) - 0.3, rel=0.02)
    line, (hx, hy) = _arrow(trimmed)
    assert np.allclose((hx[0], hy[0]), trimmed[-1])
    assert length(line) < length(trimmed)


def test_glyph_widths_encode_the_identities(xor_net):
    sc = scene(xor_net, "glyphs")
    bodies = [p for p in sc.items if getattr(p, "group", "") == "body" and p.hover]
    assert {p.hover["edge"] for p in bodies} == {"x0 -> x2", "x1 -> x2"}
    for p in bodies:
        h = p.hover
        assert h["te"] == pytest.approx(h["intrinsic"] + h["synergistic"])
        assert h["tdmi"] == pytest.approx(h["intrinsic"] + h["shared"])
        assert h["synergistic"] > 0.5  # XOR: the flow is synergy
    widest = max(p.width for p in bodies)
    assert widest == pytest.approx(9.0 + 1, abs=0.01)


def test_positions_are_shared_and_normalized(xor_net):
    pos = node_positions(xor_net)
    xy = np.array(list(pos.values()))
    assert np.abs(xy).max() == pytest.approx(1.0)
    custom = node_positions(xor_net, positions=[(0, 0), (2, 0), (1, 1)])
    assert custom["x0"][0] < custom["x2"][0] < custom["x1"][0]
    with pytest.raises(ValueError, match="layout"):
        node_positions(xor_net, layout="tree")


def test_every_view_builds(xor_net):
    for view in ("glyphs", "arcs", "stacked"):
        sc = scene(xor_net, view, hyperedges=view != "stacked")
        assert sc.items and sc.legend
    hubs = [p for p in scene(xor_net, "glyphs", hyperedges=True).items if getattr(p, "group", "") == "hubs"]
    assert len(hubs) == int(xor_net.hyperedges["significant"].sum())
    with pytest.raises(ValueError, match="hyperedge overlay"):
        scene(xor_net, "stacked", hyperedges=True)
    with pytest.raises(ValueError, match="machines=True"):
        scene(xor_net, "glyphs", machines=True)


def test_matplotlib_backend(xor_net, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    for view in ("glyphs", "arcs", "stacked", "matrices"):
        fig = draw(xor_net, view, backend="matplotlib", hyperedges=view in ("glyphs", "arcs"), common_scale=True)
        assert save(fig, tmp_path / f"{view}.png").stat().st_size > 1000


def test_bokeh_backend(xor_net, tmp_path):
    pytest.importorskip("holoviews")
    pytest.importorskip("bokeh")
    for view in ("glyphs", "arcs", "stacked", "matrices"):
        element = draw(xor_net, view, hyperedges=view in ("glyphs", "arcs"))
        out = tmp_path / f"{view}.html"
        save(element, out)
        assert out.stat().st_size > 1000
