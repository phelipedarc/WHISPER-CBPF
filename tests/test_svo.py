"""Band resolution + SVO Filter Profile Service fallback (all network calls mocked)."""
import warnings

import numpy as np
import pytest

from whisper_cbpf.io import bands, svo
from whisper_cbpf.io.svo import SvoUnavailable


@pytest.fixture
def svo_clean(tmp_path, monkeypatch):
    """Isolate the SVO cache to a temp file and reset in-memory cache + manual overrides."""
    monkeypatch.setenv("WHISPER_SVO_CACHE", str(tmp_path / "svo_cache.json"))
    svo.clear_cache(disk=True)
    svo._MANUAL_BANDS.clear()
    yield
    svo.clear_cache(disk=True)
    svo._MANUAL_BANDS.clear()


def test_filter_lookup_hit_does_not_call_svo(monkeypatch):
    """A band in FILTER_LOOKUP resolves from the LSST table without touching SVO."""
    def boom(*a, **k):
        raise AssertionError("SVO must not be queried for a FILTER_LOOKUP band")
    monkeypatch.setattr(svo, "_svo_fetch_metadata", boom)
    monkeypatch.setattr(svo, "_svo_fetch_index", boom)

    r = bands.resolve_band("g")          # raw survey code -> 'g-band' -> LSST anchor
    assert r["source"] == "lsst"
    assert r["zero_point"] == 3631.0
    assert 4000 < r["lambda_eff"] < 6000

    r2 = bands.resolve_band("Ks")        # -> 'K-band' documented NIR
    assert r2["source"] == "documented" and r2["lambda_eff"] > 20000


def test_absent_band_warns_then_svo_resolves(svo_clean, monkeypatch):
    def fake_meta(filter_id):
        return {"filter_id": filter_id, "WavelengthEff": 6200.0, "ZeroPoint": 3600.0}
    monkeypatch.setattr(svo, "_svo_fetch_metadata", fake_meta)

    with pytest.warns(UserWarning, match="not in FILTER_LOOKUP"):
        r = bands.resolve_band("PAN-STARRS/PS1.w")   # looks like an SVO id; absent from lookup
    assert r["source"] == "svo"
    assert r["lambda_eff"] == 6200.0 and r["zero_point"] == 3600.0


def test_repeat_lookup_hits_cache_not_network(svo_clean, monkeypatch):
    calls = {"n": 0}

    def counting_meta(filter_id):
        calls["n"] += 1
        return {"filter_id": filter_id, "WavelengthEff": 4800.0, "ZeroPoint": 3631.0}
    monkeypatch.setattr(svo, "_svo_fetch_metadata", counting_meta)

    a = svo.get_filter_metadata("PAN-STARRS/PS1.q")
    b = svo.get_filter_metadata("PAN-STARRS/PS1.q")
    assert a == b
    assert calls["n"] == 1   # second lookup served from cache


def test_disk_cache_survives_fresh_memory(svo_clean, monkeypatch):
    calls = {"n": 0}

    def counting_meta(filter_id):
        calls["n"] += 1
        return {"filter_id": filter_id, "WavelengthEff": 4800.0, "ZeroPoint": 3631.0}
    monkeypatch.setattr(svo, "_svo_fetch_metadata", counting_meta)

    svo.get_filter_metadata("PAN-STARRS/PS1.x")     # populates disk
    svo._META_CACHE.clear()                          # wipe memory only
    svo._DISK_LOADED = False
    svo.get_filter_metadata("PAN-STARRS/PS1.x")     # should reload from disk, no new call
    assert calls["n"] == 1


def test_svo_unavailable_degrades_then_manual_override(svo_clean, monkeypatch):
    def raises(filter_id):
        raise SvoUnavailable("network down")
    monkeypatch.setattr(svo, "_svo_fetch_metadata", raises)

    with pytest.warns(UserWarning, match="register_manual_band"):
        r = bands.resolve_band("PAN-STARRS/PS1.z2")   # id-shaped but SVO fails
    assert r["source"] == "unresolved" and r["lambda_eff"] is None

    svo.register_manual_band("PAN-STARRS/PS1.z2", 9000.0, 3631.0)
    r2 = bands.resolve_band("PAN-STARRS/PS1.z2")
    assert r2["source"] == "manual" and r2["lambda_eff"] == 9000.0


def test_unknown_band_no_hint_is_graceful(svo_clean, monkeypatch):
    """A non-id band with no wavelength hint cannot be searched -> graceful unresolved."""
    monkeypatch.setattr(svo, "_svo_fetch_index", lambda *a, **k: [])
    with pytest.warns(UserWarning):
        r = bands.resolve_band("totally_unknown_filter")
    assert r["source"] == "unresolved"


def test_ambiguous_index_warns_and_picks_closest(svo_clean, monkeypatch):
    def fake_index(lo, hi):
        return [
            {"filterID": "A/A.x", "WavelengthEff": 6300.0, "ZeroPoint": 3600.0},
            {"filterID": "B/B.y", "WavelengthEff": 6100.0, "ZeroPoint": 3600.0},
        ]
    monkeypatch.setattr(svo, "_svo_fetch_index", fake_index)
    with pytest.warns(UserWarning, match="ambiguously"):
        fid = svo.find_filter_id("custom", lambda_eff_hint=6150.0)
    assert fid == "B/B.y"   # closest to the hint


@pytest.mark.parametrize("content", [
    "[1, 2, 3]",                              # valid JSON but not a dict
    '{"X/X.r": "garbage"}',                   # dict with a string value
    '{"X/X.r": {"ZeroPoint": 3631.0}}',       # dict missing WavelengthEff
    "{ this is not json",                      # unparseable
])
def test_corrupt_cache_never_crashes_load(svo_clean, monkeypatch, content):
    import os
    svo.clear_cache(disk=True)
    with open(os.environ["WHISPER_SVO_CACHE"], "w") as fh:
        fh.write(content)

    def good_meta(filter_id):
        return {"filter_id": filter_id, "WavelengthEff": 6200.0, "ZeroPoint": 3600.0}
    monkeypatch.setattr(svo, "_svo_fetch_metadata", good_meta)

    # The malformed cached entry must be ignored and re-fetched, never raise ValueError/KeyError.
    r = bands.resolve_band("X/X.r", warn=False)
    assert r["source"] == "svo" and r["lambda_eff"] == 6200.0


def test_manual_band_resolves_without_warning(svo_clean):
    svo.register_manual_band("my_custom", 5000.0, 3631.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # a manual override must not warn about FILTER_LOOKUP
        r = bands.resolve_band("my_custom")
    assert r["source"] == "manual" and r["lambda_eff"] == 5000.0


def test_get_transmission_data_mocked(svo_clean, monkeypatch):
    # svo_clean: curves are cached on disk now, and a fake curve must not land in the real cache.
    monkeypatch.setattr(svo, "_svo_fetch_transmission",
                        lambda fid: (np.array([1000.0, 2000.0]), np.array([0.1, 0.9])))
    wl, tr = svo.get_transmission_data("PAN-STARRS/PS1.r")
    assert np.allclose(wl, [1000.0, 2000.0]) and np.allclose(tr, [0.1, 0.9])


def test_transmission_is_cached_in_memory_and_on_disk(svo_clean, monkeypatch):
    """A model bound to an SVO filter must rebuild offline: the curve is cached like the metadata."""
    calls = {"n": 0}

    def counting(fid):
        calls["n"] += 1
        return np.array([5000.0, 5500.0, 6000.0]), np.array([0.0, 0.8, 0.0])
    monkeypatch.setattr(svo, "_svo_fetch_transmission", counting)
    a = svo.get_transmission_data("X/X.q")
    b = svo.get_transmission_data("X/X.q")                 # memory
    svo._CURVE_CACHE.clear()
    c = svo.get_transmission_data("X/X.q")                 # disk
    assert calls["n"] == 1
    for got in (b, c):
        assert np.array_equal(got[0], a[0]) and np.array_equal(got[1], a[1])
    svo.get_transmission_data("X/X.q", use_cache=False)
    assert calls["n"] == 2


class _FakeFilter:
    """What ``pyphot.svo.get_pyphot_filter`` returns: wavelength as a Quantity, in nm."""

    def __init__(self):
        import astropy.units as u
        self.wavelength = np.array([400.0, 450.0, 500.0]) * u.nm
        self.transmit = np.array([0.0, 0.9, 0.0])


class _FakePyphotSvo:
    QUERY_URL = "http://invalid.example/fps.php"

    @staticmethod
    def get_pyphot_filter(fid):
        return _FakeFilter()


def test_transmission_goes_through_pyphot_when_installed(monkeypatch):
    """Role A of the pyphot trial: pyphot first, and its nm curve comes back in Angstrom."""
    def no_astroquery():
        raise AssertionError("astroquery must not be used when pyphot answers")
    monkeypatch.setattr(svo, "_pyphot", lambda: _FakePyphotSvo)
    monkeypatch.setattr(svo, "_svo", no_astroquery)
    wl, tr = svo._svo_fetch_transmission("LSST/LSST.g")
    assert np.allclose(wl, [4000.0, 4500.0, 5000.0]) and np.allclose(tr, [0.0, 0.9, 0.0])


def test_astroquery_is_the_fallback_without_pyphot(monkeypatch):
    class FakeSvoFps:
        @staticmethod
        def get_transmission_data(fid):
            return {"Wavelength": [4000.0, 4500.0], "Transmission": [0.2, 0.3]}
    monkeypatch.setattr(svo, "_pyphot", lambda: None)
    monkeypatch.setattr(svo, "_svo", lambda: FakeSvoFps)
    wl, tr = svo._svo_fetch_transmission("LSST/LSST.g")
    assert np.allclose(wl, [4000.0, 4500.0]) and np.allclose(tr, [0.2, 0.3])
    assert svo.transmission_backend() == "astroquery"


def test_pyphot_failure_falls_back_to_astroquery(monkeypatch):
    class Broken:
        QUERY_URL = "http://invalid.example/fps.php"

        @staticmethod
        def get_pyphot_filter(fid):
            raise RuntimeError("pyphot could not parse this")

    class FakeSvoFps:
        @staticmethod
        def get_transmission_data(fid):
            return {"Wavelength": [4000.0, 4500.0], "Transmission": [0.2, 0.3]}
    monkeypatch.setattr(svo, "_pyphot", lambda: Broken)
    monkeypatch.setattr(svo, "_svo", lambda: FakeSvoFps)
    wl, _ = svo._svo_fetch_transmission("LSST/LSST.g")
    assert np.allclose(wl, [4000.0, 4500.0])


def test_boundary_names_exist_on_pyphot():
    """The pyphot names the wrappers call must exist (attribute presence only; no network)."""
    pytest.importorskip("pyphot")
    from pyphot.io.votable import from_votable      # noqa: F401  (the metadata parser)
    pyphot_svo = svo._pyphot()
    assert pyphot_svo is not None
    assert hasattr(pyphot_svo, "get_pyphot_filter") and hasattr(pyphot_svo, "QUERY_URL")


def test_empty_svo_index_with_hint_is_graceful(svo_clean, monkeypatch):
    """A non-id band WITH a wavelength hint reaches _svo_fetch_index; an empty result degrades."""
    monkeypatch.setattr(svo, "_svo_fetch_index", lambda lo, hi: [])
    with pytest.warns(UserWarning):
        r = bands.resolve_band("customband", lambda_eff_hint=6000.0)
    assert r["source"] == "unresolved" and r["lambda_eff"] is None


def test_boundary_methods_exist_on_astroquery():
    """The three network wrappers must name astroquery methods that actually exist.

    Every other test here monkeypatches those wrappers, so a wrong method name is invisible to them
    -- which is how ``get_filter_metadata`` (a method astroquery has never had) survived. Attribute
    presence only; no network.
    """
    pytest.importorskip("astroquery")
    SvoFps = svo._svo()
    for method in ("get_filter_list", "get_filter_index", "get_transmission_data"):
        assert hasattr(SvoFps, method), f"astroquery.svo_fps.SvoFps has no {method}()"


def test_resolve_bands_vectorized(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no SVO for known bands")
    monkeypatch.setattr(svo, "_svo_fetch_metadata", boom)
    lam, zp, info = bands.resolve_bands(np.array(["g", "r", "g"]), svo_fallback=False)
    assert lam.shape == (3,) and zp.shape == (3,)
    assert info["g"]["source"] == "lsst"
    assert np.isfinite(lam).all()


_MINI_VOTABLE = b"""<?xml version="1.0"?>
<VOTABLE version="1.1" xmlns="http://www.ivoa.net/xml/VOTable/v1.1">
<RESOURCE type="results"><TABLE>
<PARAM name="filterID" value="X/X.q" datatype="char" arraysize="*"/>
<PARAM name="WavelengthEff" value="6200.5" unit="Angstrom" datatype="double"/>
<PARAM name="ZeroPoint" value="3100.25" unit="Jy" datatype="double"/>
<FIELD name="Wavelength" unit="Angstrom" datatype="double"/>
<FIELD name="Transmission" datatype="double"/>
<DATA><TABLEDATA><TR><TD>6000.0</TD><TD>0.1</TD></TR><TR><TD>6400.0</TD><TD>0.5</TD></TR>
</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>"""


def test_metadata_goes_through_pyphot_when_installed(monkeypatch):
    """With pyphot, the metadata is SVO's own VOTable PARAMs, read by pyphot's parser from the
    single-filter query -- no astroquery, and the same numbers its filter list carries."""
    pytest.importorskip("pyphot")
    import requests

    class Response:
        content = _MINI_VOTABLE

        def raise_for_status(self):
            pass

    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen.update(url=url, params=params)
        return Response()

    def no_astroquery():
        raise AssertionError("astroquery must not be used when pyphot answers")
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(svo, "_svo", no_astroquery)
    meta = svo._svo_fetch_metadata("X/X.q")
    assert meta == {"filter_id": "X/X.q", "WavelengthEff": 6200.5, "ZeroPoint": 3100.25}
    assert seen["params"] == {"ID": "X/X.q"} and seen["url"] == svo._pyphot().QUERY_URL
