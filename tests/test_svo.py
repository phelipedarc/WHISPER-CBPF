"""Band resolution + SVO Filter Profile Service fallback (all network calls mocked)."""
import warnings

import numpy as np
import pytest

from whisper_labia.io import bands, svo
from whisper_labia.io.svo import SvoUnavailable


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


def test_get_transmission_data_mocked(monkeypatch):
    monkeypatch.setattr(svo, "_svo_fetch_transmission",
                        lambda fid: (np.array([1000.0, 2000.0]), np.array([0.1, 0.9])))
    wl, tr = svo.get_transmission_data("PAN-STARRS/PS1.r")
    assert np.allclose(wl, [1000.0, 2000.0]) and np.allclose(tr, [0.1, 0.9])


def test_empty_svo_index_with_hint_is_graceful(svo_clean, monkeypatch):
    """A non-id band WITH a wavelength hint reaches _svo_fetch_index; an empty result degrades."""
    monkeypatch.setattr(svo, "_svo_fetch_index", lambda lo, hi: [])
    with pytest.warns(UserWarning):
        r = bands.resolve_band("customband", lambda_eff_hint=6000.0)
    assert r["source"] == "unresolved" and r["lambda_eff"] is None


def test_resolve_bands_vectorized(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no SVO for known bands")
    monkeypatch.setattr(svo, "_svo_fetch_metadata", boom)
    lam, zp, info = bands.resolve_bands(np.array(["g", "r", "g"]), svo_fallback=False)
    assert lam.shape == (3,) and zp.shape == (3,)
    assert info["g"]["source"] == "lsst"
    assert np.isfinite(lam).all()
