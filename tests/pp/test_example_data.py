from __future__ import annotations

import pathlib

import cellpin
import cellpin.pp._example_data as example_data


def test_load_sc_example_downloads_when_missing(monkeypatch, tmp_path):
    captured: dict[str, str] = {}

    def fake_urlretrieve(url, destination):
        captured["url"] = url
        captured["destination"] = str(destination)
        pathlib.Path(destination).write_bytes(b"fake")
        return str(destination), None

    def fake_read_h5ad(path):
        captured["read_path"] = str(path)
        return "mock_sc"

    monkeypatch.setattr(example_data.urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(example_data.ad, "read_h5ad", fake_read_h5ad)

    result = cellpin.pp.load_sc_example(cache_dir=tmp_path)

    assert result == "mock_sc"
    assert captured["url"].endswith("/datasets/sc_example.h5ad")
    assert captured["destination"].endswith("sc_example.h5ad")
    assert captured["read_path"].endswith("sc_example.h5ad")


def test_load_sc_example_uses_cache_when_present(monkeypatch, tmp_path):
    cached_file = tmp_path / "sc_example.h5ad"
    cached_file.write_bytes(b"already-there")

    was_called = {"download": False}

    def fake_urlretrieve(url, destination):
        was_called["download"] = True
        return str(destination), None

    def fake_read_h5ad(path):
        return str(path)

    monkeypatch.setattr(example_data.urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(example_data.ad, "read_h5ad", fake_read_h5ad)

    result = cellpin.pp.load_sc_example(cache_dir=tmp_path)

    assert was_called["download"] is False
    assert result == str(cached_file)


def test_load_sp_example_force_download(monkeypatch, tmp_path):
    cached_file = tmp_path / "sp_example.h5ad"
    cached_file.write_bytes(b"stale")

    was_called = {"download": False}

    def fake_urlretrieve(url, destination):
        was_called["download"] = True
        pathlib.Path(destination).write_bytes(b"fresh")
        return str(destination), None

    def fake_read_h5ad(path):
        return str(path)

    monkeypatch.setattr(example_data.urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(example_data.ad, "read_h5ad", fake_read_h5ad)

    result = cellpin.pp.load_sp_example(cache_dir=tmp_path, force_download=True)

    assert was_called["download"] is True
    assert result == str(cached_file)
