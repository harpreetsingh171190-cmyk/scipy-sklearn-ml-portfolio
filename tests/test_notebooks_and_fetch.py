import importlib.util
import zipfile
from pathlib import Path

import pytest

import fetch_data

ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_notebooks", ROOT / "scripts" / "check_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_notebooks_are_valid_and_parse():
    checker = _load_checker()
    paths = sorted((ROOT / "notebooks").glob("*.ipynb")) + [ROOT / "check_env.ipynb"]
    assert len(paths) >= 30
    problems = [p for path in paths for p in checker.check(path)]
    assert problems == []


def test_notebook_checker_flags_syntax_errors(tmp_path):
    checker = _load_checker()
    bad = tmp_path / "bad.ipynb"
    bad.write_text(
        '{"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": '
        '[{"cell_type": "code", "source": ["print(1"], "metadata": {}, "outputs": [], "execution_count": null}]}',
        encoding="utf-8",
    )
    assert any("cell 0" in p for p in checker.check(bad))


def test_fetch_data_extracts_archive_into_notebooks_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_data, "__file__", str(tmp_path / "fetch_data.py"))
    (tmp_path / "notebooks").mkdir()
    with zipfile.ZipFile(tmp_path / "notebooks" / "datasets.zip", "w") as zf:
        zf.writestr("datasets/sample.txt", "hello")
    folder = fetch_data.get_datasets_folder()
    assert (Path(folder) / "sample.txt").read_text() == "hello"


def test_fetch_data_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_data, "__file__", str(tmp_path / "fetch_data.py"))
    (tmp_path / "notebooks").mkdir()
    with zipfile.ZipFile(tmp_path / "notebooks" / "datasets.zip", "w") as zf:
        zf.writestr("../evil.txt", "nope")
    with pytest.raises(ValueError, match="Unsafe path"):
        fetch_data.get_datasets_folder()
    assert not (tmp_path / "evil.txt").exists()
