import importlib

from component4.common import config


def test_output_folders_exist():
    for folder in (config.DATA_DIR, config.MODELS_DIR, config.RESULTS_DIR):
        assert folder.is_dir(), f"missing folder: {folder}"


def test_every_module_package_imports():
    for name in config.MODULE_PACKAGES:
        importlib.import_module(f"component4.{name}")


def test_marks_folder_is_git_ignored():
    gitignore = (config.BASE_DIR.parent.parent / ".gitignore").read_text(encoding="utf-8")
    assert "backend/component4/data/marks/" in gitignore
