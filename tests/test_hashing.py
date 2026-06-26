import pytest

from phonoflow.exceptions import ConfigError
from phonoflow.io.hash_utils import sha256_file


def test_sha256_file_hashes_existing_file(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("abc", encoding="utf-8")

    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_file_missing_path_is_clear(tmp_path):
    with pytest.raises(ConfigError, match="Cannot hash missing file"):
        sha256_file(tmp_path / "missing.txt")
