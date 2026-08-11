from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def data_dir():
    return DATA_DIR


@pytest.fixture
def at2017gfo_csv():
    return DATA_DIR / "at2017gfo.csv"
