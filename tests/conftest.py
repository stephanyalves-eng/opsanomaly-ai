import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
import pytest
from opsanomaly.synthetic import generate
from opsanomaly.ingest import validate


@pytest.fixture
def frame():
    return validate(generate(rows=100))[0]
