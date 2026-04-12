from fastapi.testclient import TestClient
import pytest
from dotenv import load_dotenv
load_dotenv(override=True)

from backend.main import app

@pytest.fixture()
def client():
    yield TestClient(app)