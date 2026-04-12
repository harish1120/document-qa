import json
import pytest
from io import BytesIO


def test_health(client):
    res = client.get("/health")
    print(res.json())
    assert res.status_code == 200


def test_upload_pdf(client):
    # Create a fake PDF
    pdf_content = b"%PDF-1.4 fake content"
    files = {"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")}

    response = client.post("/upload_pdf", files=files)

    assert response.status_code == 200
    assert "message" in response.json()
