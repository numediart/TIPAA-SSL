import pytest

from app import init_app


@pytest.fixture
def app():
    app = init_app("app.config.TestingConfig")
    yield app


@pytest.fixture
def client(app):
    return app.test_client()
