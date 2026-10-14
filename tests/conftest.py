import pytest


@pytest.fixture(autouse=True)
def no_real_tagger(monkeypatch):
    """Uploads rate the image; tests never download or run the model."""
    from chagama.infrastructure.imaging import safety

    def unavailable(*_args, **_kwargs):
        raise safety.SafetyUnavailable("disabled in tests")

    monkeypatch.setattr(safety, "_load", unavailable)
