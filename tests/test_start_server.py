import runpy
from pathlib import Path

import pytest
import uvicorn


@pytest.mark.parametrize(
    ("configured_port", "expected_port"),
    [("10000", 10000), (None, 8000)],
)
def test_start_server_uses_configured_port(
    monkeypatch: pytest.MonkeyPatch,
    configured_port: str | None,
    expected_port: int,
) -> None:
    if configured_port is None:
        monkeypatch.delenv("PORT", raising=False)
    else:
        monkeypatch.setenv("PORT", configured_port)

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda application, **kwargs: captured.update(kwargs),
    )

    script_path = Path(__file__).resolve().parents[1] / "start_server.py"
    runpy.run_path(str(script_path), run_name="__main__")

    assert captured == {"host": "0.0.0.0", "port": expected_port}
