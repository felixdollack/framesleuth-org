"""Tests for sidecar ingestion (flat extension stream and structured form)."""

from framesleuth.pipeline.sidecars import (
    derive_error_evidence,
    derive_repro_steps,
    environment_from,
    parse_sidecars,
)

FLAT_EVENTS = [
    {
        "t": 0.0,
        "source": "env",
        "ua": "Mozilla/5.0 (Windows NT 10.0) Chrome/137",
        "url": "https://shop.test/cart/checkout",
    },
    {"t": 5.0, "source": "click", "selector": "button.save", "text": "Save"},
    {"t": 6.0, "source": "cursor", "x": 10, "y": 20},
    {"t": 7.0, "source": "network", "method": "post", "url": "/api/save", "status": 500},
    {"t": 7.1, "source": "network", "method": "get", "url": "/api/ok", "status": 200},
    {
        "t": 7.2,
        "source": "console",
        "text": "TypeError: x is undefined",
        "stack": "at save (a.js:1)",
    },
]


def test_parse_flat_stream_groups_by_source() -> None:
    """The flat event stream is grouped by its source discriminator."""
    parsed = parse_sidecars(FLAT_EVENTS)
    assert len(parsed.clicks) == 1
    assert len(parsed.network) == 2
    assert len(parsed.console_errors) == 1
    assert len(parsed.cursor) == 1
    assert parsed.env["url"].endswith("checkout")
    assert not parsed.is_empty


def test_parse_structured_form() -> None:
    """The documented structured Sidecars dict is parsed directly."""
    parsed = parse_sidecars(
        {
            "console_errors": [{"t": 1.0, "text": "boom"}],
            "network": [{"t": 2.0, "method": "GET", "url": "/x", "status": 404}],
            "clicks": [],
            "env": {"ua": "Chrome/137", "url": "https://a.test/"},
        }
    )
    assert parsed.console_errors[0]["text"] == "boom"
    assert parsed.network[0]["status"] == 404


def test_parse_none_and_garbage_is_safe() -> None:
    """Malformed or absent payloads never raise."""
    assert parse_sidecars(None).is_empty
    assert parse_sidecars("not-json").is_empty
    assert parse_sidecars(42).is_empty


def test_derive_error_evidence_includes_failed_network_and_console() -> None:
    """Console errors and 4xx/5xx responses become error evidence; 2xx is ignored."""
    parsed = parse_sidecars(FLAT_EVENTS)
    evidence = derive_error_evidence(parsed)
    sources = [e.source for e in evidence]
    assert sources.count("network") == 1  # only the 500, not the 200
    assert "console" in sources
    console = next(e for e in evidence if e.source == "console")
    assert "at save (a.js:1)" in console.text  # stack appended


def test_derive_repro_steps_from_clicks() -> None:
    """Clicks become numbered reproduction steps with citations."""
    parsed = parse_sidecars(FLAT_EVENTS)
    steps = derive_repro_steps(parsed)
    assert len(steps) == 1
    assert steps[0].action == "Click 'Save'"
    assert steps[0].evidence == ["click:0"]


def test_environment_detection() -> None:
    """Environment metadata is derived from the user agent and URL."""
    parsed = parse_sidecars(FLAT_EVENTS)
    env = environment_from(parsed)
    assert env["browser"] == "Chrome"
    assert env["os"] == "Windows"
    assert env["app"] == "shop.test"
    assert env["component"] == "cart/checkout"
