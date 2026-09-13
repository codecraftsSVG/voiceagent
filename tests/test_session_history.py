from src.services.session_history import SessionHistory


def test_session_history_round_trip(tmp_path):
    path = tmp_path / "sessions.sqlite3"
    store = SessionHistory(path=str(path))
    session = store.create()

    store.append(session["id"], {"role": "user", "content": "  Hello  "})
    store.append(session["id"], {
        "role": "assistant",
        "content": "Hi there",
        "audio": b"audio must not be persisted",
        "latency": 0.42,
    })

    reloaded = SessionHistory(path=str(path)).list_sessions()[0]
    assert reloaded["title"] == "Hello"
    assert reloaded["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there", "latency": 0.42},
    ]
    assert "audio" not in reloaded["messages"][1]

    assert SessionHistory(path=str(path)).delete(session["id"])
    assert SessionHistory(path=str(path)).list_sessions() == []