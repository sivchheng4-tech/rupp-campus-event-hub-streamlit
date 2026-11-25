import pytest

from backend import db


@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    return db


def test_seed_data(fresh_db):
    events = fresh_db.list_events()
    assert len(events) == 6
    first = next(e for e in events if e["id"] == "rupp-ai-for-good")
    assert first["spots_left"] == 22            # 60 capacity - 38 already taken


def test_register_and_login(fresh_db):
    uid, err = fresh_db.register_user("Sophea Chan", "Sophea@Example.com", "secret1")
    assert err is None and uid
    assert fresh_db.authenticate("sophea@example.com", "secret1") == uid
    assert fresh_db.authenticate("sophea@example.com", "wrong") is None
    assert fresh_db.authenticate("nobody@example.com", "secret1") is None


def test_register_validation(fresh_db):
    assert fresh_db.register_user("", "a@b.com", "secret1")[1]
    assert fresh_db.register_user("A", "not-an-email", "secret1")[1]
    assert fresh_db.register_user("A", "a@b.com", "123")[1]
    fresh_db.register_user("A", "a@b.com", "secret1")
    assert "already exists" in fresh_db.register_user("B", "A@B.com", "secret2")[1]


def test_admin_account_seeded(fresh_db):
    uid = fresh_db.authenticate("admin@rupp.edu.kh", "admin123")
    assert fresh_db.get_user(uid)["role"] == "admin"


def test_rsvp_toggle_and_capacity(fresh_db):
    uid, _ = fresh_db.register_user("A", "a@b.com", "secret1")
    assert fresh_db.toggle_rsvp(uid, "sunrise-run") == "joined"
    ev = next(e for e in fresh_db.list_events() if e["id"] == "sunrise-run")
    assert ev["spots_left"] == 80 - 52 - 1
    assert fresh_db.toggle_rsvp(uid, "sunrise-run") == "left"
    assert fresh_db.toggle_rsvp(uid, "nope") == "missing"

    eid = fresh_db.create_event(uid, "Tiny meetup", "Workshops", "2026-12-20", "1:00 PM – 2:00 PM",
                                "Room 1", "Club", "", 1, "A tiny meetup for testing things", "")
    u2, _ = fresh_db.register_user("B", "b@b.com", "secret1")
    assert fresh_db.toggle_rsvp(uid, eid) == "joined"
    assert fresh_db.toggle_rsvp(u2, eid) == "full"


def test_create_delete_event(fresh_db):
    uid, _ = fresh_db.register_user("A", "a@b.com", "secret1")
    eid = fresh_db.create_event(uid, "  Coding night  ", "Tech & ITE", "2026-12-20", "6:00 PM – 8:00 PM",
                                "Lab 2", "Tech Club", "", 30, "Come and write some code together", "")
    ev = next(e for e in fresh_db.list_events() if e["id"] == eid)
    assert ev["title"] == "Coding night" and ev["speaker"] == "Presented by Tech Club"
    assert ev["image"] == db.PHOTOS["collaboration"]
    fresh_db.toggle_rsvp(uid, eid)
    fresh_db.delete_event(eid)
    assert all(e["id"] != eid for e in fresh_db.list_events())
    assert eid not in fresh_db.user_rsvps(uid)


def test_reviews_and_profile(fresh_db):
    uid, _ = fresh_db.register_user("A", "a@b.com", "secret1")
    fresh_db.add_review("khmer-night", uid, "A", 4, "Lovely evening")
    fresh_db.add_review("khmer-night", uid, "A", 5, "Went twice")
    ev = next(e for e in fresh_db.list_events() if e["id"] == "khmer-night")
    assert ev["n_reviews"] == 2 and ev["avg_rating"] == 4.5
    assert fresh_db.update_profile(uid, "Sophea", "RU-1", "ITE", "Year 3", "new@b.com") is None
    assert fresh_db.get_user(uid)["major"] == "ITE"
    other, _ = fresh_db.register_user("B", "b@b.com", "secret1")
    assert fresh_db.update_profile(other, "B", "", "", "Year 1", "new@b.com")  # email taken -> error text


def test_login_tracking(fresh_db):
    uid, _ = fresh_db.register_user("A", "a@b.com", "secret1")
    assert fresh_db.authenticate("a@b.com", "secret1") == uid
    assert fresh_db.authenticate("a@b.com", "wrong") is None
    assert fresh_db.authenticate("ghost@b.com", "x") is None
    s = fresh_db.admin_stats()
    assert s["logins_ok"] == 1 and s["logins_failed"] == 2 and s["logins_today"] == 1
    user = next(u for u in fresh_db.list_users() if u["id"] == uid)
    assert user["logins"] == 1 and user["last_login"]
    assert [r["success"] for r in fresh_db.list_logins()] == [0, 0, 1]      # newest first
    assert fresh_db.logins_by_day(14)[0]["ok"] == 1


def test_delete_is_logged_and_counted(fresh_db):
    admin = next(u for u in fresh_db.list_users() if u["role"] == "admin")
    fresh_db.delete_event("khmer-night", admin["id"])
    fresh_db.delete_event("does-not-exist", admin["id"])       # unknown id: nothing logged
    assert fresh_db.admin_stats()["events_deleted"] == 1
    last = fresh_db.list_audit()[0]
    assert last["action"] == "event_deleted" and last["user_name"] == "Campus Admin"
    assert last["detail"] == "Khmer Stories, Shared Worlds"
    assert fresh_db.admin_stats()["events"] == 5
