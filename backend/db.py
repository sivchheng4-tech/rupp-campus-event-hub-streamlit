"""BACKEND: data layer for the Campus Event Hub (SQLite, no Streamlit code in here).

Replaces what the Next.js app kept in localStorage and in Clerk:
users + login, events, RSVPs, reviews and profiles.
"""
import hashlib
import secrets
import sqlite3
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "campus_hub.db"   # lives at the project root, next to .gitignore

CATEGORIES = ["Tech & ITE", "Language & Culture", "Workshops", "Sports"]
YEARS = ["Year 1", "Year 2", "Year 3", "Year 4", "Graduate"]

PHOTOS = {
    "technology": "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=1200&q=85",
    "campus": "https://images.unsplash.com/photo-1523240795612-9a054b0db644?auto=format&fit=crop&w=1200&q=85",
    "workshop": "https://images.unsplash.com/photo-1517048676732-d65bc937f952?auto=format&fit=crop&w=1200&q=85",
    "culture": "https://images.unsplash.com/photo-1528605248644-14dd04022da1?auto=format&fit=crop&w=1200&q=85",
    "sports": "https://images.unsplash.com/photo-1517649763962-0c623066013b?auto=format&fit=crop&w=1200&q=85",
    "collaboration": "https://images.unsplash.com/photo-1521737711867-e3b97375f902?auto=format&fit=crop&w=1200&q=85",
}

# (id, title, category, date, time, location, club, image, seats already taken, capacity, description, speaker)
SEED_EVENTS = [
    ("rupp-ai-for-good", "AI for Good: Build a Better Future", "Tech & ITE", "2026-11-18", "2:00 PM – 4:30 PM",
     "STEM Building · Room 204", "RUPP Tech Society", PHOTOS["technology"], 38, 60,
     "A lively afternoon exploring how artificial intelligence can solve real challenges in our communities. "
     "Join student builders for short talks, demos, and a friendly idea sprint — no coding experience needed.",
     "Sokha Chann, AI Research Fellow"),
    ("khmer-night", "Khmer Stories, Shared Worlds", "Language & Culture", "2026-11-21", "5:30 PM – 8:00 PM",
     "IFL Campus · Garden Hall", "Khmer Language Club", PHOTOS["culture"], 24, 50,
     "An evening of contemporary Khmer storytelling, music, and conversation. Meet new friends, share a favorite "
     "phrase, and discover the many ways our language keeps changing and growing.",
     "Hosted by Vannak & the KLC team"),
    ("design-thinking-lab", "Design Thinking, Made Practical", "Workshops", "2026-11-25", "9:00 AM – 12:00 PM",
     "Building A · Innovation Lab", "RUPP Entrepreneurship Club", PHOTOS["workshop"], 17, 32,
     "Turn a fuzzy campus challenge into a testable idea in this hands-on workshop. We’ll practice listening, "
     "rapid prototyping, and giving useful feedback in small teams.",
     "Dara Lim, Product Designer at Koompi"),
    ("sunrise-run", "Sunrise Run & Campus Reset", "Sports", "2026-11-28", "6:15 AM – 8:00 AM",
     "Main Campus · North Gate", "RUPP Active Students", PHOTOS["sports"], 52, 80,
     "Start your weekend with an easy-paced loop around campus, a relaxed stretch, and iced coffee with the crew. "
     "All fitness levels are welcome — just bring comfortable shoes.",
     "Led by RUPP Active Students"),
    ("global-language-cafe", "Language Café: Around the Table", "Language & Culture", "2026-12-02", "3:00 PM – 5:00 PM",
     "IFL Campus · Courtyard", "International Students Circle", PHOTOS["campus"], 29, 45,
     "Pick up a new phrase, practice one you already know, and meet language learners from across campus. "
     "Rotating conversation tables make it easy to drop in at any level.",
     "With student hosts from 8 language groups"),
    ("student-maker-day", "Student Maker Day: Show & Tell", "Tech & ITE", "2026-12-05", "10:00 AM – 3:00 PM",
     "STEM Building · Atrium", "ITE Makers Collective", PHOTOS["collaboration"], 43, 100,
     "A welcoming showcase of student projects, small experiments, and big ideas. Wander the demos, find a "
     "collaborator, or bring a work-in-progress to share with the community.",
     "Featuring 20+ student makers"),
]


# --------------------------------------------------------------------------
# low-level helpers
# --------------------------------------------------------------------------
def q(sql, params=()):
    """Run one SQL statement; return rows as a list of dicts."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        conn.commit()
    return rows


def _now():
    """Server local time as 'YYYY-MM-DD HH:MM:SS' (what the login and activity logs store)."""
    return datetime.now().isoformat(sep=" ", timespec="seconds")


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 120_000).hex()
    return salt, digest


def init_db():
    q("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL, salt TEXT NOT NULL, pw_hash TEXT NOT NULL,
            name TEXT NOT NULL, student_id TEXT DEFAULT '', major TEXT DEFAULT '',
            year TEXT DEFAULT 'Year 1', role TEXT DEFAULT 'student')""")
    q("""CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, category TEXT NOT NULL,
            date TEXT NOT NULL, time TEXT NOT NULL, location TEXT NOT NULL, club TEXT NOT NULL,
            image TEXT, base_taken INTEGER DEFAULT 0, capacity INTEGER NOT NULL,
            description TEXT, speaker TEXT, created_by INTEGER)""")
    q("""CREATE TABLE IF NOT EXISTS rsvps (
            user_id INTEGER NOT NULL, event_id TEXT NOT NULL, PRIMARY KEY (user_id, event_id))""")
    q("""CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL, user_id INTEGER,
            name TEXT NOT NULL, rating INTEGER NOT NULL, comment TEXT NOT NULL)""")
    q("""CREATE TABLE IF NOT EXISTS logins (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, email TEXT NOT NULL,
            success INTEGER NOT NULL, at TEXT NOT NULL)""")
    q("""CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, user_id INTEGER,
            user_name TEXT, action TEXT NOT NULL, detail TEXT)""")
    if not q("SELECT id FROM events LIMIT 1"):
        for e in SEED_EVENTS:
            q("""INSERT INTO events (id, title, category, date, time, location, club, image,
                                     base_taken, capacity, description, speaker)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", e)
    if not q("SELECT id FROM users WHERE role='admin'"):
        salt, h = hash_password("admin123")
        q("INSERT INTO users (email, salt, pw_hash, name, role) VALUES (?,?,?,?,?)",
          ("admin@rupp.edu.kh", salt, h, "Campus Admin", "admin"))


# --------------------------------------------------------------------------
# users
# --------------------------------------------------------------------------
def register_user(name, email, password):
    """Returns (user_id, error_message). Exactly one of them is None."""
    name, email = name.strip(), email.strip().lower()
    if not name or not email or not password:
        return None, "Please fill in your name, email and password."
    if "@" not in email or "." not in email.split("@")[-1]:
        return None, "Please enter a valid email address."
    if len(password) < 6:
        return None, "Password must be at least 6 characters."
    if q("SELECT id FROM users WHERE email = ?", (email,)):
        return None, "An account with this email already exists. Try signing in."
    salt, h = hash_password(password)
    q("INSERT INTO users (email, salt, pw_hash, name) VALUES (?,?,?,?)", (email, salt, h, name))
    uid = q("SELECT id FROM users WHERE email = ?", (email,))[0]["id"]
    log_action(uid, name, "account_created", email)
    return uid, None


def authenticate(email, password):
    """Returns the user id, or None. Every attempt (good or bad) is written to the logins table."""
    email = email.strip().lower()
    uid = None
    rows = q("SELECT * FROM users WHERE email = ?", (email,))
    if rows:
        user = rows[0]
        _, h = hash_password(password, user["salt"])
        if secrets.compare_digest(h, user["pw_hash"]):
            uid = user["id"]
    q("INSERT INTO logins (user_id, email, success, at) VALUES (?,?,?,?)",
      (uid, email, 1 if uid else 0, _now()))
    return uid


def get_user(user_id):
    rows = q("SELECT id, email, name, student_id, major, year, role FROM users WHERE id = ?", (user_id,))
    return rows[0] if rows else None


def update_profile(user_id, name, student_id, major, year, email):
    email = email.strip().lower()
    clash = q("SELECT id FROM users WHERE email = ? AND id != ?", (email, user_id))
    if clash:
        return "Another account already uses that email."
    q("UPDATE users SET name=?, student_id=?, major=?, year=?, email=? WHERE id=?",
      (name.strip(), student_id.strip(), major.strip(), year, email, user_id))
    return None


# --------------------------------------------------------------------------
# events, RSVPs, reviews
# --------------------------------------------------------------------------
def list_events():
    rows = q("""SELECT e.*,
                  e.base_taken + (SELECT COUNT(*) FROM rsvps r WHERE r.event_id = e.id) AS taken,
                  (SELECT ROUND(AVG(rating), 1) FROM reviews v WHERE v.event_id = e.id) AS avg_rating,
                  (SELECT COUNT(*) FROM reviews v WHERE v.event_id = e.id) AS n_reviews
                FROM events e ORDER BY e.date, e.title""")
    for r in rows:
        r["spots_left"] = max(0, r["capacity"] - r["taken"])
    return rows


def create_event(user_id, title, category, date, time_text, location, club, image, seats, description, speaker):
    event_id = f"event-{int(time.time() * 1000)}"
    q("""INSERT INTO events (id, title, category, date, time, location, club, image,
                             base_taken, capacity, description, speaker, created_by)
         VALUES (?,?,?,?,?,?,?,?,0,?,?,?,?)""",
      (event_id, title.strip(), category, date, time_text, location.strip(), club.strip(),
       image.strip() or PHOTOS["collaboration"], int(seats), description.strip(),
       speaker.strip() or f"Presented by {club.strip()}", user_id))
    who = get_user(user_id)
    log_action(user_id, who["name"] if who else "", "event_created", title.strip())
    return event_id


def delete_event(event_id, user_id=None):
    """Deletes an event with its RSVPs and reviews, and records who did it in the activity log."""
    found = q("SELECT title FROM events WHERE id = ?", (event_id,))
    if found:
        who = get_user(user_id) if user_id else None
        log_action(user_id, who["name"] if who else "", "event_deleted", found[0]["title"])
    q("DELETE FROM events WHERE id = ?", (event_id,))
    q("DELETE FROM rsvps WHERE event_id = ?", (event_id,))
    q("DELETE FROM reviews WHERE event_id = ?", (event_id,))


def user_rsvps(user_id):
    return {r["event_id"] for r in q("SELECT event_id FROM rsvps WHERE user_id = ?", (user_id,))}


def toggle_rsvp(user_id, event_id):
    """Returns 'joined', 'left', 'full' or 'missing'."""
    ev = [e for e in list_events() if e["id"] == event_id]
    if not ev:
        return "missing"
    if event_id in user_rsvps(user_id):
        q("DELETE FROM rsvps WHERE user_id = ? AND event_id = ?", (user_id, event_id))
        return "left"
    if ev[0]["spots_left"] <= 0:
        return "full"
    q("INSERT OR IGNORE INTO rsvps (user_id, event_id) VALUES (?, ?)", (user_id, event_id))
    return "joined"


def add_review(event_id, user_id, name, rating, comment):
    q("INSERT INTO reviews (event_id, user_id, name, rating, comment) VALUES (?,?,?,?,?)",
      (event_id, user_id, name, int(rating), comment.strip()))


def get_reviews(event_id):
    return q("SELECT name, rating, comment FROM reviews WHERE event_id = ? ORDER BY id DESC", (event_id,))


# --------------------------------------------------------------------------
# admin reporting: counts, users, login history, activity log
# --------------------------------------------------------------------------
def log_action(user_id, user_name, action, detail=""):
    q("INSERT INTO audit_log (at, user_id, user_name, action, detail) VALUES (?,?,?,?,?)",
      (_now(), user_id, user_name, action, detail))


def _count(sql, params=()):
    return q(sql, params)[0]["n"]


def admin_stats():
    return {
        "users": _count("SELECT COUNT(*) AS n FROM users"),
        "students": _count("SELECT COUNT(*) AS n FROM users WHERE role = 'student'"),
        "admins": _count("SELECT COUNT(*) AS n FROM users WHERE role = 'admin'"),
        "events": _count("SELECT COUNT(*) AS n FROM events"),
        "rsvps": _count("SELECT COUNT(*) AS n FROM rsvps"),
        "reviews": _count("SELECT COUNT(*) AS n FROM reviews"),
        "logins_ok": _count("SELECT COUNT(*) AS n FROM logins WHERE success = 1"),
        "logins_failed": _count("SELECT COUNT(*) AS n FROM logins WHERE success = 0"),
        "logins_today": _count("SELECT COUNT(*) AS n FROM logins WHERE success = 1 "
                               "AND date(at) = date('now', 'localtime')"),
        "events_deleted": _count("SELECT COUNT(*) AS n FROM audit_log WHERE action = 'event_deleted'"),
    }


def logins_by_day(days=14):
    return q("""SELECT date(at) AS day, SUM(success) AS ok, SUM(1 - success) AS failed
                FROM logins WHERE date(at) >= date('now', 'localtime', ?)
                GROUP BY day ORDER BY day""", (f"-{int(days) - 1} days",))


def list_users():
    return q("""SELECT u.id, u.name, u.email, u.role, u.student_id, u.major, u.year,
                  (SELECT COUNT(*) FROM logins l WHERE l.user_id = u.id AND l.success = 1) AS logins,
                  (SELECT MAX(at) FROM logins l WHERE l.user_id = u.id AND l.success = 1) AS last_login,
                  (SELECT COUNT(*) FROM rsvps r WHERE r.user_id = u.id) AS rsvps
                FROM users u ORDER BY u.id""")


def list_logins(limit=100):
    return q("SELECT at, email, success FROM logins ORDER BY id DESC LIMIT ?", (int(limit),))


def list_audit(limit=100):
    return q("SELECT at, user_name, action, detail FROM audit_log ORDER BY id DESC LIMIT ?", (int(limit),))
