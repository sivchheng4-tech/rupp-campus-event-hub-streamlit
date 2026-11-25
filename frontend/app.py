"""FRONTEND: RUPP Campus Event Hub - Streamlit pages.

Run from the project root with:   streamlit run frontend/app.py
The UI talks to the backend (backend/db.py) and the ML pipeline (ml/service.py).
"""
import sys
from datetime import date, datetime
from pathlib import Path

# Make the project root importable so `backend` and `ml` work wherever Streamlit is started from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from backend import db
from ml import config as cfg
from ml import service

st.set_page_config(page_title="RUPP Campus Event Hub", page_icon="🎓", layout="wide")

CREST_URL = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/image-SnVhkQSQof1zAoopTigqQPp8Otp9Yo.png"
LEVEL_TEXT = {"low": "Low turnout expected", "medium": "Medium turnout expected", "high": "High turnout expected"}

st.markdown(
    """
<style>
.hero {background: linear-gradient(135deg,#174a37 0%,#245f48 100%); color:#fffefa; padding:46px 40px;
       border-radius:20px; margin-bottom:18px;}
.hero small {letter-spacing:.14em; opacity:.8;}
.hero h1 {color:#fffefa; font-size:3rem; line-height:1.1; margin:.4rem 0 .6rem 0;}
.hero em {color:#f2e9d5;}
.hero p {font-size:1.1rem; opacity:.9; margin:0;}
.banner {background:#f2e9d5; border-radius:16px; padding:22px 26px; margin-top:24px;}
.banner h3 {margin:0 0 4px 0;}
.eyebrow {letter-spacing:.12em; font-size:.75rem; color:#777b72; font-weight:700;}
div[data-testid="stImage"] img {border-radius:12px; aspect-ratio:16/9; object-fit:cover;}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# cached data + small helpers
# --------------------------------------------------------------------------
@st.cache_resource
def setup():
    db.init_db()
    return True


@st.cache_data(ttl=60, show_spinner=False)
def load_events():
    """Cached list of events. Cleared by refresh() after every change."""
    return db.list_events()


def refresh():
    load_events.clear()


@st.cache_resource(show_spinner="Training the attendance model for the first time (a few seconds)…")
def ensure_model():
    """Cleans the data and trains the model if models/model.joblib is missing."""
    if not cfg.MODEL_PATH.exists():
        from ml.data import make_dataset
        from ml.train import run_training
        make_dataset.main()
        run_training(verbose=False)
    return True


def fmt_date(s):
    d = datetime.strptime(s, "%Y-%m-%d")
    return {"day": f"{d.day:02d}", "month": d.strftime("%b").upper(), "full": f"{d:%A, %B} {d.day}, {d.year}"}


def stars(avg, n):
    if not n or avg is None:
        return "No reviews yet"
    return f"{'★' * int(round(avg))}{'☆' * (5 - int(round(avg)))}  {avg}/5 ({n})"


def current_user():
    uid = st.session_state.get("user_id")
    return db.get_user(uid) if uid else None


def goto(page):
    """Change page from normal code (the radio widget already exists on this run)."""
    st.session_state["_goto"] = page
    st.rerun()


def set_page(page):
    """Use as a button callback: runs before the widgets are created."""
    st.session_state["page"] = page


def sign_in_prompt(message="Please sign in to continue."):
    st.info(message)
    st.button("Go to sign in", on_click=set_page, args=("Sign in",))


# --------------------------------------------------------------------------
# event cards + dialogs
# --------------------------------------------------------------------------
def do_rsvp(user, event_id):
    if not user:
        st.toast("Please sign in to RSVP.", icon="🔒")
        return
    result = db.toggle_rsvp(user["id"], event_id)
    refresh()
    messages = {"joined": ("You're on the list! 🎉", None), "left": ("RSVP cancelled.", None),
                "full": ("Sorry, this event is full.", "⚠️"), "missing": ("That event no longer exists.", "⚠️")}
    text, icon = messages[result]
    st.toast(text, icon=icon)
    st.rerun()


@st.dialog("Event details", width="large")
def event_dialog(event_id):
    ev = next((e for e in load_events() if e["id"] == event_id), None)
    user = current_user()
    if ev is None:
        st.warning("This event no longer exists.")
        return
    going = bool(user) and event_id in db.user_rsvps(user["id"])
    d = fmt_date(ev["date"])
    st.image(ev["image"])
    st.caption(f"{ev['category'].upper()}  ·  {d['full']}")
    st.header(ev["title"])
    st.write(ev["description"])
    c1, c2 = st.columns(2)
    c1.markdown(f"📍 **Location**  \n{ev['location']}")
    c1.markdown(f"🏛️ **Organized by**  \n{ev['club']}")
    c2.markdown(f"🕒 **Time**  \n{ev['time']}")
    c2.markdown(f"✨ **Featured speaker**  \n{ev['speaker']}")
    st.metric("Spots left", f"{ev['spots_left']} / {ev['capacity']}")

    if st.button("✓ You're on the list (tap to cancel)" if going else "Confirm RSVP",
                 type="secondary" if going else "primary", key=f"dlg_rsvp_{event_id}"):
        do_rsvp(user, event_id)
    if user and user["role"] == "admin":
        sure = st.checkbox("I understand: delete this event with its RSVPs and reviews", key=f"dlg_sure_{event_id}")
        if st.button("🗑️ Delete event (admin)", key=f"dlg_del_{event_id}", disabled=not sure):
            db.delete_event(event_id, user["id"])
            refresh()
            st.toast("Event deleted.")
            st.rerun()

    st.divider()
    st.subheader("Community notes")
    st.caption(stars(ev["avg_rating"], ev["n_reviews"]))
    for r in db.get_reviews(event_id)[:5]:
        st.markdown(f"**{'★' * r['rating']}** — {r['comment']}  \n*{r['name']}*")
    if not user:
        st.caption("Sign in to leave a review.")
    else:
        with st.form(f"review_{event_id}", clear_on_submit=True):
            rating = st.slider("Your rating", 1, 5, 5, help="1 = disappointing, 5 = excellent")
            comment = st.text_area("Share a thought with the community…", max_chars=240)
            if st.form_submit_button("Post review"):
                if not comment.strip():
                    st.error("Please write a short comment.")
                else:
                    db.add_review(event_id, user["id"], user["name"], rating, comment)
                    refresh()
                    st.toast("Thanks for sharing your thoughts.")
                    st.rerun()


def event_card(ev, user, rsvps, prefix):
    d = fmt_date(ev["date"])
    going = ev["id"] in rsvps
    with st.container(border=True):
        st.image(ev["image"])
        st.caption(f"**{ev['category'].upper()}**  ·  {d['month']} {d['day']}")
        st.markdown(f"#### {ev['title']}")
        st.caption(f"🕒 {ev['time'].split(' – ')[0]}   ·   📍 {ev['location'].split(' · ')[0]}")
        st.write(ev["club"])
        st.caption(f"👥 {ev['spots_left']} spots left  ·  {stars(ev['avg_rating'], ev['n_reviews'])}")
        c1, c2 = st.columns(2)
        if c1.button("Details", key=f"{prefix}_details_{ev['id']}"):
            event_dialog(ev["id"])
        if c2.button("✓ Going" if going else "RSVP", key=f"{prefix}_rsvp_{ev['id']}",
                     type="primary" if going else "secondary"):
            do_rsvp(user, ev["id"])


def render_grid(events, user, rsvps, prefix):
    for i in range(0, len(events), 3):
        row = events[i:i + 3]
        cols = st.columns(3)
        for col, ev in zip(cols, row):
            with col:
                event_card(ev, user, rsvps, prefix)


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------
def page_home(user):
    events = load_events()
    rsvps = db.user_rsvps(user["id"]) if user else set()

    st.markdown(
        """<div class="hero"><small>PHNOM PENH, CAMBODIA · EST. 1960</small>
        <h1>Find your people.<br><em>Find your thing.</em></h1>
        <p>Ideas, friendships, and the best of campus life — all happening right here at RUPP.</p></div>""",
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Events to explore", len(events))
    m2.metric("Active clubs", len({e["club"] for e in events}))
    m3.metric("Your RSVPs", len(rsvps) if user else "–", help="Sign in to RSVP to events.")

    st.markdown('<span class="eyebrow">THE CAMPUS CALENDAR</span>', unsafe_allow_html=True)
    st.header("A good week starts here.")
    f1, f2 = st.columns([3, 2])
    category = f1.radio("Category", ["All"] + db.CATEGORIES, horizontal=True, key="filter_category",
                        help="Show only events of one kind.")
    query = f2.text_input("Search", key="filter_query", placeholder="Search events, clubs…",
                          help="Matches the title, club, location and category.")

    q_text = query.strip().lower()
    shown = [e for e in events
             if (category == "All" or e["category"] == category)
             and q_text in f"{e['title']} {e['club']} {e['location']} {e['category']}".lower()]
    st.caption(f"{len(shown)} upcoming gathering(s)")

    if not shown:
        st.info("No events found just yet. Try another search or category to find your next campus moment.")
        st.button("Clear filters", on_click=lambda: st.session_state.update(filter_category="All", filter_query=""))
    else:
        counts = pd.Series([e["category"] for e in shown]).value_counts()
        st.caption("Events per category (updates with your filters)")
        st.bar_chart(counts, x_label="Category", y_label="Number of events")
        render_grid(shown, user, rsvps, "home")

    st.markdown(
        """<div class="banner"><span class="eyebrow">GOT SOMETHING IN MIND?</span>
        <h3>Your club could be next.</h3><p>Make a little room for something wonderful.</p></div>""",
        unsafe_allow_html=True,
    )
    st.button("Create an event", on_click=set_page, args=("Create Event" if user else "Sign in",), type="primary")


def page_sign_in():
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.title("Welcome to the Hub")
        st.write("Sign in to RSVP to events, leave reviews and publish your own.")
        tab_in, tab_up = st.tabs(["Sign in", "Create account"])
        with tab_in:
            with st.form("sign_in_form"):
                email = st.text_input("Email", help="The email you signed up with.")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Sign in", type="primary"):
                    uid = db.authenticate(email, password)
                    if uid:
                        st.session_state["user_id"] = uid
                        goto("Home")
                    else:
                        st.error("Wrong email or password.")
            st.caption("Demo admin: admin@rupp.edu.kh / admin123")
        with tab_up:
            with st.form("sign_up_form"):
                name = st.text_input("Full name")
                email = st.text_input("Email", key="signup_email")
                password = st.text_input("Password (at least 6 characters)", type="password", key="signup_pw")
                if st.form_submit_button("Create account", type="primary"):
                    uid, err = db.register_user(name, email, password)
                    if err:
                        st.error(err)
                    else:
                        st.session_state["user_id"] = uid
                        goto("Profile")


def page_my_rsvps(user):
    events = load_events()
    rsvps = db.user_rsvps(user["id"])
    joined = sorted((e for e in events if e["id"] in rsvps), key=lambda e: e["date"])

    st.markdown('<span class="eyebrow">YOUR CAMPUS CALENDAR</span>', unsafe_allow_html=True)
    st.title("Good things you're going to.")
    st.metric("Events saved", len(joined))

    today = date.today().isoformat()
    upcoming = [e for e in joined if e["date"] >= today]
    if upcoming:
        nxt = upcoming[0]
        st.info(f"🔔 **Your next campus moment:** {nxt['title']} — {fmt_date(nxt['date'])['full']} · "
                f"{nxt['time']} · {nxt['location']}")
    if joined:
        st.subheader(f"On your calendar ({len(joined)})")
        render_grid(joined, user, rsvps, "mine")
    else:
        st.info("Your calendar is waiting. RSVP to an event and it will be right here, ready when you are.")
        st.button("Find your first event", on_click=set_page, args=("Home",), type="primary")

    with st.container(border=True):
        st.markdown('<span class="eyebrow">YOUR RUPP PROFILE</span>', unsafe_allow_html=True)
        st.markdown(f"**{user['name']}**  \n{user['major'] or 'Major not set'} · {user['year']} · "
                    f"{user['student_id'] or 'No student ID'}")
        st.button("Edit profile", on_click=set_page, args=("Profile",))


def page_create(user):
    st.markdown('<span class="eyebrow">FOR THE PEOPLE WHO MAKE IT HAPPEN</span>', unsafe_allow_html=True)
    st.title("Put your next big idea on the map.")
    st.write("Bring your campus community together. Tell us what you're planning and we'll save a spot for everyone.")
    with st.form("create_event"):
        title = st.text_input("Event name *", max_chars=70, placeholder="e.g. Creative Coding Night",
                              help="At least 5 characters.")
        c1, c2 = st.columns(2)
        category = c1.selectbox("Category *", db.CATEGORIES)
        club = c2.text_input("Organizing club *", placeholder="Your student club")
        c3, c4, c5 = st.columns(3)
        when = c3.date_input("Date *", value=date.today(), min_value=date.today())
        start = c4.time_input("Start time *", value=datetime.strptime("14:00", "%H:%M").time())
        end = c5.time_input("End time *", value=datetime.strptime("16:00", "%H:%M").time())
        location = st.text_input("Campus location *", placeholder="Building A · Room 201")
        c6, c7 = st.columns(2)
        seats = c6.number_input("Available seats *", min_value=1, max_value=500, value=30, step=1)
        speaker = c7.text_input("Speaker / host", placeholder="Name and role")
        image = st.text_input("Event cover image URL", placeholder="https://… (optional)")
        description = st.text_area("A few words about it *", max_chars=500,
                                   placeholder="What will students experience? Who should come?",
                                   help="At least 20 characters.")
        if st.form_submit_button("Publish event", type="primary"):
            errors = []
            if len(title.strip()) < 5:
                errors.append("Event name must be at least 5 characters.")
            if not club.strip():
                errors.append("Please enter the organizing club.")
            if not location.strip():
                errors.append("Please enter the campus location.")
            if len(description.strip()) < 20:
                errors.append("Please write at least 20 characters about the event.")
            if end <= start:
                errors.append("End time must be after the start time.")
            if image.strip() and not image.strip().startswith(("http://", "https://")):
                errors.append("The image URL must start with http:// or https://")
            if errors:
                for e in errors:
                    st.error(e)
            else:
                fmt = lambda t: t.strftime("%I:%M %p").lstrip("0")
                db.create_event(user["id"], title, category, when.isoformat(), f"{fmt(start)} – {fmt(end)}",
                                location, club, image, seats, description, speaker)
                refresh()
                st.session_state["flash"] = "Your event is live on the RUPP campus calendar."
                st.session_state["filter_category"] = "All"
                st.session_state["filter_query"] = ""
                goto("Home")


def predict_input(f):
    """One Streamlit widget per field in FEATURE_SCHEMA. Returns the value (or None if optional and skipped)."""
    key = f"pred_{f['name']}"
    if f["type"] == "select":
        help_text = f"Choose one option for {f['label'].lower()}."
        labels = [o["label"] for o in f["options"]]
        idx = next((i for i, o in enumerate(f["options"]) if str(o["value"]) == str(f.get("default"))), 0)
        label = st.selectbox(f["label"], labels, index=idx, help=help_text, key=key)
        return next(o["value"] for o in f["options"] if o["label"] == label)
    if f["type"] == "date":
        return st.date_input(f["label"], value=date.fromisoformat(f["default"]), min_value=date.today(),
                             help="Pick the date of the event.", key=key).isoformat()
    # number
    is_int = all(float(f[k]).is_integer() for k in ("min", "max", "step", "default"))
    cast = int if is_int else float
    help_text = f"A number from {f['min']} to {f['max']}."
    if not f["required"]:
        if not st.checkbox(f"Include: {f['label']}", value=True, key=key + "_on",
                           help="Optional. Untick if you don't know it; the model will fill in a typical value."):
            return None
    return st.number_input(f["label"], min_value=cast(f["min"]), max_value=cast(f["max"]),
                           value=cast(f["default"]), step=cast(f["step"]), help=help_text, key=key)


def page_predict(user):
    st.markdown('<span class="eyebrow">DATA SCIENCE</span>', unsafe_allow_html=True)
    st.title("Predict event attendance")
    st.write("Describe your event and a model trained on past RUPP events estimates how many people will come.")

    try:
        ensure_model()
        metrics = service.get_metrics()
    except Exception as exc:  # noqa: BLE001 - show any training problem in the page, not a red traceback
        st.error(f"The prediction model could not be prepared: {exc}")
        return

    left, right = st.columns([3, 2])
    with left:
        with st.form("predict_form"):
            values = {}
            fields = service.get_schema()
            grid = st.columns(2)
            for i, f in enumerate(fields):
                with grid[i % 2]:
                    values[f["name"]] = predict_input(f)
            submitted = st.form_submit_button("Predict attendance", type="primary")
        if submitted:
            if values.get("is_free") == 1:
                values["ticket_price"] = 0  # free events have no price
            try:
                with st.spinner("Calculating…"):
                    st.session_state["pred_result"] = service.predict(values)
            except ValueError as exc:
                st.session_state.pop("pred_result", None)
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.session_state.pop("pred_result", None)
                st.error(f"The prediction service isn't available right now: {exc}")

        res = st.session_state.get("pred_result")
        if res:
            st.subheader("Result")
            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted attendance", f"{res['predicted_attendance']} people")
            r2.metric("Venue capacity", res["capacity"])
            r3.metric("Fill rate", f"{res['fill_rate'] * 100:.0f}%")
            st.progress(min(1.0, max(0.0, res["fill_rate"])))
            {"low": st.warning, "medium": st.info, "high": st.success}[res["level"]](LEVEL_TEXT[res["level"]])
        else:
            st.caption("Fill in the form and press **Predict attendance** to see the result here.")

    with right:
        st.subheader("How good is the model?")
        t = metrics["test"]
        k1, k2 = st.columns(2)
        k1.metric("Average error", f"{t['mae']:.1f} people", help="Mean absolute error on events the model never saw.")
        k2.metric("R² score", f"{t['r2']:.2f}", help="1.0 is perfect, 0 is no better than guessing the average.")
        st.caption(f"Best model: **{metrics['best_model']}** · trained on {metrics['n_train']} events, "
                   f"tested on {metrics['n_test']}.")

        st.markdown("**Model error vs guessing the average**")
        st.caption("Average error in people (shorter bar is better)")
        st.bar_chart(pd.DataFrame({"Average error (people)": [t["mae"], metrics["baseline"]["test_mae"]]},
                                  index=["Our model", "Guess the average"]))

        st.markdown("**What matters most to the prediction**")
        st.caption("Top 5 features (longer bar means more important)")
        top = sorted(metrics["feature_importance"], key=lambda r: r["importance"], reverse=True)[:5]
        st.bar_chart(pd.DataFrame({"Importance": [max(r["importance"], 0) for r in top]},
                                  index=[r["feature"] for r in top]))

    with st.expander("Model comparison (cross-validation)"):
        st.dataframe(pd.DataFrame(metrics["cv_results"]).rename(
            columns={"model": "Model", "rmse_mean": "RMSE (mean)", "rmse_std": "RMSE (std)"}), hide_index=True)
    history = service.get_history()
    if history:
        with st.expander(f"Recent predictions ({len(history)})"):
            st.dataframe(pd.DataFrame([{**h["features"], **h["result"], "time": h["time"]} for h in history]),
                         hide_index=True)

    if user["role"] == "admin":
        st.divider()
        if st.button("Retrain model (admin)", help="Re-runs training on data/processed/clean.csv."):
            with st.spinner("Training… this can take a minute."):
                try:
                    service.train()
                    st.toast("Model retrained.", icon="✅")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Training failed: {exc}")


def page_admin(user):
    """Admin-only dashboard: totals, login history, users, event management and the activity log."""
    st.markdown('<span class="eyebrow">ADMIN</span>', unsafe_allow_html=True)
    st.title("Admin Dashboard")
    if not user or user["role"] != "admin":
        st.error("Only admins can see this page.")
        return

    stats = db.admin_stats()
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Accounts", stats["users"], help=f"{stats['students']} students · {stats['admins']} admin(s)")
    a2.metric("Successful sign-ins", stats["logins_ok"], help="Counted since login tracking was added.")
    a3.metric("Sign-ins today", stats["logins_today"])
    a4.metric("Failed sign-ins", stats["logins_failed"])
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Events", stats["events"])
    b2.metric("Events deleted", stats["events_deleted"])
    b3.metric("RSVPs", stats["rsvps"])
    b4.metric("Reviews", stats["reviews"])

    tab_login, tab_users, tab_events, tab_log = st.tabs(["Sign-ins", "Users", "Events", "Activity log"])

    with tab_login:
        by_day = db.logins_by_day(14)
        if by_day:
            chart = pd.DataFrame(by_day).rename(columns={"ok": "Successful", "failed": "Failed"}).set_index("day")
            st.caption("Sign-ins per day, last 14 days")
            st.bar_chart(chart[["Successful", "Failed"]])
        else:
            st.info("No sign-ins recorded yet. They appear here as soon as someone signs in.")
        history = db.list_logins(100)
        if history:
            df = pd.DataFrame(history)
            df["success"] = df["success"].map({1: "✓ success", 0: "✗ failed"})
            st.dataframe(df.rename(columns={"at": "Time (server)", "email": "Email", "success": "Result"}),
                         hide_index=True, use_container_width=True)

    with tab_users:
        users = pd.DataFrame(db.list_users())
        users["last_login"] = users["last_login"].fillna("never")
        st.dataframe(users.rename(columns={"id": "ID", "name": "Name", "email": "Email", "role": "Role",
                                           "student_id": "Student ID", "major": "Major", "year": "Year",
                                           "logins": "Sign-ins", "last_login": "Last sign-in", "rsvps": "RSVPs"}),
                     hide_index=True, use_container_width=True)

    with tab_events:
        events = load_events()
        if events:
            st.dataframe(pd.DataFrame([{"Title": e["title"], "Date": e["date"], "Club": e["club"],
                                        "Taken": e["taken"], "Capacity": e["capacity"],
                                        "Reviews": e["n_reviews"]} for e in events]),
                         hide_index=True, use_container_width=True)
            st.subheader("Delete an event")
            labels = {f"{e['title']} ({e['date']})": e["id"] for e in events}
            choice = st.selectbox("Event", list(labels), key="admin_del_choice")
            sure = st.checkbox("I understand: this also deletes its RSVPs and reviews.", key="admin_del_sure")
            if st.button("🗑️ Delete event", disabled=not sure, key="admin_del_btn"):
                db.delete_event(labels[choice], user["id"])
                refresh()
                st.toast("Event deleted.")
                st.rerun()
        else:
            st.info("There are no events.")

    with tab_log:
        audit = db.list_audit(100)
        if audit:
            names = {"event_deleted": "Event deleted", "event_created": "Event created",
                     "account_created": "Account created"}
            df = pd.DataFrame(audit)
            df["action"] = df["action"].map(lambda a: names.get(a, a))
            st.dataframe(df.rename(columns={"at": "Time (server)", "user_name": "Who", "action": "Action",
                                            "detail": "Details"}),
                         hide_index=True, use_container_width=True)
        else:
            st.info("Nothing logged yet. Creating or deleting an event, or a new account, shows up here.")


def page_profile(user):
    st.markdown('<span class="eyebrow">YOUR CAMPUS PROFILE</span>', unsafe_allow_html=True)
    st.title("A little about you.")
    st.caption("Your details help clubs welcome you to campus events.")
    c1, c2 = st.columns(2)
    c1.metric("Role", user["role"].title())
    c2.metric("Events saved", len(db.user_rsvps(user["id"])))
    with st.form("profile_form"):
        name = st.text_input("Full name", user["name"])
        student_id = st.text_input("Student ID", user["student_id"], placeholder="RU-2024-01842")
        major = st.text_input("Major", user["major"], placeholder="Information Technology Engineering")
        year = st.selectbox("Year of study", db.YEARS,
                            index=db.YEARS.index(user["year"]) if user["year"] in db.YEARS else 0)
        email = st.text_input("Student email", user["email"])
        if st.form_submit_button("Save profile", type="primary"):
            if not name.strip() or "@" not in email:
                st.error("Please enter your name and a valid email.")
            else:
                err = db.update_profile(user["id"], name, student_id, major, year, email)
                if err:
                    st.error(err)
                else:
                    st.success("Profile saved.")
                    st.rerun()


# --------------------------------------------------------------------------
# app entry
# --------------------------------------------------------------------------
setup()
user = current_user()
if st.session_state.get("user_id") and not user:  # account removed: drop the stale login
    st.session_state.pop("user_id", None)

PAGES_MEMBER = ["Home", "My RSVPs", "Create Event", "Predict Attendance", "Profile"]
PAGES_GUEST = ["Home", "Sign in"]
pages = (PAGES_MEMBER + ["Admin Dashboard"] if user["role"] == "admin" else PAGES_MEMBER) if user else PAGES_GUEST

if "_goto" in st.session_state:
    st.session_state["page"] = st.session_state.pop("_goto")
if st.session_state.get("page") not in pages:
    st.session_state["page"] = "Home"

with st.sidebar:
    st.image(CREST_URL, width=64)
    st.markdown("## RUPP Campus Event Hub")
    if user:
        st.write(f"Hello, **{user['name']}**" + (" · Admin" if user["role"] == "admin" else ""))
    st.radio("Navigate", pages, key="page")
    if user:
        st.button("Log out", on_click=lambda: st.session_state.pop("user_id", None))

flash = st.session_state.pop("flash", None)
if flash:
    st.success(flash)

page = st.session_state["page"]
if page == "Home":
    page_home(user)
elif page == "Sign in":
    page_sign_in()
elif page == "My RSVPs":
    page_my_rsvps(user)
elif page == "Create Event":
    page_create(user)
elif page == "Predict Attendance":
    page_predict(user)
elif page == "Profile":
    page_profile(user)
elif page == "Admin Dashboard":
    page_admin(user)
