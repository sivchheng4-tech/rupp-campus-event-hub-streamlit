# 🎓 RUPP Campus Event Hub (Streamlit)

The whole project in **Python + Streamlit only**: event hub, login, RSVPs, reviews, event creation,
and the **attendance predictor** (raw data → cleaning → model → prediction).

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run frontend/app.py
```
Open http://localhost:8501. The first time you open **Predict Attendance**, the model trains itself (a few seconds).
Demo admin: `admin@rupp.edu.kh` / `admin123`.

## What changed from the Next.js version
| Next.js / React | Streamlit |
|---|---|
| `components/event-hub.tsx`, `app/page.tsx` | `frontend/app.py` (Home, My RSVPs, Create Event, Profile pages) |
| `app/predict/page.tsx` + FastAPI `backend_fastapi/` + `lib/api.ts` | Predict page in `frontend/app.py` calls `ml/service.py` directly (no API server) |
| Clerk sign in / sign up, admin role | Built-in accounts in `backend/db.py` (PBKDF2 password hashing, `role` column for admin) |
| `localStorage` (events, RSVPs, profile) | SQLite file `campus_hub.db` (created on first run) |
| Tailwind / shadcn styling | `.streamlit/config.toml` theme + a little CSS in `frontend/app.py` |
| `ml/` pipeline, `data/`, `tests/` | Kept as is |

## Features
Browse + search + category filter · event details dialog · RSVP with seat counts · reviews and star ratings ·
My RSVPs with next-event reminder · create events · profile editing · admin can delete events and retrain the model ·
attendance predictor with metrics, charts and prediction history ·
**Admin Dashboard** (admins only): account and sign-in counts, sign-ins per day, user list with last sign-in,
event management and an activity log of who created/deleted what.

## Files
```
frontend/
  app.py        Streamlit pages (the UI only)
backend/
  db.py         database + login + RSVPs + reviews + admin reports (no Streamlit code, easy to test)
ml/             cleaning, features, training, prediction (config.py, service.py, train.py ...)
data/raw/       events.csv (sample data)
models/         trained model + metrics (created on first run, git-ignored)
notebooks/      experiments (empty for now)
reports/        generated figures (git-ignored)
tests/          pytest tests for the backend and the ML pipeline
.streamlit/     theme (config.toml)
campus_hub.db   SQLite file, created on first run (git-ignored)
```
How the pieces connect: `frontend/app.py` -> `backend/db.py` (users, events, RSVPs, reviews)
and `frontend/app.py` -> `ml/service.py` (attendance predictions). There is no API server in between,
so it deploys as one Streamlit app.

## Tests
```bash
pytest
```

## Push to GitHub
```bash
git init
git add .
git commit -m "Campus Event Hub: Streamlit version"
git branch -M main
git remote add origin https://github.com/<your-username>/rupp-campus-event-hub.git
git push -u origin main
```
Never commit `.env.local` or `.streamlit/secrets.toml` (both are ignored).

## Notes
- `data/raw/events.csv` is a regenerated 700-row **synthetic** sample (`python -m ml.data.make_raw`).
  The file in the export had only 10 rows, so it is kept as `events_export_10rows.csv`.
  Put your real event data in `events.csv` (same columns) and run `python -m ml.data.make_dataset` then `python -m ml.train`.
- On Streamlit Community Cloud the SQLite file resets when the app restarts; use a hosted database for real deployment.
