"""
Streamlit frontend -- an alternative UI to ui/index.html, talking to the same
FastAPI backend over REST (no ML logic here, per R1: this is UI only).

Run standalone (with the API already running separately):
    streamlit run streamlit_app.py

Or set API_BASE_URL if the API isn't on localhost:8000.
"""
import os
import time
import requests
import streamlit as st
import pandas as pd

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Teachable Machine Classifier", page_icon="\U0001F9E0", layout="wide")

st.markdown("""
<style>
:root {
  --bg: #0d0f14; --panel: #171b23; --panel2: #1e2330; --panel3: #262c3b;
  --border: #2a3040; --border-strong: #3a4258;
  --text: #eef0f4; --text-secondary: #c3c9d6; --muted: #8b93a7;
  --accent: #7c9eff; --accent2: #6fd7a3; --danger: #ff6b6b; --warn: #ffb84d;
  --accent-soft: rgba(124,158,255,0.13); --accent2-soft: rgba(111,215,163,0.13);
  --danger-soft: rgba(255,107,107,0.13); --warn-soft: rgba(255,184,77,0.13);
  --r-sm: 6px; --r-md: 10px; --r-lg: 14px;
}
[data-testid="stMetric"] {
  background: var(--panel); border: 1px solid var(--border); border-radius: var(--r-lg);
  padding: 16px; transition: transform 150ms ease, border-color 150ms ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: var(--border-strong); }
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: var(--r-lg) !important; transition: border-color 150ms ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: var(--border-strong) !important; }
[data-testid="stButton"] button {
  border-radius: var(--r-sm) !important; font-weight: 600 !important;
  transition: transform 100ms ease !important;
}
[data-testid="stButton"] button:active { transform: translateY(1px); }
[data-testid="stAlert"] { border-radius: var(--r-md) !important; }
[data-testid="stFileUploaderDropzone"] {
  background: var(--panel2) !important; border: 2px dashed var(--border) !important;
  border-radius: var(--r-md) !important; transition: border-color 150ms ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; }
[data-testid="stDataFrame"] { border-radius: var(--r-md); overflow: hidden; }
h1, h2, h3 { letter-spacing: -0.01em; }
</style>
""", unsafe_allow_html=True)

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "cnn": "CNN (trained from scratch)",
}


# --------------------------------------------------------------- API calls --
def api_get(path, **kwargs):
    return requests.get(f"{API_BASE_URL}{path}", timeout=30, **kwargs)


def api_post(path, **kwargs):
    return requests.post(f"{API_BASE_URL}{path}", timeout=60, **kwargs)


def api_delete(path, **kwargs):
    return requests.delete(f"{API_BASE_URL}{path}", timeout=30, **kwargs)


@st.cache_data(ttl=2)
def get_classes():
    r = api_get("/api/classes")
    r.raise_for_status()
    return r.json()["classes"]


@st.cache_data(ttl=2)
def get_dataset_status():
    r = api_get("/api/dataset/status")
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=2)
def get_dashboard():
    r = api_get("/api/dashboard")
    r.raise_for_status()
    return r.json()


def get_run(run_id):
    r = api_get(f"/api/runs/{run_id}")
    r.raise_for_status()
    return r.json()


def get_runs():
    r = api_get("/api/runs")
    r.raise_for_status()
    return r.json()["runs"]


# ------------------------------------------------------------------ Sidebar --
try:
    api_get("/api/classes")
except requests.exceptions.ConnectionError:
    st.error(
        f"Can't reach the API at **{API_BASE_URL}**. Make sure the FastAPI "
        f"backend is running (`uvicorn main:app --port 8000`), or set the "
        f"`API_BASE_URL` environment variable."
    )
    st.stop()

with st.sidebar:
    st.markdown("### \U0001F9E0 Teachable Machine")
    st.caption("Multi-Class Image Classifier")
    # Bug fix: the "Go train a model" button on the Dashboard used to set
    # st.session_state["_nav_hint"] and rerun, but nothing ever read that
    # key -- the radio below always fell back to its default ("Dashboard"),
    # so the button silently did nothing. Giving the radio an explicit
    # `key` and having the button set that same session_state key directly
    # (before the widget is created) makes the navigation actually work.
    page = st.radio(
        "Navigate",
        ["Dashboard", "Classes & Data", "Train", "Results", "Predict"],
        label_visibility="collapsed",
        key="_nav_radio",
    )
    st.divider()
    quick_status = get_dataset_status()
    if quick_status["ready_to_train"]:
        st.success("Dataset ready to train")
    else:
        st.warning("Dataset not ready")
    st.caption(f"API: {API_BASE_URL}")

st.title(page)

# ------------------------------------------------------------- Dashboard --
if page == "Dashboard":
    st.caption("An overview of your dataset and latest training run.")
    dash = get_dashboard()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classes", dash["total_classes"])
    c2.metric("Images uploaded", dash["total_images"])
    c3.metric("Training runs", dash["total_runs"])
    c4.metric("Completed runs", dash["completed_runs"])

    st.subheader("\U0001F4CA Images per class")
    with st.container(border=True):
        if dash["class_counts"]:
            df = pd.DataFrame(
                {"class": list(dash["class_counts"].keys()), "images": list(dash["class_counts"].values())}
            ).set_index("class")
            st.bar_chart(df)
        else:
            st.caption("No classes yet.")

    st.subheader("\U0001F3C1 Latest training run")
    with st.container(border=True):
        if not dash["latest_run"]:
            st.caption("No completed runs yet.")
            if st.button("Go train a model"):
                st.session_state["_nav_radio"] = "Train"
                st.rerun()
        else:
            acc_rows = {
                MODEL_LABELS[m["model_type"]]: (m["accuracy"] or 0) * 100
                for m in dash["latest_run"]["models"] if m["status"] == "done"
            }
            if acc_rows:
                df = pd.DataFrame({"model": list(acc_rows.keys()), "accuracy %": list(acc_rows.values())}).set_index("model")
                st.bar_chart(df)
            for m in dash["latest_run"]["models"]:
                if m["status"] != "done":
                    st.caption(f"{MODEL_LABELS[m['model_type']]}: {m['status']}")

# ------------------------------------------------------------ Classes & Data --
elif page == "Classes & Data":
    status = get_dataset_status()
    if status["ready_to_train"]:
        total = sum(status["class_counts"].values())
        st.success(f"Dataset ready — {len(status['class_counts'])} classes, {total} images total.")
    else:
        st.warning(f"Not ready: {status['reason']}")

    with st.form("add_class_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        new_name = col1.text_input("New class name", placeholder="e.g. cat", label_visibility="collapsed")
        submitted = col2.form_submit_button("+ Add class", use_container_width=True)
        if submitted and new_name.strip():
            api_post("/api/classes", data={"name": new_name.strip()})
            st.cache_data.clear()
            st.rerun()

    classes = get_classes()
    if not classes:
        st.info("No classes yet — add at least 2 to start training.")
    else:
        min_per_class = status["min_images_per_class"]
        cols = st.columns(min(3, len(classes)))
        for i, c in enumerate(classes):
            with cols[i % len(cols)]:
                with st.container(border=True):
                    ok = c["image_count"] >= min_per_class
                    st.markdown(
                        f"**{c['name']}**  \n"
                        f"{'✅' if ok else '⚠️'} {c['image_count']}/{min_per_class} images"
                    )
                    uploaded = st.file_uploader(
                        f"Upload images for '{c['name']}'",
                        type=["jpg", "jpeg", "png", "webp"],
                        accept_multiple_files=True,
                        key=f"upload_{c['name']}",
                        label_visibility="collapsed",
                    )
                    if uploaded:
                        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded]
                        resp = api_post(f"/api/classes/{c['name']}/images", files=files)
                        result = resp.json()
                        if result.get("rejected"):
                            for r in result["rejected"]:
                                st.error(f"{r['filename']}: {r['reason']}")
                        if result.get("accepted"):
                            st.success(f"Added {len(result['accepted'])} image(s).")
                            st.cache_data.clear()
                            st.rerun()
                    if st.button("\U0001F5D1️ Delete class", key=f"del_{c['name']}", help="Delete this class"):
                        api_delete(f"/api/classes/{c['name']}")
                        st.cache_data.clear()
                        st.rerun()

# ----------------------------------------------------------------- Train --
elif page == "Train":
    status = get_dataset_status()
    st.write("Runs Logistic Regression, Random Forest, and CNN together (R2). Progress updates live.")
    disabled = not status["ready_to_train"]
    if disabled:
        st.caption(f"Training disabled: {status['reason']}")

    if st.button("Train all models", type="primary", disabled=disabled):
        resp = api_post("/api/train")
        if resp.status_code != 200:
            st.error(resp.json().get("detail", "Could not start training"))
        else:
            run_id = resp.json()["run_id"]
            st.session_state["training_run_id"] = run_id
            st.session_state["training_active"] = True

    run_id = st.session_state.get("training_run_id")
    if run_id and st.session_state.get("training_active"):
        # Bug fix: this used to be `while True: ...; time.sleep(2)` inside
        # a single script execution. In Streamlit that blocks the entire
        # session for the whole loop's duration -- no other widget on any
        # page (sidebar nav included) can respond until the loop finally
        # breaks, because Streamlit only processes one script run at a
        # time per session. Doing exactly one status check per run and
        # ending with an explicit st.rerun() is the standard non-blocking
        # polling pattern: each rerun is its own fresh, short script
        # execution, so the app stays responsive between ticks.
        run = get_run(run_id)
        cols = st.columns(3)
        all_done = True
        for i, m in enumerate(run["models"]):
            with cols[i], st.container(border=True):
                label = MODEL_LABELS[m["model_type"]]
                badge = {"pending": "⏳", "training": "🔄", "done": "✅", "failed": "❌"}[m["status"]]
                st.markdown(f"**{label}** {badge} `{m['status']}`")
                if m["status"] not in ("done", "failed"):
                    all_done = False
                if m["status"] == "done":
                    st.progress(1.0)
                    st.caption(f"Held-out accuracy: {m['accuracy']*100:.1f}%")
                elif m["status"] == "training":
                    st.progress(0.5)
                elif m["status"] == "failed":
                    st.progress(0.0)
                    st.caption(f"Error: {m['error']}")
                else:
                    st.progress(0.0)

        if all_done:
            st.session_state["training_active"] = False
            st.cache_data.clear()
            st.success("Training complete.")
        else:
            time.sleep(2)
            st.rerun()

# --------------------------------------------------------------- Results --
elif page == "Results":
    runs = get_runs()
    if not runs:
        st.info("No training runs yet.")
    else:
        options = {f"{r['id']} — {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r['started_at']))} ({r['status']})": r["id"] for r in runs}
        selected_label = st.selectbox("Run", list(options.keys()))
        selected_run_id = options[selected_label]
        detail = get_run(selected_run_id)

        cols = st.columns(3)
        for i, m in enumerate(detail["models"]):
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"**{MODEL_LABELS[m['model_type']]}**")
                    if m["status"] == "done":
                        st.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
                        labels = m["labels"]
                        cm = m["confusion_matrix"]
                        df = pd.DataFrame(cm, index=labels, columns=labels)
                        st.dataframe(df.style.background_gradient(cmap="Greens", axis=None))
                    else:
                        st.caption(m.get("error") or "unavailable")

# -------------------------------------------------------------- Predict --
elif page == "Predict":
    input_col, results_col = st.columns([1, 2])

    with input_col:
        mode = st.radio("Image source", ["Upload", "Webcam"], horizontal=True, label_visibility="collapsed")

        image_bytes = None
        if mode == "Upload":
            f = st.file_uploader("Choose an image to classify", type=["jpg", "jpeg", "png", "webp"])
            if f:
                image_bytes = f.getvalue()
                with st.container(border=True):
                    st.image(image_bytes, use_container_width=True)
        else:
            cam = st.camera_input("Capture a frame")
            if cam:
                image_bytes = cam.getvalue()

    with results_col:
        if image_bytes:
            with st.spinner("Predicting..."):
                resp = api_post("/api/predict", files={"file": ("frame.jpg", image_bytes, "image/jpeg")})
            result = resp.json()
            if result.get("error"):
                st.error(result["error"])
            else:
                predictions = list(result["predictions"].items())
                ok = [(mt, p) for mt, p in predictions if p["status"] == "ok"]
                winner_mt = None
                if ok:
                    winner_mt = max(ok, key=lambda kv: max(kv[1]["probabilities"].values()))[0]
                    st.success(f"\U0001F3C6 Highest confidence: **{MODEL_LABELS[winner_mt]}**")
                    predictions.sort(key=lambda kv: 0 if kv[0] == winner_mt else 1)

                cols = st.columns(3)
                for i, (mt, p) in enumerate(predictions):
                    with cols[i]:
                        with st.container(border=True):
                            title = f"**{MODEL_LABELS[mt]}**"
                            if mt == winner_mt:
                                title += " \U0001F3C6"
                            st.markdown(title)
                            if p["status"] == "ok":
                                st.markdown(f"### {p['predicted_class']}")
                                for j, (cls, prob) in enumerate(sorted(p["probabilities"].items(), key=lambda x: -x[1])):
                                    label = f"**{cls}: {prob*100:.1f}%**" if j == 0 else f"{cls}: {prob*100:.1f}%"
                                    st.write(label)
                                    st.progress(prob)
                            else:
                                st.caption(f"Unavailable: {p.get('reason')}")
        else:
            st.caption("Choose an image or capture a webcam frame to see predictions here.")
