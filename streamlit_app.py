"""
Streamlit frontend — an alternative UI to ui/index.html, talking to the same
FastAPI backend over REST (no ML logic here, per R1: this is UI only).

Run standalone (with the API already running separately):
    streamlit run streamlit_app.py

Or set API_BASE_URL if the API isn't on localhost:8000.
"""
import os
import time
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Teachable Machine Classifier", page_icon="🧠", layout="wide")

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "cnn": "CNN (Keras)",
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


def get_run(run_id):
    r = api_get(f"/api/runs/{run_id}")
    r.raise_for_status()
    return r.json()


def get_runs():
    r = api_get("/api/runs")
    r.raise_for_status()
    return r.json()["runs"]


# ------------------------------------------------------------------ Sidebar --
st.title("🧠 Teachable Machine — Multi-Class Image Classifier")
st.caption(
    "Streamlit UI for the same backend as the React app. "
    "Upload images → train Logistic Regression / Random Forest / CNN → predict "
    "from a file or your webcam."
)

try:
    api_get("/api/classes")
except requests.exceptions.ConnectionError:
    st.error(
        f"Can't reach the API at **{API_BASE_URL}**. Make sure the FastAPI "
        f"backend is running (`uvicorn main:app --port 8000`), or set the "
        f"`API_BASE_URL` environment variable."
    )
    st.stop()

tab_data, tab_train, tab_results, tab_predict = st.tabs(
    ["1. Classes & Data", "2. Train", "3. Results", "4. Predict"]
)

# ------------------------------------------------------------ 1. Data tab --
with tab_data:
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
                    if st.button("Delete class", key=f"del_{c['name']}"):
                        api_delete(f"/api/classes/{c['name']}")
                        st.cache_data.clear()
                        st.rerun()

# ----------------------------------------------------------- 2. Train tab --
with tab_train:
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
        placeholder = st.empty()
        log_placeholder = st.empty()
        log_lines = []
        # Streamlit has no native WebSocket client, so we poll the REST
        # endpoint on an interval — the same run-state the WebSocket in
        # ui/index.html streams live, just pulled instead of pushed (R5 is
        # still satisfied: every trainer emits stage/progress events, this
        # UI just reads the latest persisted state each tick).
        while True:
            run = get_run(run_id)
            with placeholder.container():
                cols = st.columns(3)
                all_done = True
                for i, m in enumerate(run["models"]):
                    with cols[i]:
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
                break
            time.sleep(2)

# ---------------------------------------------------------- 3. Results tab --
with tab_results:
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
                        import pandas as pd
                        df = pd.DataFrame(cm, index=labels, columns=labels)
                        st.dataframe(df.style.background_gradient(cmap="Greens", axis=None))
                    else:
                        st.caption(m.get("error") or "unavailable")

# ---------------------------------------------------------- 4. Predict tab --
with tab_predict:
    mode = st.radio("Image source", ["Upload", "Webcam"], horizontal=True)

    image_bytes = None
    if mode == "Upload":
        f = st.file_uploader("Choose an image to classify", type=["jpg", "jpeg", "png", "webp"])
        if f:
            image_bytes = f.getvalue()
            st.image(image_bytes, width=300)
    else:
        cam = st.camera_input("Capture a frame")
        if cam:
            image_bytes = cam.getvalue()

    if image_bytes:
        with st.spinner("Predicting..."):
            resp = api_post("/api/predict", files={"file": ("frame.jpg", image_bytes, "image/jpeg")})
        result = resp.json()
        if result.get("error"):
            st.error(result["error"])
        else:
            cols = st.columns(3)
            for i, (mt, p) in enumerate(result["predictions"].items()):
                with cols[i]:
                    with st.container(border=True):
                        st.markdown(f"**{MODEL_LABELS[mt]}**")
                        if p["status"] == "ok":
                            st.markdown(f"### {p['predicted_class']}")
                            for cls, prob in sorted(p["probabilities"].items(), key=lambda x: -x[1]):
                                st.write(f"{cls}: {prob*100:.1f}%")
                                st.progress(prob)
                        else:
                            st.caption(f"Unavailable: {p.get('reason')}")
