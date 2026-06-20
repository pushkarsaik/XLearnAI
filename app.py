import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import random
import os
import time

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="X-LEARN AI",
    page_icon="⚫",
    layout="centered"
)

# =========================================
# MODEL LOAD (cached so it runs only once)
# =========================================

@st.cache_resource
def load_fracture_model():
    return load_model("fracture_model.h5")

# Model is active
model = load_fracture_model()

# =========================================
# DIFFICULTY SETTINGS
# =========================================

DIFFICULTY_SETTINGS = {
    "EASY":   {"total_questions": 5, "timer": 30},
    "MEDIUM": {"total_questions": 5, "timer": 20},
    "HARD":   {"total_questions": 5, "timer": 10},
}

DATASET_PATHS = {
    "fractured":     "dataset/test/fractured",
    "not_fractured": "dataset/test/not_fractured",
}

# =========================================
# SESSION STATE INITIALISATION
# =========================================

DEFAULTS = {
    "page":                "home",
    "score":               0,
    "question_number":     1,
    "quiz_started":        False,
    "quiz_finished":       False,
    "current_image":       None,        # Path string or None
    "actual_class":        None,        # "fractured" | "not_fractured"
    "question_start_time": None,        # float (time.time())
    "answer_submitted":    False,       # Blocks duplicate submits
    "difficulty":          "EASY",      # Persisted across reruns
    "used_images":         [],          # Avoid repeating images
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =========================================
# HELPERS
# =========================================

def pick_random_image(used: list) -> tuple[str, str]:
    """Return (image_path, actual_class) avoiding recently used images."""
    actual_class = random.choice(["fractured", "not_fractured"])
    folder = DATASET_PATHS[actual_class]

    try:
        all_images = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    except FileNotFoundError:
        st.error(f"Dataset folder not found: {folder}")
        st.stop()

    if not all_images:
        st.error(f"No images found in: {folder}")
        st.stop()

    available = [img for img in all_images if img not in used]
    if not available:
        available = all_images      

    chosen = random.choice(available)
    return os.path.join(folder, chosen), actual_class, chosen


def load_next_question():
    """Populate session state with a new random question."""
    path, cls, name = pick_random_image(st.session_state.used_images)
    st.session_state.current_image       = path
    st.session_state.actual_class        = cls
    st.session_state.question_start_time = time.time()
    st.session_state.answer_submitted    = False

    st.session_state.used_images.append(name)
    if len(st.session_state.used_images) > 10:
        st.session_state.used_images.pop(0)


def reset_quiz():
    """Reset all quiz-related state without touching page."""
    st.session_state.score            = 0
    st.session_state.question_number  = 1
    st.session_state.quiz_started     = False
    st.session_state.quiz_finished    = False
    st.session_state.current_image    = None
    st.session_state.actual_class     = None
    st.session_state.question_start_time = None
    st.session_state.answer_submitted = False
    st.session_state.used_images      = []


def get_remaining_time(timer_seconds: int) -> float:
    """Return seconds remaining for the current question."""
    if st.session_state.question_start_time is None:
        return timer_seconds
    elapsed = time.time() - st.session_state.question_start_time
    return max(0.0, timer_seconds - elapsed)

# =========================================
# CUSTOM CSS - NEUMORPHIC DARK & GOLD
# =========================================

st.markdown("""
<style>

/* ---------- APP HIDE ELEMENTS ---------- */
header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ---------- BASE APP ---------- */
.stApp {
    background-color: #17181a; /* Matte dark grey/black */
    color: #8c8c8e; /* Muted subtext */
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    border: 4px solid #c49d6c; /* Elegant 4-sided gold border */
    box-sizing: border-box;
}

hr {
    border-color: #212225;
}

/* ---------- TYPOGRAPHY ---------- */
.main-title {
    text-align: center;
    font-size: 56px;
    font-weight: 700; 
    color: #c49d6c !important; 
    margin-top: 80px; /* Pushed down since the logo is removed */
    margin-bottom: 10px;
    letter-spacing: normal; 
    text-shadow: 2px 2px 4px rgba(0,0,0,0.4); 
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #7b7d82;
    margin-bottom: 50px;
    font-weight: 300;
    letter-spacing: 1px;
}

/* ---------- BUTTONS (NEUMORPHIC) ---------- */
.stButton > button {
    height: 60px;
    font-size: 14px;
    font-weight: 500;
    border-radius: 30px !important; /* Soft pill shape */
    border: none !important;
    background-color: #17181a !important;
    color: #c49d6c !important;
    letter-spacing: 1.5px;
    transition: all 0.2s ease-in-out;
    /* Soft outset shadow for raised effect */
    box-shadow: 6px 6px 14px #0f0f11, -6px -6px 14px #1f2123 !important;
}

.stButton > button:hover, .stButton > button:active {
    /* Soft inset shadow for pressed effect */
    box-shadow: inset 6px 6px 14px #0f0f11, inset -6px -6px 14px #1f2123 !important;
    color: #dfb683 !important; /* Slightly brighter gold */
    transform: scale(0.98);
}

/* ---------- FEATURE CARDS (HOME PAGE) ---------- */
.feature-wrapper {
    display: flex;
    justify-content: space-between;
    gap: 20px;
    margin-top: 60px;
    margin-bottom: 40px;
}

.feature-card {
    background: #17181a;
    border-radius: 20px;
    padding: 30px 20px;
    text-align: center;
    flex: 1;
    /* Outset shadow matching the theme */
    box-shadow: 6px 6px 14px #0f0f11, -6px -6px 14px #1f2123;
    transition: transform 0.3s ease;
}

.feature-card:hover {
    transform: translateY(-5px);
}

.feature-icon {
    font-size: 32px;
    margin-bottom: 15px;
    opacity: 0.9;
}

.feature-title {
    font-size: 16px;
    font-weight: 500;
    color: #c49d6c; /* Gold */
    margin-bottom: 12px;
    letter-spacing: 1px;
}

.feature-text {
    font-size: 13px;
    color: #7b7d82;
    line-height: 1.6;
}

/* ---------- QUIZ SPECIFIC ---------- */
.quiz-heading {
    font-size: 32px;
    font-weight: 600;
    color: #c49d6c;
    letter-spacing: 1px;
}

.score-badge {
    display: inline-block;
    background: #17181a;
    color: #c49d6c;
    border-radius: 20px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 1px;
    margin-bottom: 24px;
    /* Inset to look like a screen display */
    box-shadow: inset 4px 4px 8px #0f0f11, inset -4px -4px 8px #1f2123;
}

/* ---------- CARDS ---------- */
.final-card {
    background: #17181a;
    border-radius: 30px;
    padding: 50px;
    text-align: center;
    margin-top: 30px;
    margin-bottom: 40px;
    /* Outset card shadow */
    box-shadow: 8px 8px 20px #0f0f11, -8px -8px 20px #1f2123;
}

.final-score-number {
    font-size: 80px;
    font-weight: 500;
    color: #c49d6c;
    line-height: 1;
}

.final-score-label {
    font-size: 14px;
    color: #7b7d82;
    margin-top: 15px;
    letter-spacing: 1px;
}

.final-grade {
    font-size: 22px;
    font-weight: 500;
    color: #c49d6c;
    margin-top: 25px;
    letter-spacing: 1px;
}

/* ---------- IMAGE & INPUTS ---------- */
img {
    border-radius: 20px;
    /* Subtle framing */
    padding: 8px;
    background: #17181a;
    box-shadow: 5px 5px 12px #0f0f11, -5px -5px 12px #1f2123;
}

div[role="radiogroup"] {
    padding: 20px;
    background: #17181a;
    border-radius: 20px;
    margin-top: 10px;
    /* Inset well for radio buttons */
    box-shadow: inset 5px 5px 10px #0f0f11, inset -5px -5px 10px #1f2123;
}

/* Override radio text colors */
.stRadio > label {
    color: #7b7d82 !important;
    font-weight: 400;
}
</style>
""", unsafe_allow_html=True)


# =========================================
# HOME PAGE
# =========================================

if st.session_state.page == "home":

    st.markdown('<div class="main-title">X-Learn AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Focussing on the power of clinical diagnosis.</div>',
        unsafe_allow_html=True,
    )

    st.write("") 

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Doubt Solver", use_container_width=True):
            st.session_state.page = "doubt_solver"
            st.rerun()

    with col2:
        if st.button("Quiz Mode", use_container_width=True):
            st.session_state.page = "quiz_mode"
            st.rerun()

    # --- FEATURE SECTION ---
    st.markdown("""
    <div class="feature-wrapper">
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Rapid Analysis</div>
            <div class="feature-text">Get instant AI-driven diagnostic insights within seconds to aid your clinical workflow.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">High Precision</div>
            <div class="feature-text">Trained on thousands of diverse radiographs to ensure robust and reliable abnormality detection.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Interactive Training</div>
            <div class="feature-text">Test your own diagnostic skills against the AI in timed, gamified training modules.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================
# DOUBT SOLVER
# =========================================

elif st.session_state.page == "doubt_solver":

    st.markdown("<h2 class='quiz-heading'>Doubt Solver</h2>", unsafe_allow_html=True)
    st.write("")

    if st.button("Return"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("---")
    st.write("Upload radiograph to analyze structure.")

    uploaded_file = st.file_uploader(
        "Select File", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_container_width=True)

        img_resized  = img.resize((128, 128))
        img_array    = image.img_to_array(img_resized)
        img_array    = np.expand_dims(img_array, axis=0) / 255.0

        prediction   = model.predict(img_array)
        confidence   = float(prediction[0][0])

        st.markdown("---")

        if confidence > 0.5:
            st.markdown("<h3 style='color:#c49d6c;'>Result: Normal Structure</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:#c49d6c;'>Result: Abnormality Detected</h3>", unsafe_allow_html=True)

# =========================================
# QUIZ MODE
# =========================================

elif st.session_state.page == "quiz_mode":

    # ---- Top bar ----
    col_back, col_title = st.columns([1, 4])

    with col_back:
        if st.button("Return"):
            reset_quiz()
            st.session_state.page = "home"
            st.rerun()

    with col_title:
        st.markdown(
            "<h1 class='quiz-heading' style='margin-top:-5px;'>Quiz Mode</h1>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ---- Difficulty selector (only when quiz is NOT running) ----
    if not st.session_state.quiz_started:

        difficulty = st.selectbox(
            "Select Parameters",
            list(DIFFICULTY_SETTINGS.keys()),
            index=list(DIFFICULTY_SETTINGS.keys()).index(
                st.session_state.difficulty
            ),
        )
        st.session_state.difficulty = difficulty

        settings        = DIFFICULTY_SETTINGS[difficulty]
        total_questions = settings["total_questions"]
        timer_seconds   = settings["timer"]

        st.write(f"Mode: **{difficulty}** | Subjects: **{total_questions}** | Time: **{timer_seconds}s**")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Initialize Sequence", use_container_width=True):
            reset_quiz()
            st.session_state.quiz_started  = True
            st.session_state.difficulty    = difficulty
            load_next_question()
            st.rerun()

    # ---- Quiz running ----
    elif st.session_state.quiz_started and not st.session_state.quiz_finished:

        settings        = DIFFICULTY_SETTINGS[st.session_state.difficulty]
        total_questions = settings["total_questions"]
        timer_seconds   = settings["timer"]

        remaining = get_remaining_time(timer_seconds)

        # ---- Auto-advance on timeout ----
        if remaining <= 0 and not st.session_state.answer_submitted:
            st.session_state.answer_submitted = True       # No score added (timeout)

            if st.session_state.question_number < total_questions:
                st.session_state.question_number += 1
                load_next_question()
            else:
                st.session_state.quiz_finished = True

            st.rerun()

        # ---- Score badge ----
        st.markdown(
            f'<div class="score-badge">Score: {st.session_state.score} / {total_questions}</div>',
            unsafe_allow_html=True,
        )

        # ---- Question + Timer Row ----
        col_q, col_timer = st.columns([5,2])

        with col_q:
            st.markdown(
                f"<h3 style='color:#7b7d82; font-weight:400;'>Subject {st.session_state.question_number} of {total_questions}</h3>",
                unsafe_allow_html=True
            )

        with col_timer:
            secs_int = int(remaining)
            flash_color = "#c49d6c" if secs_int > 3 else "#7b7d82"
            
            # Neumorphic inset timer panel
            timer_html = f"""
            <div style="
                background: #17181a;
                padding: 12px 0px;
                border-radius: 20px;
                text-align: center;
                width: 100%;
                box-shadow: inset 5px 5px 10px #0f0f11, inset -5px -5px 10px #1f2123;
            ">
                <div style="
                    font-size: 10px;
                    color: #7b7d82;
                    margin-bottom: 2px;
                    letter-spacing: 1px;
                ">
                    TIME
                </div>
                <div style="
                    font-size: 24px;
                    font-weight: 500;
                    color: {flash_color};
                    line-height: 1;
                ">
                    {secs_int}
                </div>
            </div>
            """
            st.components.v1.html(timer_html, height=80)

        # ---- Quiz Image ----
        try:
            quiz_img = Image.open(
                st.session_state.current_image
            ).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            st.error(f"System Error: {e}")
            load_next_question()
            st.rerun()

        st.write("")
        st.image(
            quiz_img,
            width=300
        )

        # ---- Answer radio ----
        user_answer = st.radio(
            "Classification",
            ["Fractured", "Normal"],
            key=f"answer_q{st.session_state.question_number}",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        
        # ---- Submit button ----
        submit_disabled = st.session_state.answer_submitted

        if st.button(
            "Submit Selection",
            use_container_width=True,
            disabled=submit_disabled,
        ):
            if not st.session_state.answer_submitted:
                st.session_state.answer_submitted = True

                class_map = {"Fractured": "fractured", "Normal": "not_fractured"}
                correct = (class_map[user_answer] == st.session_state.actual_class)

                if correct:
                    st.session_state.score += 1
                    st.toast("Result: Correct", icon="✔️")
                else:
                    actual_label = "Fractured" if st.session_state.actual_class == "fractured" else "Normal"
                    st.toast(f"Result: Incorrect. Expected: {actual_label}", icon="✖️")

                time.sleep(1.0)     # Brief pause so toast is visible

                if st.session_state.question_number < total_questions:
                    st.session_state.question_number += 1
                    load_next_question()
                else:
                    st.session_state.quiz_finished = True

                st.rerun()

        # ---- Tick the timer every second ----
        if not st.session_state.answer_submitted and remaining > 0:
            time.sleep(1)
            st.rerun()

    # ---- Quiz finished ----
    elif st.session_state.quiz_finished:

        settings        = DIFFICULTY_SETTINGS[st.session_state.difficulty]
        total_questions = settings["total_questions"]
        score           = st.session_state.score
        pct             = score / total_questions if total_questions else 0

        if pct == 1.0:
            grade = "Flawless"
        elif pct >= 0.67:
            grade = "Acceptable"
        elif pct >= 0.34:
            grade = "Suboptimal"
        else:
            grade = "Requires Review"

        st.markdown(
            f"""
            <div class="final-card">
                <div class="final-score-number">{score}</div>
                <div class="final-score-label">Out of {total_questions}</div>
                <div class="final-grade">{grade}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_restart, col_home = st.columns(2)

        with col_restart:
            if st.button("Restart", use_container_width=True):
                reset_quiz()
                st.session_state.difficulty = st.session_state.get("difficulty", "EASY")
                st.rerun()

        with col_home:
            if st.button("End Session", use_container_width=True):
                reset_quiz()
                st.session_state.page = "home"
                st.rerun()