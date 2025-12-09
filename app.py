# app.py  - FINAL PROFESSIONAL VERSION
# ✅ USER SIGNUP + LOGIN (Stored in DB)
# ✅ TAKE ATTENDANCE
# ✅ ADD NEW STUDENT
# ✅ ATTENDANCE SHEET
# ✅ ATTENDANCE ANALYTICS (PERCENTAGE)
# ✅ LOGOUT SYSTEM

import os
import pickle
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

from utils import l2_normalize
import enroll

# ✅ AUTH SYSTEM
from auth import init_auth_db, create_user, verify_user
init_auth_db()


# ------------------ PATH CONFIG ------------------
DB_PATH = os.path.join("db", "face_db.pkl")
ENROLL_DIR = os.path.join("data", "enroll")
ATTENDANCE_LOG_DIR = "attendance_logs"
ATTENDANCE_CSV = os.path.join(ATTENDANCE_LOG_DIR, "attendance.csv")
UNKNOWN_FACES_DIR = os.path.join("output", "unknown_faces")

THRESHOLD = 0.35


# ------------------ CACHE MODELS ------------------

@st.cache_resource
def load_face_app():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


@st.cache_resource
def load_database():
    if not os.path.exists(DB_PATH):
        return None, None

    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)

    return db["embeddings"], db["names"]


def clear_db_cache():
    load_database.clear()


# ------------------ ATTENDANCE CSV UPDATE ------------------

def ensure_dirs():
    os.makedirs(ATTENDANCE_LOG_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)


def update_attendance_csv(present_names, mode="append"):
    ensure_dirs()
    today = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)
    else:
        df = pd.DataFrame({"student_name": []})

    df = df.drop_duplicates(subset=["student_name"]).reset_index(drop=True)

    existing = set(df["student_name"].astype(str))
    new_students = [n for n in present_names if n not in existing]
    if new_students:
        df = pd.concat(
            [df, pd.DataFrame({"student_name": new_students})],
            ignore_index=True,
        )

    if today not in df.columns:
        df[today] = "A"

    if mode == "overwrite":
        df[today] = "A"

    for name in present_names:
        df.loc[df["student_name"] == name, today] = "P"

    df = df.sort_values("student_name").reset_index(drop=True)
    df.to_csv(ATTENDANCE_CSV, index=False)
    return df


# ------------------ ATTENDANCE ANALYTICS ------------------

def calculate_attendance_percentage(df):
    if "student_name" not in df.columns or len(df.columns) == 1:
        return None

    date_columns = df.columns[1:]
    total_days = len(date_columns)

    summary = []

    for _, row in df.iterrows():
        name = row["student_name"]
        present_count = (row[date_columns] == "P").sum()
        percentage = (present_count / total_days * 100) if total_days > 0 else 0

        summary.append({
            "Student Name": name,
            "Days Present": present_count,
            "Total Days": total_days,
            "Attendance %": round(percentage, 2)
        })

    return pd.DataFrame(summary)


# ------------------ FACE RECOGNITION ------------------

def run_recognition_on_image(image_bgr, overwrite_mode):
    embeddings, db_names = load_database()
    if embeddings is None:
        st.error("Face database not found. Add students first.")
        st.stop()

    app = load_face_app()
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = app.get(img_rgb)

    present_names = set()
    unknown_counter = 0

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = image_bgr.shape[:2]

        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)

        emb = l2_normalize(face.embedding.astype(np.float32))
        sims = cosine_similarity(emb.reshape(1, -1), embeddings)[0]

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= THRESHOLD:
            name = db_names[best_idx]
            label_text = f"{name} ({best_sim:.2f})"
            present_names.add(name)
        else:
            label_text = "Unknown"
            unknown_face = image_bgr[y1:y2, x1:x2]
            cv2.imwrite(
                os.path.join(UNKNOWN_FACES_DIR, f"unknown_{unknown_counter}.jpg"),
                unknown_face,
            )
            unknown_counter += 1

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_bgr,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    df = update_attendance_csv(sorted(present_names), mode=overwrite_mode)
    return image_bgr, sorted(present_names), df, len(faces)


# ------------------ LOGIN & SIGNUP PAGE ------------------

def login_page():
    st.title("🔐 Institute Login System")

    tab1, tab2 = st.tabs(["Login", "Create Account"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if verify_user(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success("Login successful ✅")
                st.rerun()
            else:
                st.error("Invalid login credentials ❌")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            if not new_user or not new_pass:
                st.warning("Fill all fields")
            else:
                success = create_user(new_user, new_pass)
                if success:
                    st.success("Account created successfully ✅")
                else:
                    st.error("Username already exists ❌")


# ------------------ MAIN APP ------------------

def main():
    st.set_page_config(page_title="Attendance System", layout="wide")
    st.title("📸 Group Photo Based Attendance System")

    st.sidebar.write(f"👤 Logged in as: {st.session_state['username']}")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📷 Take Attendance", "📊 Attendance Sheet", "➕ Add New Student", "📈 Attendance Analytics"]
    )

    # ---------- TAB 1 ----------
    with tab1:
        overwrite_choice = st.radio(
            "If today's attendance already exists:",
            ("Append", "Overwrite"),
            index=0,
        )
        mode = "append" if overwrite_choice == "Append" else "overwrite"

        uploaded_file = st.file_uploader(
            "Upload a group photo", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            annotated_bgr, present_names, df_updated, face_count = run_recognition_on_image(
                image_bgr, overwrite_mode=mode
            )

            st.success(f"Detected {face_count} faces")
            st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.write("### ✅ Present Students")
            st.write(", ".join(present_names) if present_names else "None")
            st.write("### 📄 Updated Attendance Preview")
            st.dataframe(df_updated)

    # ---------- TAB 2 ----------
    with tab2:
        if os.path.exists(ATTENDANCE_CSV):
            df = pd.read_csv(ATTENDANCE_CSV)
            st.dataframe(df)
        else:
            st.info("No attendance file yet.")

    # ---------- TAB 3 ----------
    with tab3:
        st.subheader("➕ Add New Student to Database")

        new_name = st.text_input("Enter Student Name")

        new_images = st.file_uploader(
            "Upload Student Face Images (2–10 recommended)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if st.button("✅ Add Student"):
            if not new_name.strip():
                st.warning("Please enter a student name.")
            elif not new_images:
                st.warning("Please upload at least one image.")
            else:
                student_folder = os.path.join(ENROLL_DIR, new_name.strip())
                os.makedirs(student_folder, exist_ok=True)

                for img in new_images:
                    img_bytes = img.read()
                    img_path = os.path.join(student_folder, img.name)
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

                st.success(f"{len(new_images)} images saved for {new_name}")

                with st.spinner("Updating face database..."):
                    enroll.main()
                    clear_db_cache()

                st.success("✅ Face database updated successfully!")

    # ---------- TAB 4 ----------
    with tab4:
        st.subheader("📈 Student Attendance Percentage")

        if os.path.exists(ATTENDANCE_CSV):
            df = pd.read_csv(ATTENDANCE_CSV)
            report_df = calculate_attendance_percentage(df)

            if report_df is not None:
                st.dataframe(report_df, use_container_width=True)
                st.bar_chart(report_df.set_index("Student Name")["Attendance %"])
        else:
            st.info("No attendance data available yet.")


# ------------------ LOGIN PROTECTION ------------------

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    main()
else:
    login_page()
