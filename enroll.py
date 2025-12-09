# enroll.py

import os
import pickle
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from utils import l2_normalize

ENROLL_DIR = os.path.join("data", "enroll")
DB_DIR = "db"
DB_PATH = os.path.join(DB_DIR, "face_db.pkl")

ATTENDANCE_LOG_DIR = "attendance_logs"
ATTENDANCE_CSV = os.path.join(ATTENDANCE_LOG_DIR, "attendance.csv")


def init_face_app(ctx_id: int = -1):
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def update_attendance_csv(names):
    """
    Ensure attendance.csv exists with:
    column 0: student_name
    other columns: date-wise attendance
    Only updates/creates the student_name column and rows.
    """
    os.makedirs(ATTENDANCE_LOG_DIR, exist_ok=True)

    names = sorted(set(names))  # unique & sorted

    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)

        if "student_name" not in df.columns:
            # If old format, recreate from scratch
            df = pd.DataFrame({"student_name": names})
        else:
            # Add new students that are not yet in CSV
            existing = set(df["student_name"].astype(str))
            new_students = [n for n in names if n not in existing]

            if new_students:
                add_df = pd.DataFrame({"student_name": new_students})
                df = pd.concat([df, add_df], ignore_index=True)

        # Drop duplicates just in case
        df = df.drop_duplicates(subset=["student_name"]).reset_index(drop=True)
    else:
        # Fresh CSV: only student_name column for now
        df = pd.DataFrame({"student_name": names})

    df.to_csv(ATTENDANCE_CSV, index=False)
    print(f"[INFO] attendance.csv updated with {len(names)} students.")


def enroll_students():
    if not os.path.isdir(ENROLL_DIR):
        raise FileNotFoundError(f"Enrollment directory not found: {ENROLL_DIR}")

    os.makedirs(DB_DIR, exist_ok=True)

    app = init_face_app(ctx_id=-1)  # CPU

    embeddings = []
    names = []

    for student_folder in os.listdir(ENROLL_DIR):
        student_path = os.path.join(ENROLL_DIR, student_folder)
        if not os.path.isdir(student_path):
            continue

        student_name = student_folder  # folder name = student name
        student_embs = []

        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)

            if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            faces = app.get(img_rgb)

            if len(faces) == 0:
                print(f"[WARN] No face detected in: {img_path}")
                continue

            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )
            face = faces[0]

            emb = face.embedding.astype(np.float32)
            emb = l2_normalize(emb)
            student_embs.append(emb)

        if len(student_embs) == 0:
            print(f"[WARN] No valid faces for: {student_name}")
            continue

        mean_emb = np.mean(student_embs, axis=0)
        mean_emb = l2_normalize(mean_emb)

        embeddings.append(mean_emb)
        names.append(student_name)

        print(f"[INFO] Enrolled {student_name} with {len(student_embs)} images.")

    if len(embeddings) == 0:
        raise RuntimeError("No students enrolled. Check images in data/enroll.")

    embeddings = np.vstack(embeddings)

    db = {
        "embeddings": embeddings,
        "names": names,
    }

    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

    print(f"\n[INFO] Enrollment complete. Database saved to: {DB_PATH}")

    # NEW: update master attendance CSV with student list
    update_attendance_csv(names)


def main():
    enroll_students()

if __name__ == "__main__":
    main()

