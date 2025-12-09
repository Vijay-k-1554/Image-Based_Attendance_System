# recognize.py

import os
import pickle
import cv2
import numpy as np
from datetime import datetime
from argparse import ArgumentParser

from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from utils import l2_normalize
import pandas as pd

DB_PATH = os.path.join("db", "face_db.pkl")
OUTPUT_ANNOTATED_DIR = os.path.join("output", "annotated")
UNKNOWN_FACES_DIR = os.path.join("output", "unknown_faces")
ATTENDANCE_LOG_DIR = "attendance_logs"
ATTENDANCE_CSV = os.path.join(ATTENDANCE_LOG_DIR, "attendance.csv")

THRESHOLD = 0.35  # adjust after testing


def init_face_app(ctx_id: int = -1):
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def load_database():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database file not found: {DB_PATH}. Run enroll.py first."
        )

    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)

    embeddings = db["embeddings"]
    names = db["names"]
    return embeddings, names


def ensure_dirs():
    os.makedirs(OUTPUT_ANNOTATED_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(ATTENDANCE_LOG_DIR, exist_ok=True)


def save_attendance(present_names):
    """
    Interactive attendance update:

    - If today's column exists:
        Ask user whether to OVERWRITE or NOT.
    - If overwrite = YES → reset to 'A' and mark today's P.
    - If overwrite = NO  → keep previous values and only set P for detected students.
    """
    ensure_dirs()
    today = datetime.now().strftime("%Y-%m-%d")

    # Load or create CSV
    if os.path.exists(ATTENDANCE_CSV):
        df = pd.read_csv(ATTENDANCE_CSV)
    else:
        df = pd.DataFrame({"student_name": []})

    if "student_name" not in df.columns:
        df = pd.DataFrame({"student_name": []})

    # Ensure detected students exist in CSV
    existing = set(df["student_name"].astype(str))
    new_students = [n for n in present_names if n not in existing]
    if new_students:
        add_df = pd.DataFrame({"student_name": new_students})
        df = pd.concat([df, add_df], ignore_index=True)

    # Sort once
    df = df.drop_duplicates(subset=["student_name"]).sort_values("student_name").reset_index(drop=True)

    # ✅ CASE 1: Today's column already exists → ASK USER
    if today in df.columns:
        print(f"\n⚠️ Attendance for {today} already exists!")
        choice = input("Do you want to overwrite today's attendance? (y/n): ").strip().lower()

        if choice == "y":
            print("[INFO] Overwriting today's attendance...")
            df[today] = "A"  # reset all to absent
        else:
            print("[INFO] Keeping previous attendance. Only adding new 'P' marks.")

    # ✅ CASE 2: Today's column does NOT exist → create it fresh
    else:
        df[today] = "A"

    # ✅ Mark present students as 'P'
    for name in present_names:
        df.loc[df["student_name"] == name, today] = "P"

    df.to_csv(ATTENDANCE_CSV, index=False)
    print(f"\n✅ Attendance updated for {len(present_names)} students on {today}.")


def recognize_group_photo(image_path: str, output_path: str):
    ensure_dirs()

    embeddings, db_names = load_database()
    app = init_face_app(ctx_id=-1)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = app.get(img_rgb)

    print(f"[INFO] Detected {len(faces)} faces in {image_path}")

    present_names = set()
    unknown_counter = 0

    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        h, w = img_bgr.shape[:2]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

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
            if x2 > x1 and y2 > y1:
                unknown_face = img_bgr[y1:y2, x1:x2]
                unknown_path = os.path.join(
                    UNKNOWN_FACES_DIR, f"unknown_{unknown_counter}.jpg"
                )
                cv2.imwrite(unknown_path, unknown_face)
                unknown_counter += 1

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_bgr,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Save annotated image
    cv2.imwrite(output_path, img_bgr)
    print(f"[INFO] Annotated image saved to: {output_path}")

    # 🔹 NEW: show image in a window
    # Optionally resize if it's too big for your screen
    max_width = 1200
    h, w = img_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        img_show = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    else:
        img_show = img_bgr

    cv2.imshow("Attendance - Detected Faces", img_show)
    print("[INFO] Press any key in the image window to close it...")
    cv2.waitKey(0)          # waits for a key press
    cv2.destroyAllWindows() # closes the window

    # Save attendance
    save_attendance(sorted(present_names))

    print("\n=== Present Students ===")
    for n in sorted(present_names):
        print(n)
    print("========================\n")

    return present_names


def parse_args():
    parser = ArgumentParser(description="Group photo attendance (date-wise CSV + display)")
    parser.add_argument("--image", "-i", required=True, help="Path to group photo")
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output path for annotated image (optional)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    img_path = args.image

    if args.out is not None:
        out_path = args.out
    else:
        base = os.path.basename(img_path)
        name, ext = os.path.splitext(base)
        out_path = os.path.join(OUTPUT_ANNOTATED_DIR, f"{name}_annotated{ext}")

    recognize_group_photo(img_path, out_path)
