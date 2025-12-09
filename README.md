📸 Image-Based Attendance System using Face Recognition

(Group Photo Based Attendance System)

A smart automated attendance system that identifies students from a single group photo using deep learning face recognition (InsightFace + ArcFace) and marks attendance in real time using a Streamlit web interface.

This project eliminates manual attendance, reduces proxy attendance, and improves accuracy using AI-based biometric verification.

🚀 Key Features

✅ Group photo face detection & recognition
✅ Automatic attendance marking
✅ Deep learning–based face embeddings (ArcFace)
✅ Fast and accurate face detection using RetinaFace (via InsightFace)
✅ Live Streamlit web interface
✅ Add new students dynamically from UI
✅ Attendance stored in CSV format
✅ Overwrite / Append attendance option
✅ Secure login system
✅ Deployed using Docker
✅ GitHub + Hugging Face ready

🧠 Technologies Used
Category	Technology
Programming Language	Python 3.10
Web Interface	Streamlit
Face Detection	RetinaFace (via InsightFace)
Face Recognition	ArcFace
Image Processing	OpenCV
Machine Learning	NumPy, Scikit-learn
Authentication	Passlib (Password Hashing)
Database	Pickle + CSV
Deployment	Docker
Version Control	Git & GitHub
🏗️ Project Architecture
Image-Based_Attendance_System/
│
├── app.py                → Main Streamlit application
├── auth.py               → User authentication system
├── enroll.py             → Student face enrollment system
├── utils.py              → Helper functions
├── requirements.txt      → Python dependencies
├── packages.txt          → System packages (optional)
├── Dockerfile            → Docker deployment config
│
├── data/
│   └── enroll/           → Student images (one folder per student)
│
├── db/
│   └── face_db.pkl       → Face embedding database
│
├── attendance_logs/
│   └── attendance.csv   → Attendance record
│
├── output/
│   └── unknown_faces/   → Unknown detected faces
│
└── README.md

🔍 How the System Works (Step-by-Step)
✅ 1. Student Enrollment

Each student has a separate folder inside data/enroll/

Multiple face images are stored per student

enroll.py:

Detects faces

Extracts embeddings using ArcFace

Stores embeddings + student names in face_db.pkl

✅ 2. Attendance Marking

Upload a group photo

Faces detected using RetinaFace

Face embeddings matched with database using cosine similarity

If similarity ≥ threshold → student marked Present (P)

Others marked Absent (A)

Output stored in attendance.csv

✅ 3. Attendance Format
student_name	2025-12-08
Vijay	P
Ansh	A
Harsha	P
✅ 4. Overwrite / Append Option

Append → keeps previous attendance for the day

Overwrite → replaces today’s attendance

✅ 5. Unknown Face Handling

Unrecognized faces are saved in:

output/unknown_faces/

✅ 6. Secure Login System

Passwords stored using PBKDF2 SHA-256 encryption

Prevents unauthorized access

🖥️ Web Interface (Streamlit)

The UI contains:

📷 Take Attendance

📊 Attendance Sheet

➕ Add New Student

📈 Attendance Analytics

🔐 Login System

🐳 Docker Deployment
✅ Build Image
docker build -t attendance-app .

✅ Run Container
docker run -p 7860:7860 attendance-app


Open:

http://localhost:7860

📦 Installation Guide (Local Setup)
✅ 1. Clone Repository
git clone https://github.com/Vijay-k-1554/Image-Based_Attendance_System.git
cd Image-Based_Attendance_System

✅ 2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate

✅ 3. Install Requirements
pip install -r requirements.txt

✅ 4. Run Application
streamlit run app.py

📄 Requirements (requirements.txt)
insightface
onnxruntime
opencv-python
numpy
scikit-learn
streamlit
passlib

🎯 Use Cases

✅ College classroom attendance

✅ Corporate office attendance

✅ Conferences & seminars

✅ Exam hall monitoring

✅ Smart campuses

📊 Advantages

✅ No physical contact
✅ Eliminates proxy attendance
✅ High accuracy using deep learning
✅ Saves time
✅ Works with group photos
✅ Scalable for large classrooms

⚠️ Limitations

❌ Requires good lighting
❌ Face mask may reduce accuracy
❌ High-resolution group photos take more processing time

🔮 Future Enhancements

✅ Live CCTV integration

✅ Cloud-based deployment

✅ Mobile app integration

✅ Face mask recognition

✅ Emotion-based analytics

✅ Auto timetable mapping

📜 License

This project is licensed under the MIT License.

👨‍💻 Developer

Name: Vijay
GitHub: https://github.com/Vijay-k-1554

Project Title: Image-Based Attendance System
