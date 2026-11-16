# Smart Attendance System — Face Recognition Based

A complete, locally running **Smart Attendance Management System** built with **Python, Streamlit, OpenCV (Haar Cascades + LBPH), SQLite, Pandas, and OpenPyXL**. Designed for academic use on Windows without requiring paid APIs or cloud services.

---

## Key Features

- **Face-Based Attendance (`face_recognition_module.py`)**: Local Haar Cascade face detection + LBPH (Local Binary Patterns Histograms) recognition with operator confirmation before saving, unknown-face alerting, and manual attendance backup.
- **Student Management (`database.py`)**: Add, search, view profiles, edit, and delete student records with consent-gated face photograph storage and privacy-compliant face image deletion.
- **Attendance Management (`attendance.py`)**: Same-day duplicate prevention, manual attendance entry, daily Present/Absent roster view, and authorized attendance corrections.
- **Reports & Analytics (`reports.py`)**: Daily, Weekly, Monthly, and Student-wise attendance summaries calculated via `Attendance Percentage = (Present Days / Total Working Days) × 100` with Streamlit charts.
- **Formatted Excel Export (`excel_export.py`)**: Multi-sheet `.xlsx` downloads containing styled **Attendance** and **Students** worksheets.
- **Database Backup & Sample Data**: One-click SQLite `.db` backup and built-in academic sample dataset.

---

## Project Structure

```text
Smart_Attendance_System/
├── app.py                      # Main Streamlit dashboard & 7-section UI
├── database.py                 # SQLite schema, CRUD, backup & demo seed
├── face_recognition_module.py  # OpenCV Haar Cascade + LBPH recognizer
├── attendance.py               # Attendance confirmation & duplicate guards
├── reports.py                  # Pandas analytics (Daily/Weekly/Monthly/Student)
├── excel_export.py             # OpenPyXL formatted Excel workbook exporter
├── requirements.txt            # Python package dependencies
├── README.md                   # Documentation & Windows setup guide
├── database/
│   └── attendance.db           # Auto-initialized SQLite database
├── faces/                      # Locally stored student face samples
├── models/                     # Trained LBPH .yml model & label map
└── assets/
    └── logo.png                # Institutional emblem
```

---

## Step-by-Step Setup in VS Code on Windows

### 1. Prerequisites
- Install **Python 3.10 or 3.11** (64-bit) from [python.org](https://www.python.org/downloads/). Check **"Add python.exe to PATH"** during installation.
- Install **Visual Studio Code** and the **Python** extension by Microsoft.
- Ensure a working webcam (built-in or USB) for live face capture.

### 2. Open Project & Create Virtual Environment
Open the `Smart_Attendance_System` folder in VS Code (`File -> Open Folder`), press `` Ctrl + ` `` to open the integrated terminal, and run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> If PowerShell blocks script execution, run once:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 3. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application
```powershell
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## Troubleshooting Common Errors

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `AttributeError: module 'cv2' has no attribute 'face'` | Standard `opencv-python` was installed instead of `opencv-contrib-python` | Run `pip uninstall opencv-python -y` then `pip install opencv-contrib-python` |
| `Webcam not opening` | Another app (Zoom/Teams/Camera) holds the device, or Windows camera privacy is off | Close other video apps and enable **Camera access for desktop apps** in Windows Settings |
| `No face detected` | Dim backlighting, tilted head, or low-resolution camera | Ensure even front lighting and face the webcam directly |
| `pip install numpy` build error on Python 3.13 | Older wheel mismatch | Use Python **3.10 or 3.11**, or run `pip install --upgrade pip wheel` first |

---

## Limitations & Security Notes

- **Lighting & Pose Sensitivity:** OpenCV LBPH performs best under consistent frontal lighting. Extreme shadows, side profiles, or occlusions lower match confidence.
- **No Liveness Detection:** LBPH compares grayscale texture histograms and does **not** verify 3D depth or blinking. Therefore, the system **never marks attendance automatically**—an operator must visually verify the live student and click **Confirm & Save Attendance**.
- **Privacy First:** All student records, face images, and SQLite databases remain 100% local on your machine. Student consent is required before saving biometric photos, and stored face images can be deleted at any time from **Student Management**.
