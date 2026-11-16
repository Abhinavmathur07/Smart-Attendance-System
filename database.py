"""
database.py
SQLite persistence layer for the Smart Attendance System.
Handles schema initialization, student CRUD, attendance tracking,
settings storage, database backup, and optional sample/demo seeding.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "database"
DB_PATH = DB_DIR / "attendance.db"
BACKUP_DIR = DB_DIR / "backups"
FACES_DIR = BASE_DIR / "faces"
MODELS_DIR = BASE_DIR / "models"


def ensure_directories() -> None:
    """Create required local storage folders if they do not exist."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys and row factory enabled."""
    ensure_directories()
    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to connect to SQLite database at {db_path}: {exc}") from exc


def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize Students, Attendance, and Settings tables safely."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS Students (
                    student_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    course TEXT NOT NULL,
                    department TEXT NOT NULL,
                    semester TEXT NOT NULL,
                    email TEXT NOT NULL,
                    face_image TEXT,
                    registration_date TEXT NOT NULL,
                    consent_given INTEGER NOT NULL DEFAULT 1
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS Attendance (
                    attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    attendance_date TEXT NOT NULL,
                    attendance_time TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('Present', 'Absent', 'Late')),
                    method TEXT NOT NULL DEFAULT 'Manual',
                    notes TEXT DEFAULT '',
                    FOREIGN KEY (student_id) REFERENCES Students(student_id) ON DELETE CASCADE,
                    UNIQUE(student_id, attendance_date)
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS Settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            # Seed default settings if absent
            defaults = {
                "confidence_threshold": "65.0",
                "institution_name": "Department of Computer Science & Engineering",
                "academic_term": "Fall 2026",
            }
            for key, val in defaults.items():
                conn.execute(
                    "INSERT OR IGNORE INTO Settings (key, value) VALUES (?, ?);",
                    (key, val),
                )
    finally:
        conn.close()


def validate_student_payload(
    student_id: str,
    name: str,
    course: str,
    department: str,
    semester: str,
    email: str,
    consent_given: bool = True,
) -> Tuple[bool, str]:
    """Validate student input fields before saving to SQLite."""
    if not student_id or not re.match(r"^[A-Za-z0-9_-]{2,32}$", student_id.strip()):
        return False, "Student ID must be 2-32 alphanumeric characters (dashes/underscores allowed)."
    if not name or len(name.strip()) < 2:
        return False, "Full name must be at least 2 characters."
    if not course or len(course.strip()) < 2:
        return False, "Course name is required."
    if not department or len(department.strip()) < 2:
        return False, "Department name is required."
    if not semester or not semester.strip():
        return False, "Semester is required."
    if not email or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()):
        return False, "Please enter a valid institutional email address."
    if not consent_given:
        return False, "Student biometric consent is required before storing face images."
    return True, ""


def add_student(
    student_id: str,
    name: str,
    course: str,
    department: str,
    semester: str,
    email: str,
    face_image: Optional[str] = None,
    registration_date: Optional[str] = None,
    consent_given: bool = True,
    db_path: Path = DB_PATH,
) -> Tuple[bool, str]:
    """Insert a new student record into SQLite."""
    valid, error = validate_student_payload(
        student_id, name, course, department, semester, email, consent_given
    )
    if not valid:
        return False, error

    reg_date = registration_date or date.today().isoformat()
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO Students (
                    student_id, name, course, department, semester,
                    email, face_image, registration_date, consent_given
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    student_id.strip().upper(),
                    name.strip(),
                    course.strip(),
                    department.strip(),
                    semester.strip(),
                    email.strip().lower(),
                    face_image,
                    reg_date,
                    1 if consent_given else 0,
                ),
            )
        return True, f"Student {name.strip()} ({student_id.strip().upper()}) registered successfully."
    except sqlite3.IntegrityError:
        return False, f"A student with ID '{student_id.strip().upper()}' already exists."
    except sqlite3.Error as exc:
        return False, f"Database error while adding student: {exc}"
    finally:
        conn.close()


def get_all_students(search_query: str = "", db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all students, optionally filtered by ID, name, course, or department."""
    conn = get_connection(db_path)
    try:
        if search_query.strip():
            like = f"%{search_query.strip()}%"
            rows = conn.execute(
                """
                SELECT * FROM Students
                WHERE student_id LIKE ? OR name LIKE ? OR course LIKE ? OR department LIKE ?
                ORDER BY registration_date DESC, name ASC;
                """,
                (like, like, like, like),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM Students ORDER BY registration_date DESC, name ASC;"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_student_by_id(student_id: str, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    """Fetch a single student record by primary key."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM Students WHERE student_id = ?;",
            (student_id.strip().upper(),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_student(
    student_id: str,
    name: str,
    course: str,
    department: str,
    semester: str,
    email: str,
    face_image: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> Tuple[bool, str]:
    """Update editable fields for an existing student."""
    valid, error = validate_student_payload(
        student_id, name, course, department, semester, email, True
    )
    if not valid:
        return False, error

    conn = get_connection(db_path)
    try:
        with conn:
            if face_image is not None:
                cur = conn.execute(
                    """
                    UPDATE Students
                    SET name = ?, course = ?, department = ?, semester = ?, email = ?, face_image = ?
                    WHERE student_id = ?;
                    """,
                    (
                        name.strip(),
                        course.strip(),
                        department.strip(),
                        semester.strip(),
                        email.strip().lower(),
                        face_image,
                        student_id.strip().upper(),
                    ),
                )
            else:
                cur = conn.execute(
                    """
                    UPDATE Students
                    SET name = ?, course = ?, department = ?, semester = ?, email = ?
                    WHERE student_id = ?;
                    """,
                    (
                        name.strip(),
                        course.strip(),
                        department.strip(),
                        semester.strip(),
                        email.strip().lower(),
                        student_id.strip().upper(),
                    ),
                )
            if cur.rowcount == 0:
                return False, f"Student '{student_id}' not found."
        return True, f"Updated profile for {name.strip()}."
    except sqlite3.Error as exc:
        return False, f"Database error while updating student: {exc}"
    finally:
        conn.close()


def clear_student_face_image(student_id: str, db_path: Path = DB_PATH) -> Tuple[bool, str]:
    """Remove stored face image path from SQLite for privacy compliance."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE Students SET face_image = NULL WHERE student_id = ?;",
                (student_id.strip().upper(),),
            )
        return True, f"Face image reference cleared for {student_id.strip().upper()}."
    except sqlite3.Error as exc:
        return False, f"Database error clearing face image: {exc}"
    finally:
        conn.close()


def delete_student(student_id: str, db_path: Path = DB_PATH) -> Tuple[bool, str]:
    """Delete a student record and cascading attendance entries."""
    conn = get_connection(db_path)
    try:
        with conn:
            cur = conn.execute(
                "DELETE FROM Students WHERE student_id = ?;",
                (student_id.strip().upper(),),
            )
            if cur.rowcount == 0:
                return False, f"Student '{student_id}' not found."
        return True, f"Student {student_id.strip().upper()} and associated records deleted."
    except sqlite3.Error as exc:
        return False, f"Database error while deleting student: {exc}"
    finally:
        conn.close()


def get_setting(key: str, default: str = "", db_path: Path = DB_PATH) -> str:
    """Read a persistent setting value."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT value FROM Settings WHERE key = ?;", (key,)).fetchone()
        return row["value"] if row else default
    finally:
        conn.close()


def set_setting(key: str, value: str, db_path: Path = DB_PATH) -> None:
    """Upsert a persistent setting value."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO Settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value;
                """,
                (key, str(value)),
            )
    finally:
        conn.close()


def backup_database(db_path: Path = DB_PATH, backup_dir: Path = BACKUP_DIR) -> Tuple[bool, str]:
    """Create a timestamped SQLite backup copy on local disk."""
    ensure_directories()
    if not db_path.exists():
        return False, "Database file does not exist yet."
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"attendance_backup_{stamp}.db"
    try:
        shutil.copy2(db_path, dest)
        return True, str(dest)
    except OSError as exc:
        return False, f"Failed to create backup: {exc}"


def seed_sample_data(db_path: Path = DB_PATH) -> Tuple[bool, str]:
    """Populate 22 realistic academic sample students and 7+ days of attendance records."""
    init_db(db_path)
    sample_students = [
        ("CS2026001", "Aarav Sharma", "B.Tech CSE", "Computer Science", "Semester 5", "aarav.sharma@college.edu"),
        ("CS2026002", "Priya Nair", "B.Tech CSE", "Computer Science", "Semester 5", "priya.nair@college.edu"),
        ("CS2026003", "Rohan Verma", "B.Tech AI & DS", "Artificial Intelligence", "Semester 3", "rohan.verma@college.edu"),
        ("CS2026004", "Ananya Iyer", "B.Tech ECE", "Electronics", "Semester 7", "ananya.iyer@college.edu"),
        ("CS2026005", "Kabir Mehta", "B.Tech CSE", "Computer Science", "Semester 5", "kabir.mehta@college.edu"),
        ("CS2026006", "Ishita Deshmukh", "B.Tech CSE", "Computer Science", "Semester 5", "ishita.deshmukh@college.edu"),
        ("CS2026007", "Vivaan Kulkarni", "B.Tech AI & DS", "Artificial Intelligence", "Semester 3", "vivaan.kulkarni@college.edu"),
        ("CS2026008", "Diya Chatterjee", "B.Tech IT", "Information Technology", "Semester 5", "diya.chatterjee@college.edu"),
        ("CS2026009", "Aditya Bansal", "B.Tech CSE", "Computer Science", "Semester 7", "aditya.bansal@college.edu"),
        ("CS2026010", "Sneha Reddy", "B.Tech ECE", "Electronics", "Semester 5", "sneha.reddy@college.edu"),
        ("CS2026011", "Arjun Malhotra", "B.Tech AI & DS", "Artificial Intelligence", "Semester 3", "arjun.malhotra@college.edu"),
        ("CS2026012", "Meera Krishnan", "B.Tech CSE", "Computer Science", "Semester 5", "meera.krishnan@college.edu"),
        ("CS2026013", "Siddharth Joshi", "B.Tech IT", "Information Technology", "Semester 3", "siddharth.joshi@college.edu"),
        ("CS2026014", "Kavya Menon", "B.Tech ECE", "Electronics", "Semester 7", "kavya.menon@college.edu"),
        ("CS2026015", "Nikhil Agarwal", "B.Tech CSE", "Computer Science", "Semester 5", "nikhil.agarwal@college.edu"),
        ("CS2026016", "Tanvi Kapoor", "B.Tech AI & DS", "Artificial Intelligence", "Semester 5", "tanvi.kapoor@college.edu"),
        ("CS2026017", "Pranav Tiwari", "B.Tech CSE", "Computer Science", "Semester 3", "pranav.tiwari@college.edu"),
        ("CS2026018", "Riya Sengupta", "B.Tech IT", "Information Technology", "Semester 7", "riya.sengupta@college.edu"),
        ("CS2026019", "Harshvardhan Rao", "B.Tech ECE", "Electronics", "Semester 5", "harsh.rao@college.edu"),
        ("CS2026020", "Nandini Bhatia", "B.Tech CSE", "Computer Science", "Semester 5", "nandini.bhatia@college.edu"),
        ("CS2026021", "Devansh Saxena", "B.Tech AI & DS", "Artificial Intelligence", "Semester 3", "devansh.saxena@college.edu"),
        ("CS2026022", "Pooja Choudhary", "B.Tech IT", "Information Technology", "Semester 5", "pooja.choudhary@college.edu"),
    ]
    conn = get_connection(db_path)
    try:
        with conn:
            today = date.today()
            for idx, (sid, name, course, dept, sem, email) in enumerate(sample_students):
                reg = (today - timedelta(days=35 - idx)).isoformat()
                conn.execute(
                    """
                    INSERT OR IGNORE INTO Students (
                        student_id, name, course, department, semester,
                        email, face_image, registration_date, consent_given
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, 1);
                    """,
                    (sid, name, course, dept, sem, email, reg),
                )

            # Seed past 9 working days + today partial attendance
            for day_offset in range(9, -1, -1):
                target_day = today - timedelta(days=day_offset)
                if target_day.weekday() >= 5:
                    continue
                d_str = target_day.isoformat()
                for idx, (sid, *_rest) in enumerate(sample_students):
                    # Leave several students unmarked today so Present/Absent contrast is visible
                    if day_offset == 0 and idx >= 16:
                        continue
                    status = "Present" if (idx + day_offset) % 7 != 0 else "Absent"
                    if day_offset == 0:
                        status = "Present"
                    method = "Face Recognition" if idx < 4 else "Manual"
                    t_str = f"09:{10 + idx:02d}:15"
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO Attendance (
                            student_id, attendance_date, attendance_time, status, method, notes
                        ) VALUES (?, ?, ?, ?, ?, ?);
                        """,
                        (sid, d_str, t_str, status, method, "Sample academic seed"),
                    )
        return True, f"{len(sample_students)} sample students and attendance history loaded into SQLite."
    except sqlite3.Error as exc:
        return False, f"Error seeding sample data: {exc}"
    finally:
        conn.close()
