"""
attendance.py
Attendance business logic: confirmation-gated face attendance, manual backup,
duplicate same-day prevention, daily roster generation, and authorized corrections.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from database import DB_PATH, get_connection, get_student_by_id


def has_attendance_on_date(
    student_id: str,
    target_date: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> Optional[Dict[str, Any]]:
    """Return existing attendance row if student already has a record on target_date."""
    d_str = target_date or date.today().isoformat()
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT a.*, s.name, s.course, s.department
            FROM Attendance a
            JOIN Students s ON a.student_id = s.student_id
            WHERE a.student_id = ? AND a.attendance_date = ?;
            """,
            (student_id.strip().upper(), d_str),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def mark_attendance(
    student_id: str,
    status: str = "Present",
    method: str = "Face Recognition",
    confirmed: bool = False,
    attendance_date: Optional[str] = None,
    attendance_time: Optional[str] = None,
    notes: str = "",
    db_path: Path = DB_PATH,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Record attendance for a registered student.
    When method == 'Face Recognition', requires confirmed=True so attendance is
    never marked automatically without explicit operator review.
    """
    sid = student_id.strip().upper()
    if method == "Face Recognition" and not confirmed:
        return False, "Operator confirmation is required before saving face recognition attendance.", None

    if status not in ("Present", "Absent", "Late"):
        return False, "Invalid attendance status. Must be Present, Absent, or Late.", None

    student = get_student_by_id(sid, db_path=db_path)
    if not student:
        return False, f"Student ID '{sid}' does not exist in the database.", None

    now = datetime.now()
    d_str = attendance_date or now.date().isoformat()
    t_str = attendance_time or now.strftime("%H:%M:%S")

    existing = has_attendance_on_date(sid, d_str, db_path=db_path)
    if existing:
        return (
            False,
            f"Duplicate prevented: {student['name']} ({sid}) already marked '{existing['status']}' on {d_str} at {existing['attendance_time']}.",
            existing,
        )

    conn = get_connection(db_path)
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO Attendance (
                    student_id, attendance_date, attendance_time, status, method, notes
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                (sid, d_str, t_str, status, method, notes.strip()),
            )
            record_id = cur.lastrowid
        record = {
            "attendance_id": record_id,
            "student_id": sid,
            "name": student["name"],
            "attendance_date": d_str,
            "attendance_time": t_str,
            "status": status,
            "method": method,
            "notes": notes.strip(),
        }
        return True, f"Attendance marked '{status}' for {student['name']} ({sid}) at {t_str}.", record
    except sqlite3.IntegrityError:
        return False, f"Attendance already exists for {sid} on {d_str}.", None
    except sqlite3.Error as exc:
        return False, f"Database error recording attendance: {exc}", None
    finally:
        conn.close()


def correct_attendance_record(
    attendance_id: int,
    new_status: str,
    reason: str = "",
    db_path: Path = DB_PATH,
) -> Tuple[bool, str]:
    """Update an existing attendance record for authorized corrections."""
    if new_status not in ("Present", "Absent", "Late"):
        return False, "Status must be Present, Absent, or Late."

    conn = get_connection(db_path)
    try:
        with conn:
            cur = conn.execute(
                """
                UPDATE Attendance
                SET status = ?, notes = CASE
                    WHEN ? != '' THEN ?
                    ELSE notes
                END
                WHERE attendance_id = ?;
                """,
                (new_status, reason.strip(), f"Corrected: {reason.strip()}", attendance_id),
            )
            if cur.rowcount == 0:
                return False, f"Attendance record #{attendance_id} not found."
        return True, f"Record #{attendance_id} updated to '{new_status}'."
    except sqlite3.Error as exc:
        return False, f"Database error correcting record: {exc}"
    finally:
        conn.close()


def get_attendance_records(
    target_date: Optional[str] = None,
    student_id: Optional[str] = None,
    status_filter: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """Query joined Attendance + Students rows with optional filters."""
    query = """
        SELECT
            a.attendance_id,
            a.student_id,
            s.name,
            s.course,
            s.department,
            s.semester,
            a.attendance_date,
            a.attendance_time,
            a.status,
            a.method,
            a.notes
        FROM Attendance a
        JOIN Students s ON a.student_id = s.student_id
        WHERE 1=1
    """
    params: List[Any] = []
    if target_date:
        query += " AND a.attendance_date = ?"
        params.append(target_date)
    if student_id and student_id != "ALL":
        query += " AND a.student_id = ?"
        params.append(student_id.strip().upper())
    if status_filter and status_filter != "ALL":
        query += " AND a.status = ?"
        params.append(status_filter)

    query += " ORDER BY a.attendance_date DESC, a.attendance_time DESC;"

    conn = get_connection(db_path)
    try:
        rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_daily_roster(target_date: Optional[str] = None, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """
    Return every registered student along with their attendance status on target_date.
    Students without a record on that date are shown as 'Absent' (Unmarked).
    """
    d_str = target_date or date.today().isoformat()
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                s.student_id,
                s.name,
                s.course,
                s.department,
                s.semester,
                a.attendance_id,
                COALESCE(a.status, 'Absent') AS status,
                COALESCE(a.attendance_time, '-') AS attendance_time,
                COALESCE(a.method, 'Not Marked') AS method,
                CASE WHEN a.attendance_id IS NULL THEN 0 ELSE 1 END AS is_marked
            FROM Students s
            LEFT JOIN Attendance a
                ON s.student_id = a.student_id AND a.attendance_date = ?
            ORDER BY s.student_id ASC;
            """,
            (d_str,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
