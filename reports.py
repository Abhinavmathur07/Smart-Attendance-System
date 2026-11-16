"""
reports.py
Analytics and Pandas reporting module for Daily, Weekly, Monthly, and Student-wise
attendance reports. Implements:
Attendance Percentage = (Present Days / Total Working Days) * 100
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from database import DB_PATH, get_connection


def _load_tables(db_path: Path = DB_PATH):
    conn = get_connection(db_path)
    try:
        students_df = pd.read_sql_query("SELECT * FROM Students ORDER BY student_id ASC;", conn)
        attendance_df = pd.read_sql_query(
            """
            SELECT a.*, s.name, s.course, s.department, s.semester
            FROM Attendance a
            JOIN Students s ON a.student_id = s.student_id
            ORDER BY a.attendance_date DESC, a.attendance_time DESC;
            """,
            conn,
        )
        return students_df, attendance_df
    finally:
        conn.close()


def get_total_working_days(attendance_df: pd.DataFrame) -> int:
    """Count distinct dates on which attendance sessions were conducted."""
    if attendance_df.empty:
        return 0
    return int(attendance_df["attendance_date"].nunique())


def get_dashboard_metrics(target_date: Optional[str] = None, db_path: Path = DB_PATH) -> Dict[str, Any]:
    """Compute top-level KPI metrics for the Dashboard view."""
    d_str = target_date or date.today().isoformat()
    students_df, attendance_df = _load_tables(db_path)

    total_students = int(len(students_df))
    if total_students == 0:
        return {
            "total_students": 0,
            "present_today": 0,
            "absent_today": 0,
            "overall_attendance_pct": 0.0,
            "total_working_days": 0,
        }

    today_df = attendance_df[attendance_df["attendance_date"] == d_str] if not attendance_df.empty else pd.DataFrame()
    present_today = (
        int(today_df[today_df["status"].isin(["Present", "Late"])]["student_id"].nunique())
        if not today_df.empty
        else 0
    )
    absent_today = max(0, total_students - present_today)

    working_days = get_total_working_days(attendance_df)
    if working_days == 0:
        overall_pct = 0.0
    else:
        present_records = (
            int(attendance_df[attendance_df["status"].isin(["Present", "Late"])].shape[0])
            if not attendance_df.empty
            else 0
        )
        overall_pct = round((present_records / (total_students * working_days)) * 100.0, 1)

    return {
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": absent_today,
        "overall_attendance_pct": overall_pct,
        "total_working_days": working_days,
    }


def get_daily_report(target_date: Optional[str] = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Generate a complete student-by-student report for a single date."""
    d_str = target_date or date.today().isoformat()
    students_df, attendance_df = _load_tables(db_path)
    if students_df.empty:
        return pd.DataFrame(
            columns=["student_id", "name", "course", "department", "attendance_date", "attendance_time", "status", "method"]
        )

    day_att = attendance_df[attendance_df["attendance_date"] == d_str] if not attendance_df.empty else pd.DataFrame()
    if day_att.empty:
        merged = students_df[["student_id", "name", "course", "department"]].copy()
        merged["attendance_date"] = d_str
        merged["attendance_time"] = "-"
        merged["status"] = "Absent"
        merged["method"] = "Not Marked"
        return merged

    merged = students_df[["student_id", "name", "course", "department"]].merge(
        day_att[["student_id", "attendance_date", "attendance_time", "status", "method"]],
        on="student_id",
        how="left",
    )
    merged["attendance_date"] = merged["attendance_date"].fillna(d_str)
    merged["attendance_time"] = merged["attendance_time"].fillna("-")
    merged["status"] = merged["status"].fillna("Absent")
    merged["method"] = merged["method"].fillna("Not Marked")
    return merged


def get_weekly_report(end_date: Optional[date] = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Summarize attendance across the trailing 7-day window ending on end_date."""
    end = end_date or date.today()
    start = end - timedelta(days=6)
    students_df, attendance_df = _load_tables(db_path)
    if attendance_df.empty:
        return pd.DataFrame(columns=["student_id", "name", "course", "present_days", "working_days", "attendance_pct"])

    mask = (attendance_df["attendance_date"] >= start.isoformat()) & (
        attendance_df["attendance_date"] <= end.isoformat()
    )
    window_df = attendance_df[mask]
    working_days = max(1, int(window_df["attendance_date"].nunique())) if not window_df.empty else 0

    rows = []
    for _, stu in students_df.iterrows():
        s_att = window_df[window_df["student_id"] == stu["student_id"]]
        present = int(s_att[s_att["status"].isin(["Present", "Late"])].shape[0])
        pct = round((present / working_days) * 100.0, 1) if working_days > 0 else 0.0
        rows.append(
            {
                "student_id": stu["student_id"],
                "name": stu["name"],
                "course": stu["course"],
                "present_days": present,
                "absent_days": max(0, working_days - present),
                "working_days": working_days,
                "attendance_pct": pct,
            }
        )
    return pd.DataFrame(rows)


def get_monthly_report(year: int, month: int, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Summarize attendance for a specific YYYY-MM calendar month."""
    prefix = f"{year:04d}-{month:02d}"
    students_df, attendance_df = _load_tables(db_path)
    if students_df.empty:
        return pd.DataFrame()

    month_df = (
        attendance_df[attendance_df["attendance_date"].str.startswith(prefix)]
        if not attendance_df.empty
        else pd.DataFrame()
    )
    working_days = int(month_df["attendance_date"].nunique()) if not month_df.empty else 0

    rows = []
    for _, stu in students_df.iterrows():
        s_att = month_df[month_df["student_id"] == stu["student_id"]] if not month_df.empty else pd.DataFrame()
        present = int(s_att[s_att["status"].isin(["Present", "Late"])].shape[0]) if not s_att.empty else 0
        pct = round((present / working_days) * 100.0, 1) if working_days > 0 else 0.0
        rows.append(
            {
                "student_id": stu["student_id"],
                "name": stu["name"],
                "course": stu["course"],
                "department": stu["department"],
                "present_days": present,
                "absent_days": max(0, working_days - present),
                "working_days": working_days,
                "attendance_pct": pct,
            }
        )
    return pd.DataFrame(rows)


def get_student_wise_report(search_query: str = "", db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Compute cumulative student-wise attendance statistics using:
    Attendance Percentage = (Present Days / Total Working Days) * 100
    """
    students_df, attendance_df = _load_tables(db_path)
    if students_df.empty:
        return pd.DataFrame(
            columns=[
                "student_id",
                "name",
                "course",
                "department",
                "semester",
                "present_days",
                "absent_days",
                "total_working_days",
                "attendance_pct",
            ]
        )

    working_days = get_total_working_days(attendance_df)
    rows = []
    q = search_query.strip().lower()
    for _, stu in students_df.iterrows():
        if q and q not in stu["student_id"].lower() and q not in stu["name"].lower():
            continue
        s_att = attendance_df[attendance_df["student_id"] == stu["student_id"]] if not attendance_df.empty else pd.DataFrame()
        present = int(s_att[s_att["status"].isin(["Present", "Late"])].shape[0]) if not s_att.empty else 0
        pct = round((present / working_days) * 100.0, 1) if working_days > 0 else 0.0
        rows.append(
            {
                "student_id": stu["student_id"],
                "name": stu["name"],
                "course": stu["course"],
                "department": stu["department"],
                "semester": stu["semester"],
                "present_days": present,
                "absent_days": max(0, working_days - present),
                "total_working_days": working_days,
                "attendance_pct": pct,
            }
        )
    return pd.DataFrame(rows)


def get_attendance_trend(days: int = 14, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return daily present vs absent counts for trend charting."""
    students_df, attendance_df = _load_tables(db_path)
    total_students = int(len(students_df))
    if attendance_df.empty or total_students == 0:
        return pd.DataFrame(columns=["date", "Present", "Absent", "Attendance %"])

    grouped = (
        attendance_df[attendance_df["status"].isin(["Present", "Late"])]
        .groupby("attendance_date")["student_id"]
        .nunique()
        .reset_index(name="Present")
        .sort_values("attendance_date")
        .tail(days)
    )
    grouped["Absent"] = (total_students - grouped["Present"]).clip(lower=0)
    grouped["Attendance %"] = ((grouped["Present"] / total_students) * 100.0).round(1)
    grouped = grouped.rename(columns={"attendance_date": "date"})
    return grouped
