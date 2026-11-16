"""
excel_export.py
Formatted multi-worksheet Excel export engine built with OpenPyXL.
Generates .xlsx files in memory for Streamlit st.download_button.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from database import DB_PATH, get_all_students
from attendance import get_attendance_records
from reports import get_daily_report, get_monthly_report, get_student_wise_report

HEADER_FILL = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
SUBHEADER_FILL = PatternFill(start_color="E0F2FE", end_color="E0F2FE", fill_type="solid")
ALT_ROW_FILL = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
TITLE_FONT = Font(name="Calibri", size=14, bold=True, color="0F172A")
BODY_FONT = Font(name="Calibri", size=10, color="1E293B")
THIN_BORDER = Border(
    left=Side(style="thin", color="CBD5E1"),
    right=Side(style="thin", color="CBD5E1"),
    top=Side(style="thin", color="CBD5E1"),
    bottom=Side(style="thin", color="CBD5E1"),
)


def _write_styled_sheet(
    ws,
    title: str,
    headers: List[str],
    rows: List[Tuple],
) -> None:
    """Write a title banner, styled table headers, formatted rows, and auto-fit columns."""
    ws.views.sheetView[0].showGridLines = True
    ws.cell(row=1, column=1, value=title).font = TITLE_FONT
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(1, len(headers)))
    ws.row_dimensions[1].height = 28

    header_row = 3
    ws.row_dimensions[header_row].height = 24
    for col_idx, heading in enumerate(headers, start=1):
        cell = ws.cell(row=header_row, column=col_idx, value=heading)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = THIN_BORDER

    for r_offset, row_values in enumerate(rows, start=1):
        r_idx = header_row + r_offset
        ws.row_dimensions[r_idx].height = 20
        for c_idx, val in enumerate(row_values, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=val)
            cell.font = BODY_FONT
            cell.border = THIN_BORDER
            cell.alignment = Alignment(horizontal="left", vertical="center")
            if r_offset % 2 == 0:
                cell.fill = ALT_ROW_FILL

    # Auto-fit column widths with generous padding
    for col_idx, heading in enumerate(headers, start=1):
        max_len = len(str(heading))
        for row_values in rows:
            val_len = len(str(row_values[col_idx - 1])) if col_idx - 1 < len(row_values) else 0
            if val_len > max_len:
                max_len = val_len
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 6, 42)


def _students_sheet_data(db_path: Path = DB_PATH):
    students = get_all_students(db_path=db_path)
    headers = [
        "Student ID",
        "Full Name",
        "Course",
        "Department",
        "Semester",
        "Email",
        "Registration Date",
    ]
    rows = [
        (
            s["student_id"],
            s["name"],
            s["course"],
            s["department"],
            s["semester"],
            s["email"],
            s["registration_date"],
        )
        for s in students
    ]
    return headers, rows


def export_all_attendance_excel(db_path: Path = DB_PATH) -> bytes:
    """Create workbook with separate 'Students' and 'Attendance' worksheets."""
    wb = Workbook()

    ws_att = wb.active
    ws_att.title = "Attendance"
    records = get_attendance_records(db_path=db_path)
    att_headers = [
        "Record ID",
        "Student ID",
        "Student Name",
        "Course",
        "Date",
        "Time",
        "Status",
        "Method",
    ]
    att_rows = [
        (
            r["attendance_id"],
            r["student_id"],
            r["name"],
            r["course"],
            r["attendance_date"],
            r["attendance_time"],
            r["status"],
            r["method"],
        )
        for r in records
    ]
    _write_styled_sheet(ws_att, "Smart Attendance System — Complete Attendance Log", att_headers, att_rows)

    ws_stu = wb.create_sheet(title="Students")
    stu_headers, stu_rows = _students_sheet_data(db_path)
    _write_styled_sheet(ws_stu, "Registered Students Directory", stu_headers, stu_rows)

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def export_daily_report_excel(target_date: str, db_path: Path = DB_PATH) -> bytes:
    """Export daily attendance roster (Present & Absent) + Students sheet."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Attendance"
    df = get_daily_report(target_date=target_date, db_path=db_path)
    headers = ["Student ID", "Student Name", "Course", "Department", "Date", "Time", "Status", "Method"]
    rows = [
        (
            row["student_id"],
            row["name"],
            row["course"],
            row["department"],
            row["attendance_date"],
            row["attendance_time"],
            row["status"],
            row["method"],
        )
        for _, row in df.iterrows()
    ]
    _write_styled_sheet(ws, f"Daily Attendance Report — {target_date}", headers, rows)

    ws_stu = wb.create_sheet(title="Students")
    stu_headers, stu_rows = _students_sheet_data(db_path)
    _write_styled_sheet(ws_stu, "Registered Students Directory", stu_headers, stu_rows)

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def export_monthly_report_excel(year: int, month: int, db_path: Path = DB_PATH) -> bytes:
    """Export monthly attendance summary + Students sheet."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Attendance"
    df = get_monthly_report(year=year, month=month, db_path=db_path)
    headers = [
        "Student ID",
        "Student Name",
        "Course",
        "Department",
        "Present Days",
        "Absent Days",
        "Working Days",
        "Attendance %",
    ]
    rows = [
        (
            row["student_id"],
            row["name"],
            row["course"],
            row["department"],
            int(row["present_days"]),
            int(row["absent_days"]),
            int(row["working_days"]),
            f"{row['attendance_pct']}%",
        )
        for _, row in df.iterrows()
    ]
    _write_styled_sheet(ws, f"Monthly Attendance Report — {year:04d}-{month:02d}", headers, rows)

    ws_stu = wb.create_sheet(title="Students")
    stu_headers, stu_rows = _students_sheet_data(db_path)
    _write_styled_sheet(ws_stu, "Registered Students Directory", stu_headers, stu_rows)

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def export_student_wise_excel(db_path: Path = DB_PATH) -> bytes:
    """Export cumulative student-wise attendance report + Students sheet."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Attendance"
    df = get_student_wise_report(db_path=db_path)
    headers = [
        "Student ID",
        "Student Name",
        "Course",
        "Department",
        "Semester",
        "Present Days",
        "Absent Days",
        "Total Working Days",
        "Attendance %",
    ]
    rows = [
        (
            row["student_id"],
            row["name"],
            row["course"],
            row["department"],
            row["semester"],
            int(row["present_days"]),
            int(row["absent_days"]),
            int(row["total_working_days"]),
            f"{row['attendance_pct']}%",
        )
        for _, row in df.iterrows()
    ]
    _write_styled_sheet(ws, "Student-Wise Cumulative Attendance Summary", headers, rows)

    ws_stu = wb.create_sheet(title="Students")
    stu_headers, stu_rows = _students_sheet_data(db_path)
    _write_styled_sheet(ws_stu, "Registered Students Directory", stu_headers, stu_rows)

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
