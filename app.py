"""
app.py
Main Streamlit frontend for the Smart Attendance System (Face Recognition Based).
Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from attendance import (
    correct_attendance_record,
    get_attendance_records,
    get_daily_roster,
    mark_attendance,
)
from database import (
    DB_PATH,
    add_student,
    backup_database,
    clear_student_face_image,
    delete_student,
    get_all_students,
    get_setting,
    get_student_by_id,
    init_db,
    seed_sample_data,
    set_setting,
    update_student,
)
from excel_export import (
    export_all_attendance_excel,
    export_daily_report_excel,
    export_monthly_report_excel,
    export_student_wise_excel,
)
from face_recognition_module import (
    MODEL_PATH,
    check_opencv_environment,
    delete_student_faces,
    recognize_face,
    register_student_face,
    train_lbph_model,
)
from reports import (
    get_attendance_trend,
    get_daily_report,
    get_dashboard_metrics,
    get_monthly_report,
    get_student_wise_report,
    get_weekly_report,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"

st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize SQLite tables on every startup
init_db()

# Custom CSS for clean dark-navy + cyan enterprise academic dashboard styling
st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1340px;
        }
        [data-testid="stSidebar"] {
            background-color: #0F172A;
            color: #F8FAFC;
        }
        [data-testid="stSidebar"] * {
            color: #E2E8F0;
        }
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        }
        .kpi-label {
            font-size: 0.82rem;
            font-weight: 600;
            color: #64748B;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 1.9rem;
            font-weight: 700;
            color: #0F172A;
            font-variant-numeric: tabular-nums;
            line-height: 1.1;
        }
        .kpi-sub {
            font-size: 0.78rem;
            color: #0284C7;
            margin-top: 0.4rem;
        }
        .section-title {
            font-size: 1.45rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.2rem;
        }
        .section-sub {
            font-size: 0.92rem;
            color: #64748B;
            margin-bottom: 1.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_sidebar() -> str:
    """Render sidebar navigation and logo."""
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=72)
        st.markdown("### Smart Attendance")
        st.caption(get_setting("institution_name", "Academic Biometric Roster"))
        st.divider()
        section = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Face Recognition",
                "Student Management",
                "Attendance Records",
                "Reports",
                "Excel Export",
                "Settings",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        cv_ok, _ = check_opencv_environment()
        st.caption(f"SQLite DB: {'Connected' if DB_PATH.exists() else 'Ready'}")
        st.caption(f"OpenCV LBPH: {'Ready' if cv_ok else 'Missing opencv-contrib-python'}")
        st.caption(f"Trained Model: {'Loaded' if MODEL_PATH.exists() else 'Not trained yet'}")
    return section


def page_dashboard() -> None:
    """1. Dashboard overview with 4 KPI cards, trend chart, recent activity, and stats."""
    st.markdown('<div class="section-title">Attendance Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Real-time overview of student attendance, daily turnout, and local recognition status.</div>',
        unsafe_allow_html=True,
    )

    today_str = date.today().isoformat()
    metrics = get_dashboard_metrics(today_str)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Students</div>
                <div class="kpi-value">{metrics['total_students']}</div>
                <div class="kpi-sub">Registered in SQLite</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Present Today</div>
                <div class="kpi-value">{metrics['present_today']}</div>
                <div class="kpi-sub">{today_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Absent Today</div>
                <div class="kpi-value">{metrics['absent_today']}</div>
                <div class="kpi-sub">Pending or unmarked</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Overall Attendance %</div>
                <div class="kpi-value">{metrics['overall_attendance_pct']}%</div>
                <div class="kpi-sub">Across {metrics['total_working_days']} working day(s)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Attendance Trend (Last 14 Working Days)")
        trend_df = get_attendance_trend(14)
        if trend_df.empty:
            st.info("No attendance trend data yet. Mark attendance or load sample data in Settings.")
        else:
            st.bar_chart(trend_df.set_index("date")[["Present", "Absent"]])

    with right:
        st.subheader("System Status")
        cv_ok, cv_msg = check_opencv_environment()
        st.write(f"- **Database:** `{DB_PATH.name}` (Local SQLite)")
        st.write(f"- **Face Engine:** {cv_msg}")
        st.write(f"- **LBPH Model:** {'Trained & active' if MODEL_PATH.exists() else 'Awaiting student face photos'}")
        st.write(f"- **Confidence Threshold:** `{get_setting('confidence_threshold', '65.0')}`")

        students = get_all_students()
        if students:
            dept_counts = pd.DataFrame(students)["department"].value_counts()
            st.subheader("Students by Department")
            st.dataframe(dept_counts.rename_axis("Department").reset_index(name="Count"), use_container_width=True, hide_index=True)

    st.subheader("Recent Attendance Activity")
    recent = get_attendance_records()[:10]
    if not recent:
        st.info("No attendance logs recorded yet.")
    else:
        st.dataframe(
            pd.DataFrame(recent)[
                ["attendance_date", "attendance_time", "student_id", "name", "course", "status", "method"]
            ],
            use_container_width=True,
            hide_index=True,
        )


def page_face_recognition() -> None:
    """2. Face Recognition attendance capture with mandatory operator confirmation."""
    st.markdown('<div class="section-title">Face Recognition Attendance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Detect and match faces locally using OpenCV Haar Cascades & LBPH. Attendance requires manual operator confirmation before saving.</div>',
        unsafe_allow_html=True,
    )

    cv_ok, cv_msg = check_opencv_environment()
    if not cv_ok:
        st.error(cv_msg)

    threshold = float(get_setting("confidence_threshold", "65.0"))
    col_cam, col_res = st.columns([3, 2])

    with col_cam:
        input_mode = st.radio(
            "Capture Source",
            ["Live Webcam (st.camera_input)", "Upload Image File"],
            horizontal=True,
        )
        captured_image = None
        if input_mode == "Live Webcam (st.camera_input)":
            cam_file = st.camera_input("Align student face within frame and capture")
            if cam_file is not None:
                captured_image = Image.open(cam_file)
        else:
            uploaded = st.file_uploader("Upload student photograph", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                captured_image = Image.open(uploaded)

        if captured_image is not None:
            if st.button("Run Face Recognition", type="primary"):
                result = recognize_face(captured_image, threshold=threshold)
                st.session_state["last_recognition"] = result

    with col_res:
        st.subheader("Verification & Confirmation")
        result = st.session_state.get("last_recognition")
        if not result:
            st.info("Capture or upload a frame, then click 'Run Face Recognition'.")
        else:
            if "annotated_rgb" in result:
                st.image(result["annotated_rgb"], caption="Processed Detection Frame", use_container_width=True)

            if result.get("recognized"):
                sid = result["student_id"]
                student = get_student_by_id(sid)
                if student:
                    st.success(f"Recognized: **{student['name']}** (`{sid}`) — Match Score: **{result['confidence_pct']}%**")
                    st.write(f"**Course:** {student['course']} · **Department:** {student['department']}")
                    status_choice = st.selectbox("Attendance Status", ["Present", "Late"], key="face_att_status")
                    if st.button("Confirm & Save Attendance", type="primary", use_container_width=True):
                        ok, msg, _ = mark_attendance(
                            student_id=sid,
                            status=status_choice,
                            method="Face Recognition",
                            confirmed=True,
                        )
                        if ok:
                            st.success(msg)
                        else:
                            st.warning(msg)
                else:
                    st.error(f"Matched face label '{sid}', but student record is missing in SQLite.")
            else:
                st.warning(result.get("message", "Face not recognized."))
                with st.expander("Manual Attendance Backup", expanded=True):
                    students = get_all_students()
                    if not students:
                        st.info("No registered students available.")
                    else:
                        options = {f"{s['student_id']} — {s['name']}": s["student_id"] for s in students}
                        selected_label = st.selectbox("Select Student Manually", list(options.keys()))
                        manual_status = st.selectbox("Status", ["Present", "Late", "Absent"], key="manual_backup_status")
                        if st.button("Mark Manual Attendance"):
                            ok, msg, _ = mark_attendance(
                                student_id=options[selected_label],
                                status=manual_status,
                                method="Manual",
                                confirmed=True,
                            )
                            if ok:
                                st.success(msg)
                            else:
                                st.warning(msg)

    with st.expander("Model Limitations & Privacy Notice"):
        st.markdown(
            """
            - **Local Processing:** All Haar Cascade detection and LBPH feature matching runs offline on this machine. No biometric images leave your computer.
            - **Accuracy Constraints:** LBPH accuracy decreases under harsh backlighting, extreme head tilt, occlusions, or low-resolution webcams.
            - **No Liveness Detection:** OpenCV LBPH does **not** perform 3D anti-spoofing or liveness verification. Operators must visually confirm the student is physically present before clicking **Confirm & Save Attendance**.
            """
        )


def page_student_management() -> None:
    """3. Student CRUD, search, profile viewer, face photo capture/upload, and consent."""
    st.markdown('<div class="section-title">Student Management</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Register students, manage academic profiles, capture consent-approved face portraits, and maintain SQLite records.</div>',
        unsafe_allow_html=True,
    )

    tab_dir, tab_add, tab_edit = st.tabs(["Student Directory", "Register New Student", "Edit / Delete Student"])

    with tab_dir:
        search_q = st.text_input("Search by Student ID, Name, Course, or Department", placeholder="e.g. CS2026001 or Aarav")
        students = get_all_students(search_q)
        if not students:
            st.info("No students found matching your filter.")
        else:
            df = pd.DataFrame(students)[
                ["student_id", "name", "course", "department", "semester", "email", "registration_date"]
            ]
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.subheader("Student Profile Viewer")
            selected_sid = st.selectbox(
                "Inspect Student Profile",
                [s["student_id"] for s in students],
                format_func=lambda sid: next((f"{s['student_id']} — {s['name']}" for s in students if s["student_id"] == sid), sid),
            )
            profile = get_student_by_id(selected_sid)
            if profile:
                p_col1, p_col2 = st.columns([1, 3])
                with p_col1:
                    img_path = profile.get("face_image")
                    if img_path and Path(img_path).exists():
                        st.image(img_path, width=160, caption="Registered Face Photo")
                    else:
                        st.info("No face photo stored.")
                with p_col2:
                    st.markdown(f"**{profile['name']}** (`{profile['student_id']}`)")
                    st.write(f"Course: {profile['course']} · Department: {profile['department']} · {profile['semester']}")
                    st.write(f"Email: {profile['email']} · Registered: {profile['registration_date']}")
                    st.write(f"Biometric Consent: {'Recorded' if profile['consent_given'] else 'Pending'}")

    with tab_add:
        with st.form("add_student_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                sid = st.text_input("Student ID *", placeholder="CS2026010")
                name = st.text_input("Full Name *", placeholder="Neha Kulkarni")
                course = st.text_input("Course *", placeholder="B.Tech CSE")
            with c2:
                dept = st.text_input("Department *", placeholder="Computer Science")
                sem = st.selectbox("Semester *", [f"Semester {i}" for i in range(1, 9)])
                email = st.text_input("Institutional Email *", placeholder="neha.kulkarni@college.edu")

            photo_source = st.radio("Face Photograph Option", ["Upload Portrait", "Capture via Webcam", "Register Without Photo (Add Later)"], horizontal=True)
            face_file = None
            if photo_source == "Upload Portrait":
                face_file = st.file_uploader("Upload Front-Facing Portrait (JPG/PNG)", type=["jpg", "jpeg", "png"], key="add_stu_upload")
            elif photo_source == "Capture via Webcam":
                face_file = st.camera_input("Capture Student Portrait", key="add_stu_cam")

            consent = st.checkbox("I confirm the student has given explicit consent to store their face image locally for attendance verification.")
            submitted = st.form_submit_button("Register Student", type="primary")

            if submitted:
                face_path = None
                if face_file is not None:
                    if not consent:
                        st.error("Student biometric consent is required before saving face images.")
                        st.stop()
                    pil_img = Image.open(face_file)
                    f_ok, f_msg, saved_path = register_student_face(sid, pil_img)
                    if not f_ok:
                        st.error(f_msg)
                        st.stop()
                    face_path = saved_path

                ok, msg = add_student(
                    student_id=sid,
                    name=name,
                    course=course,
                    department=dept,
                    semester=sem,
                    email=email,
                    face_image=face_path,
                    consent_given=consent,
                )
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    with tab_edit:
        all_students = get_all_students()
        if not all_students:
            st.info("No students registered yet.")
        else:
            edit_sid = st.selectbox(
                "Select Student to Manage",
                [s["student_id"] for s in all_students],
                format_func=lambda sid: next((f"{s['student_id']} — {s['name']}" for s in all_students if s["student_id"] == sid), sid),
                key="edit_select_sid",
            )
            stu = get_student_by_id(edit_sid)
            if stu:
                with st.form("edit_student_form"):
                    e_name = st.text_input("Full Name", value=stu["name"])
                    e_course = st.text_input("Course", value=stu["course"])
                    e_dept = st.text_input("Department", value=stu["department"])
                    e_sem = st.text_input("Semester", value=stu["semester"])
                    e_email = st.text_input("Email", value=stu["email"])
                    new_photo = st.file_uploader("Replace Face Portrait (Optional)", type=["jpg", "jpeg", "png"], key="edit_photo")
                    save_btn = st.form_submit_button("Save Changes", type="primary")
                    if save_btn:
                        new_face_path = None
                        if new_photo is not None:
                            f_ok, f_msg, saved_path = register_student_face(edit_sid, Image.open(new_photo))
                            if not f_ok:
                                st.error(f_msg)
                                st.stop()
                            new_face_path = saved_path
                        ok, msg = update_student(edit_sid, e_name, e_course, e_dept, e_sem, e_email, new_face_path)
                        st.success(msg) if ok else st.error(msg)

                col_del1, col_del2 = st.columns(2)
                with col_del1:
                    if st.button("Delete Face Images Only (Privacy Reset)", use_container_width=True):
                        delete_student_faces(edit_sid)
                        ok, msg = clear_student_face_image(edit_sid)
                        st.success(msg) if ok else st.error(msg)
                with col_del2:
                    if st.button("Delete Entire Student Record", type="secondary", use_container_width=True):
                        delete_student_faces(edit_sid)
                        ok, msg = delete_student(edit_sid)
                        st.success(msg) if ok else st.error(msg)


def page_attendance_records() -> None:
    """4. View, filter, manually mark, and correct attendance records."""
    st.markdown('<div class="section-title">Attendance Records & Manual Control</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Inspect daily Present/Absent rosters, log manual attendance, and perform authorized record corrections.</div>',
        unsafe_allow_html=True,
    )

    tab_roster, tab_manual, tab_history = st.tabs(["Daily Roster (Present & Absent)", "Mark Manual Attendance", "Full Log & Corrections"])

    with tab_roster:
        sel_date = st.date_input("Roster Date", value=date.today())
        roster = get_daily_roster(sel_date.isoformat())
        if not roster:
            st.info("No students registered.")
        else:
            df = pd.DataFrame(roster)[["student_id", "name", "course", "department", "status", "attendance_time", "method"]]
            present_count = len(df[df["status"].isin(["Present", "Late"])])
            absent_count = len(df[df["status"] == "Absent"])
            st.caption(f"Showing {len(df)} students · Present/Late: {present_count} · Absent: {absent_count}")
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_manual:
        students = get_all_students()
        if not students:
            st.info("Register students before marking manual attendance.")
        else:
            with st.form("manual_att_form"):
                options = {f"{s['student_id']} — {s['name']}": s["student_id"] for s in students}
                stu_choice = st.selectbox("Student", list(options.keys()))
                att_date = st.date_input("Date", value=date.today(), key="manual_date")
                att_status = st.selectbox("Status", ["Present", "Late", "Absent"])
                notes = st.text_input("Notes (Optional)", placeholder="e.g. Verified in lab")
                if st.form_submit_button("Save Manual Entry", type="primary"):
                    ok, msg, _ = mark_attendance(
                        student_id=options[stu_choice],
                        status=att_status,
                        method="Manual",
                        confirmed=True,
                        attendance_date=att_date.isoformat(),
                        notes=notes,
                    )
                    st.success(msg) if ok else st.warning(msg)

    with tab_history:
        f1, f2, f3 = st.columns(3)
        with f1:
            use_date = st.checkbox("Filter by specific date", value=False)
            filter_date = st.date_input("Date", value=date.today(), key="hist_date") if use_date else None
        with f2:
            students = get_all_students()
            stu_list = ["ALL"] + [s["student_id"] for s in students]
            filter_stu = st.selectbox("Filter by Student ID", stu_list)
        with f3:
            filter_status = st.selectbox("Filter by Status", ["ALL", "Present", "Late", "Absent"])

        records = get_attendance_records(
            target_date=filter_date.isoformat() if filter_date else None,
            student_id=filter_stu,
            status_filter=filter_status,
        )
        if not records:
            st.info("No attendance records match the selected filters.")
        else:
            df = pd.DataFrame(records)[
                ["attendance_id", "attendance_date", "attendance_time", "student_id", "name", "course", "status", "method", "notes"]
            ]
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.subheader("Authorized Attendance Correction")
            with st.form("correction_form"):
                rec_options = {
                    f"#{r['attendance_id']} | {r['attendance_date']} | {r['student_id']} ({r['name']}) — Currently {r['status']}": r["attendance_id"]
                    for r in records
                }
                chosen_rec = st.selectbox("Select Record", list(rec_options.keys()))
                new_status = st.selectbox("Corrected Status", ["Present", "Late", "Absent"])
                reason = st.text_input("Correction Reason *", placeholder="e.g. Approved medical leave / misread entry")
                if st.form_submit_button("Apply Correction", type="primary"):
                    if not reason.strip():
                        st.error("Please provide an authorization reason for the audit note.")
                    else:
                        ok, msg = correct_attendance_record(rec_options[chosen_rec], new_status, reason)
                        st.success(msg) if ok else st.error(msg)


def page_reports() -> None:
    """5. Daily, Weekly, Monthly, and Student-wise analytics reports with charts."""
    st.markdown('<div class="section-title">Attendance Reports & Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Formula: Attendance Percentage = (Present Days / Total Working Days) × 100</div>',
        unsafe_allow_html=True,
    )

    tab_daily, tab_weekly, tab_monthly, tab_student = st.tabs(
        ["Daily Report", "Weekly Report", "Monthly Report", "Student-Wise Report"]
    )

    with tab_daily:
        d = st.date_input("Select Date", value=date.today(), key="rep_daily_date")
        df = get_daily_report(d.isoformat())
        if df.empty:
            st.info("No student records found.")
        else:
            p_cnt = int((df["status"].isin(["Present", "Late"])).sum())
            a_cnt = int((df["status"] == "Absent").sum())
            c1, c2 = st.columns(2)
            c1.metric("Present / Late", p_cnt)
            c2.metric("Absent", a_cnt)
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_weekly:
        end_d = st.date_input("Week Ending Date", value=date.today(), key="rep_week_date")
        df = get_weekly_report(end_d)
        if df.empty:
            st.info("No attendance records in this 7-day window.")
        else:
            st.bar_chart(df.set_index("name")["attendance_pct"])
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_monthly:
        m1, m2 = st.columns(2)
        year = m1.number_input("Year", min_value=2020, max_value=2035, value=date.today().year)
        month = m2.selectbox("Month", list(range(1, 13)), index=date.today().month - 1)
        df = get_monthly_report(int(year), int(month))
        if df.empty:
            st.info("No data available for the selected month.")
        else:
            st.bar_chart(df.set_index("name")["attendance_pct"])
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_student:
        q = st.text_input("Search Student Name or ID", key="rep_stu_search")
        df = get_student_wise_report(q)
        if df.empty:
            st.info("No matching student statistics found.")
        else:
            st.bar_chart(df.set_index("name")["attendance_pct"])
            st.dataframe(df, use_container_width=True, hide_index=True)


def page_excel_export() -> None:
    """6. Formatted multi-worksheet Excel exports via OpenPyXL."""
    st.markdown('<div class="section-title">Excel Export Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Download styled .xlsx workbooks containing separate Students and Attendance worksheets.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Complete Attendance Archive")
        st.write("Includes all recorded attendance entries plus the full Students directory sheet.")
        try:
            all_bytes = export_all_attendance_excel()
            st.download_button(
                "Download All Attendance (.xlsx)",
                data=all_bytes,
                file_name=f"attendance_all_{date.today().isoformat()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Excel generation failed: {exc}")

        st.divider()
        st.subheader("Student-Wise Summary Report")
        st.write("Cumulative Present Days, Working Days, and Attendance % per student.")
        try:
            stu_bytes = export_student_wise_excel()
            st.download_button(
                "Download Student-Wise Report (.xlsx)",
                data=stu_bytes,
                file_name=f"attendance_student_wise_{date.today().isoformat()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Excel generation failed: {exc}")

    with c2:
        st.subheader("Daily Attendance Export")
        daily_date = st.date_input("Select Report Date", value=date.today(), key="exp_daily_date")
        try:
            daily_bytes = export_daily_report_excel(daily_date.isoformat())
            st.download_button(
                f"Download Daily Report ({daily_date.isoformat()})",
                data=daily_bytes,
                file_name=f"attendance_daily_{daily_date.isoformat()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Excel generation failed: {exc}")

        st.divider()
        st.subheader("Monthly Attendance Export")
        mc1, mc2 = st.columns(2)
        exp_year = mc1.number_input("Year", 2020, 2035, date.today().year, key="exp_yr")
        exp_month = mc2.selectbox("Month", list(range(1, 13)), index=date.today().month - 1, key="exp_mo")
        try:
            monthly_bytes = export_monthly_report_excel(int(exp_year), int(exp_month))
            st.download_button(
                f"Download Monthly Report ({int(exp_year):04d}-{int(exp_month):02d})",
                data=monthly_bytes,
                file_name=f"attendance_monthly_{int(exp_year):04d}_{int(exp_month):02d}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Excel generation failed: {exc}")


def page_settings() -> None:
    """7. Confidence threshold, LBPH retraining, demo seeding, DB backup, and privacy info."""
    st.markdown('<div class="section-title">System Settings & Maintenance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Tune LBPH recognition sensitivity, seed sample data, back up SQLite storage, and manage privacy settings.</div>',
        unsafe_allow_html=True,
    )

    with st.form("settings_form"):
        inst_name = st.text_input("Institution / Department Name", value=get_setting("institution_name", "Department of Computer Science"))
        current_thresh = float(get_setting("confidence_threshold", "65.0"))
        new_thresh = st.slider(
            "LBPH Distance Threshold (Lower = Stricter Match, Recommended: 55–75)",
            min_value=35.0,
            max_value=100.0,
            value=current_thresh,
            step=1.0,
        )
        if st.form_submit_button("Save Settings", type="primary"):
            set_setting("institution_name", inst_name.strip())
            set_setting("confidence_threshold", str(new_thresh))
            st.success("Settings saved to SQLite.")

    st.divider()
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Model Training")
        if st.button("Retrain LBPH Model", use_container_width=True):
            ok, msg = train_lbph_model()
            st.success(msg) if ok else st.warning(msg)

    with c2:
        st.subheader("Sample Academic Data")
        if st.button("Load Demo Students & Attendance", use_container_width=True):
            ok, msg = seed_sample_data()
            st.success(msg) if ok else st.error(msg)

    with c3:
        st.subheader("SQLite Database Backup")
        if st.button("Create Local Backup Copy", use_container_width=True):
            ok, msg = backup_database()
            st.success(f"Backup created at: {msg}") if ok else st.error(msg)
        if DB_PATH.exists():
            st.download_button(
                "Download attendance.db",
                data=DB_PATH.read_bytes(),
                file_name=f"attendance_backup_{date.today().isoformat()}.db",
                mime="application/octet-stream",
                use_container_width=True,
            )


def main() -> None:
    section = render_sidebar()
    if section == "Dashboard":
        page_dashboard()
    elif section == "Face Recognition":
        page_face_recognition()
    elif section == "Student Management":
        page_student_management()
    elif section == "Attendance Records":
        page_attendance_records()
    elif section == "Reports":
        page_reports()
    elif section == "Excel Export":
        page_excel_export()
    elif section == "Settings":
        page_settings()


if __name__ == "__main__":
    main()
