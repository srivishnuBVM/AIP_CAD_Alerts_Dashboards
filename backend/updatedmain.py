# from fastapi import FastAPI  # type: ignore
# from fastapi.middleware.cors import CORSMiddleware  # type: ignore
# import pandas as pd
# import numpy as np

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# df = None  # Global variable to hold the preprocessed data


# @app.on_event("startup")
# async def load_data():
#     global df
#     df = pd.read_parquet("AttendanceDataProcessed.parquet")
#     df['STUDENT_ID'] = df['STUDENT_ID'].astype(int)


# @app.get("/students")
# async def get_students():
#     students = []
#     for _, row in df.iterrows():
#         sid = int(row["STUDENT_ID"])
#         location_id = int(row.get("LOCATION_ID", -1))
#         g = row.get("STUDENT_GRADE_LEVEL_2024", np.nan)

#         grade = "Unknown Grade" if pd.isna(g) else int(g)
#         if grade == -1:
#             grade_str = 'Pre-Kindergarten'
#         elif grade == 0:
#             grade_str = "Kindergarten"
#         elif grade == 1:
#             grade_str = "1st Grade"
#         elif grade == 2:
#             grade_str = "2nd Grade"
#         elif grade == 3:
#             grade_str = "3rd Grade"
#         elif grade >= 11:
#             grade_str = f"{grade}th Grade"
#         else:
#             suffix = {1: "st", 2: "nd", 3: "rd"}.get(grade % 10, "th")
#             grade_str = f"{grade}{suffix} Grade"

#         school_name = row.get("SCHOOL_NAME", "Unknown School")
#         district_name = row.get("DISTRICT_NAME", "Unknown District")

#         students.append({
#             "id": str(sid),
#             "grade": grade_str,
#             "locationId": location_id,
#             "schoolName": school_name,
#             "districtName": district_name
#         })
#     students.sort(key=lambda x: x["id"])
#     return students


# @app.get("/students/{student_id}/details")
# async def get_student_details(student_id: int):
#     subset = df[df["STUDENT_ID"] == student_id]
#     if subset.empty:
#         return {}
#     s = subset.iloc[0]
#     a24 = int(round(float(s["ATTENDANCE_PERCENT_2024"])))
#     p25 = int(round(float(s["Predicted_2025"])))
#     prob25 = float(round(s["Probabilities_2025"], 4))

#     if p25 >= 90:
#         level, color, desc = "Low", "#4CAF50", "Attendance is strong."
#     elif p25 < 80:
#         level, color, desc = "High", "#FF5A5F", "Requires immediate intervention."
#     else:
#         level, color, desc = "Medium", "#FFB547", "May require intervention."

#     total24 = s.get("TOTAL_DAYS_ENROLLED_2024", 0)
#     days25 = int(total24)
#     abs_val = round((100 - p25) / 100 * days25 * 2) / 2
#     unexc_prop = s.get("UNEXCUSED_ABSENT_PROPORTION_2024", 0.5)
#     unexc = round(abs_val * unexc_prop * 2) / 2
#     exc = abs_val - unexc

#     return {
#         "attendance2024": a24,
#         "predicted2025": p25,
#         "probability2025": prob25,
#         "risk": {
#             "level": level,
#             "color": color,
#             "description": desc
#         },
#         "predictedAttendance": {
#             "year": "2025",
#             "attendanceRate": p25,
#             "absences": unexc + exc,
#             "excused": exc,
#             "total": days25
#         }
#     }


# @app.get("/students/{student_id}/metrics")
# async def get_student_metrics(student_id: int):
#     subset = df[df["STUDENT_ID"] == student_id]
#     if subset.empty:
#         return []
#     s = subset.iloc[0]
#     out = []
#     for year in range(2020, 2025):
#         if year == 2025:
#             break
#         a = s.get(f"ATTENDANCE_PERCENT_{year}")
#         t = s.get(f"TOTAL_DAYS_ENROLLED_{year}")
#         ab = s.get(f"TOTAL_DAYS_ABSENT_{year}")
#         ex = s.get(f"TOTAL_DAYS_EXCUSED_ABSENT_{year}")
#         if pd.notna(a) and pd.notna(t) and t > 0:
#             out.append({
#                 "year": str(year),
#                 "attendanceRate": int(round(a)),
#                 "absences": ab if pd.notna(ab) else None,
#                 "excused": ex if pd.notna(ex) else None,
#                 "lates": 0,
#                 "total": int(t)
#             })
#     return sorted(out, key=lambda x: x["year"])


# @app.get("/students/{student_id}/trend")
# async def get_student_trend(student_id: int):
#     subset = df[df["STUDENT_ID"] == student_id]
#     if subset.empty:
#         return []
#     s = subset.iloc[0]
#     trend = []
#     for year in range(2020, 2025):
#         if year == 2025:
#             break
#         a = s.get(f"ATTENDANCE_PERCENT_{year}")
#         t = s.get(f"TOTAL_DAYS_ENROLLED_{year}", 0)
#         if pd.notna(a) and t > 0:
#             trend.append({"year": str(year), "value": int(round(a)), "isPredicted": False})
#     p = s["Predicted_2025"]
#     trend.append({"year": "2025", "value": int(round(p)), "isPredicted": True})
#     return trend


# cached code
from fastapi import FastAPI  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import ORJSONResponse  # Faster JSON serialization
import pandas as pd
import numpy as np

app = FastAPI(default_response_class=ORJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = None            # Global variable to hold the DataFrame
cached_students = []  # Global variable to cache precomputed student list


@app.on_event("startup")
async def load_data():
    global df, cached_students
    df = pd.read_parquet("AttendanceDataProcessed.parquet")
    df['STUDENT_ID'] = df['STUDENT_ID'].astype(int)

    cached_students = []  # Precompute the student list here
    for _, row in df.iterrows():
        sid = int(row["STUDENT_ID"])
        location_id = int(row.get("LOCATION_ID", -1))
        g = row.get("STUDENT_GRADE_LEVEL_2024", np.nan)

        grade = "Unknown Grade" if pd.isna(g) else int(g)
        if grade == -1:
            grade_str = 'Pre-Kindergarten'
        elif grade == 0:
            grade_str = "Kindergarten"
        elif grade == 1:
            grade_str = "1st Grade"
        elif grade == 2:
            grade_str = "2nd Grade"
        elif grade == 3:
            grade_str = "3rd Grade"
        elif grade >= 11:
            grade_str = f"{grade}th Grade"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(grade % 10, "th")
            grade_str = f"{grade}{suffix} Grade"

        school_name = row.get("SCHOOL_NAME", "Unknown School")
        district_name = row.get("DISTRICT_NAME", "Unknown District")

        cached_students.append({
            "id": str(sid),
            "grade": grade_str,
            "locationId": location_id,
            "schoolName": school_name,
            "districtName": district_name
        })

    cached_students.sort(key=lambda x: x["id"])


@app.get("/students")
def get_students():
    return cached_students


@app.get("/students/{student_id}/details")
def get_student_details(student_id: int):
    subset = df[df["STUDENT_ID"] == student_id]
    if subset.empty:
        return {}
    s = subset.iloc[0]
    a24 = int(round(float(s["ATTENDANCE_PERCENT_2024"])))
    p25 = int(round(float(s["Predicted_2025"])))
    prob25 = float(round(s["Probabilities_2025"], 4))

    if p25 >= 90:
        level, color, desc = "Low", "#4CAF50", "Attendance is strong."
    elif p25 < 80:
        level, color, desc = "High", "#FF5A5F", "Requires immediate intervention."
    else:
        level, color, desc = "Medium", "#FFB547", "May require intervention."

    total24 = s.get("TOTAL_DAYS_ENROLLED_2024", 0)
    days25 = int(total24)
    abs_val = round((100 - p25) / 100 * days25 * 2) / 2
    unexc_prop = s.get("UNEXCUSED_ABSENT_PROPORTION_2024", 0.5)
    if np.isnan(unexc_prop):
        unexc_prop = 0.5
    unexc = round(abs_val * unexc_prop * 2) / 2
    exc = abs_val - unexc

    return {
        "attendance2024": a24,
        "predicted2025": p25,
        "probability2025": prob25,
        "risk": {
            "level": level,
            "color": color,
            "description": desc
        },
        "predictedAttendance": {
            "year": "2025",
            "attendanceRate": p25,
            "absences": unexc + exc,
            "excused": exc,
            "total": days25
        }
    }


@app.get("/students/{student_id}/metrics")
def get_student_metrics(student_id: int):
    subset = df[df["STUDENT_ID"] == student_id]
    if subset.empty:
        return []
    s = subset.iloc[0]
    out = []
    for year in range(2020, 2025):
        if year == 2025:
            break
        a = s.get(f"ATTENDANCE_PERCENT_{year}")
        t = s.get(f"TOTAL_DAYS_ENROLLED_{year}")
        ab = s.get(f"TOTAL_DAYS_ABSENT_{year}")
        ex = s.get(f"TOTAL_DAYS_EXCUSED_ABSENT_{year}")
        if pd.notna(a) and pd.notna(t) and t > 0:
            out.append({
                "year": str(year),
                "attendanceRate": int(round(a)),
                "absences": ab if pd.notna(ab) else None,
                "excused": ex if pd.notna(ex) else None,
                "lates": 0,
                "total": int(t)
            })
    return sorted(out, key=lambda x: x["year"])


@app.get("/students/{student_id}/trend")
def get_student_trend(student_id: int):
    subset = df[df["STUDENT_ID"] == student_id]
    if subset.empty:
        return []
    s = subset.iloc[0]
    trend = []
    for year in range(2020, 2025):
        if year == 2025:
            break
        a = s.get(f"ATTENDANCE_PERCENT_{year}")
        t = s.get(f"TOTAL_DAYS_ENROLLED_{year}", 0)
        if pd.notna(a) and t > 0:
            trend.append({"year": str(year), "value": int(round(a)), "isPredicted": False})
    p = s["Predicted_2025"]
    trend.append({"year": "2025", "value": int(round(p)), "isPredicted": True})
    return trend
