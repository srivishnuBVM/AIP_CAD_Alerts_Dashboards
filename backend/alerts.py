from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from functools import lru_cache
import traceback
import asyncio
import logging
import io
import time
import threading
import concurrent.futures
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("attendance_api")

# Global data store
class DataStore:
    df = None
    last_loaded = None
    loading = False
    load_error = None
    indices = {}
    is_ready = False

data_store = DataStore()

# Setup background loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start data loading in background
    background_thread = threading.Thread(target=load_and_process_data)
    background_thread.daemon = True
    background_thread.start()
    
    logger.info("Starting data loading in background...")
    yield
    # Cleanup (if needed)
    logger.info("Shutting down application")

# Create the FastAPI app with lifespan
app = FastAPI(
    title="Attendance Analysis API",
    description="""
    This API provides comprehensive attendance analysis and reporting capabilities with filtering.
    It includes endpoints for downloading various reports and analyzing student attendance patterns.
    """,
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Attendance Analysis API",
        version="1.2.0",
        description="API for analyzing and reporting student attendance patterns with filtering",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080"  # Add this too in case you're using 127.0.0.1
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Type", "Content-Length"]
)



# Data models
class SummaryStatistics(BaseModel):
    total_students: int
    at_risk_students: int
    at_risk_percentage: float
    below_85_students: int
    below_85_percentage: float
    critical_risk_students: int
    critical_risk_percentage: float
    tier4_students: int
    tier4_percentage: float
    tier3_students: int
    tier3_percentage: float
    tier2_students: int
    tier2_percentage: float
    tier1_students: int
    tier1_percentage: float
    safe_students: int
    safe_percentage: float

class KeyInsight(BaseModel):
    insight: str

class Recommendation(BaseModel):
    recommendation: str

class PriorityDistrict(BaseModel):
    district_name: str
    ai_priority_score: float
    at_risk_percentage: float
    tier4_percentage: float

class PrioritySchool(BaseModel):
    school_id: str
    school_name: str
    district_name: str
    grade_level: str
    ai_priority_score: float
    at_risk_percentage: float
    potential_improvement: float

class AnalysisResponse(BaseModel):
    summary_statistics: SummaryStatistics
    key_insights: List[KeyInsight]
    recommendations: List[Recommendation]
    priority_districts: List[PriorityDistrict]
    priority_schools: List[PrioritySchool]

class StudentRecord(BaseModel):
    student_id: str
    school_id: str
    district_name: str
    grade_level: str
    attendance_percentage: float
    tier: str
    risk_score: float
    risk_level: str
    risk_factors: List[str]
    recommendations: List[str]

class FilterOptions(BaseModel):
    districts: List[Dict[str, str]]
    schools: List[Dict[str, str]]
    grades: List[Dict[str, str]]

class ApiStatus(BaseModel):
    status: str
    data_loaded: bool
    record_count: Optional[int] = None
    last_loaded: Optional[str] = None
    loading: bool
    error: Optional[str] = None
    uptime: str
    memory_usage_mb: float

# Function to load and process data
def load_and_process_data():
    """Load and process data in background"""
    try:
        data_store.loading = True
        data_store.is_ready = False
        data_store.load_error = None
        
        logger.info("Starting data loading...")
        start_time = time.time()
        
        # Load the data
        df = load_excel_file()
        
        if df is None or len(df) == 0:
            data_store.load_error = "Failed to load data"
            data_store.loading = False
            return
        
        # Process the data - calculate attendance, risk scores, etc.
        logger.info(f"Processing {len(df)} records...")
        df['Attendance_2024'] = (df['TOTAL_DAYS_PRESENT_2024'] / df['TOTAL_DAYS_ENROLLED_2024']) * 100
        
        # Create a worker pool for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Execute these in parallel
            attendance_future = executor.submit(lambda: df['Attendance_2024'].values)
            attendance_values = attendance_future.result()
            
            # Calculate risk scores and tiers
            df['RISK_SCORE'] = 100 - attendance_values  # Simplified risk calculation
            df['RISK_LEVEL'] = pd.cut(
                df['RISK_SCORE'],
                bins=[0, 20, 40, 60, 80, 100],
                labels=['Safe', 'Low', 'Medium', 'High', 'Critical']
            )
            df['TIER'] = pd.cut(
                attendance_values,
                bins=[0, 70, 75, 80, 85, 100],
                labels=['Tier 4', 'Tier 3', 'Tier 2', 'Tier 1', 'Safe'],
                right=False
            )
        
        # Create indices for faster filtering
        logger.info("Creating indices for faster filtering...")
        data_store.indices = {
            'DISTRICT_NAME': df['DISTRICT_NAME'].str.upper().to_dict(),
            'STUDENT_GRADE_LEVEL_2024': df['STUDENT_GRADE_LEVEL_2024'].astype(str).to_dict()
        }
        
        if 'SCHOOL_NAME' in df.columns:
            data_store.indices['SCHOOL_NAME'] = df['SCHOOL_NAME'].str.upper().to_dict()
        
        # Calculate and store common aggregations
        logger.info("Pre-calculating common aggregations...")
        
        # Store the data
        data_store.df = df
        data_store.last_loaded = datetime.now()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Data processing completed in {processing_time:.2f} seconds")
        
        data_store.is_ready = True
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        logger.error(traceback.format_exc())
        data_store.load_error = str(e)
    finally:
        data_store.loading = False

def load_excel_file():
    """Load the Excel file"""
    try:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "merged_output.xlsx")
        logger.info(f"Loading data from: {data_file}")
        
            
        # Use optimized Excel loading with specific dtypes
        logger.info("Starting Excel file load with optimized settings...")
        df = pd.read_excel(
            data_file,
            engine='openpyxl',
            dtype={
                'STUDENT_ID': str,
                'DISTRICT_NAME': str,
                'SCHOOL_NAME': str,
                'STUDENT_GRADE_LEVEL_2024': str,
                'TOTAL_DAYS_PRESENT_2024': float,
                'TOTAL_DAYS_ENROLLED_2024': float
            }
        )
        
        logger.info(f"Successfully loaded {len(df)} rows from Excel file")
        return df
    except Exception as e:
        logger.error(f"Error loading Excel file: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def filter_data(df, district_name=None, grade_level=None, school_name=None):
    """Filter data based on provided parameters using optimized approach"""
    try:
        start_time = time.time()
        
        # Create a boolean mask for filtering
        mask = pd.Series(True, index=df.index)
        
        if district_name:
            district_name_upper = district_name.upper()
            mask &= df['DISTRICT_NAME'].str.upper() == district_name_upper
        
        if grade_level is not None:
            grade_level_str = str(grade_level)
            mask &= df['STUDENT_GRADE_LEVEL_2024'].astype(str) == grade_level_str
        
        if school_name and 'SCHOOL_NAME' in df.columns:
            school_name_upper = school_name.upper()
            mask &= df['SCHOOL_NAME'].str.upper() == school_name_upper
        
        filtered_df = df[mask]
        
        logger.info(f"Filtering completed in {time.time() - start_time:.4f} seconds, returning {len(filtered_df)} rows")
        return filtered_df
    except Exception as e:
        logger.error(f"Error in filter_data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Helper functions
def calculate_risk_score(attendance_percentage: float, risk_factors: List[str]) -> float:
    """Calculate risk score based on attendance percentage and risk factors"""
    base_score = 100 - attendance_percentage
    risk_factor_points = len(risk_factors) * 5
    return min(100, base_score + risk_factor_points)

def get_risk_level(risk_score: float) -> str:
    """Convert risk score to risk level"""
    if risk_score >= 80: return "Critical"
    elif risk_score >= 60: return "High"
    elif risk_score >= 40: return "Medium"
    elif risk_score >= 20: return "Low"
    else: return "Safe"

def get_tier(attendance_percentage: float) -> str:
    """Get tier based on attendance percentage"""
    if attendance_percentage < 70: return "Tier 4"
    elif attendance_percentage < 75: return "Tier 3"
    elif attendance_percentage < 80: return "Tier 2"
    elif attendance_percentage < 85: return "Tier 1"
    else: return "Safe"

#working 
@app.get("/api/analysis", response_model=AnalysisResponse)
async def get_analysis(
    district_name: Optional[str] = None,
    grade_level: Optional[str] = None,
    school_name: Optional[str] = None
):
    """Get comprehensive analysis of attendance data with filtering support"""
    if not data_store.is_ready:
        raise HTTPException(status_code=503, 
                           detail="Data is still being loaded. Please try again shortly.")
    
    try:
        start_time = time.time()
        df = data_store.df
        
        # Apply filters if provided
        if district_name or grade_level is not None or school_name:
            df = filter_data(df, district_name, grade_level, school_name)
            
        if len(df) == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for filters: district={district_name}, grade={grade_level}, school={school_name}"
            )
        
        # Calculate basic statistics
        total_students = len(df)
        
        # Use efficient vectorized operations
        attendance_series = df['Attendance_2024']
        risk_series = df['RISK_SCORE']
        tier_series = df['TIER']
        risk_level_series = df['RISK_LEVEL']
        
        # Calculate metrics
        below_85_mask = attendance_series < 85
        below_85_students = below_85_mask.sum()
        
        critical_risk_mask = risk_level_series == 'Critical'
        critical_risk_students = critical_risk_mask.sum()
        
        # Calculate tier distributions using value_counts (more efficient)
        tier_counts = tier_series.value_counts()
        tier4 = tier_counts.get('Tier 4', 0)
        tier3 = tier_counts.get('Tier 3', 0)
        tier2 = tier_counts.get('Tier 2', 0)
        tier1 = tier_counts.get('Tier 1', 0)
        safe = tier_counts.get('Safe', 0)
        
        # Create summary statistics
        summary = SummaryStatistics(
            total_students=total_students,
            at_risk_students=below_85_students,
            at_risk_percentage=(below_85_students / total_students) * 100 if total_students > 0 else 0,
            below_85_students=below_85_students,
            below_85_percentage=(below_85_students / total_students) * 100 if total_students > 0 else 0,
            critical_risk_students=critical_risk_students,
            critical_risk_percentage=(critical_risk_students / total_students) * 100 if total_students > 0 else 0,
            tier4_students=tier4,
            tier4_percentage=(tier4 / total_students) * 100 if total_students > 0 else 0,
            tier3_students=tier3,
            tier3_percentage=(tier3 / total_students) * 100 if total_students > 0 else 0,
            tier2_students=tier2,
            tier2_percentage=(tier2 / total_students) * 100 if total_students > 0 else 0,
            tier1_students=tier1,
            tier1_percentage=(tier1 / total_students) * 100 if total_students > 0 else 0,
            safe_students=safe,
            safe_percentage=(safe / total_students) * 100 if total_students > 0 else 0
        )
        
        # Generate insights
        tier4_pct = (tier4/total_students*100) if total_students > 0 else 0
        tier3_pct = (tier3/total_students*100) if total_students > 0 else 0
        tier2_pct = (tier2/total_students*100) if total_students > 0 else 0
        tier1_pct = (tier1/total_students*100) if total_students > 0 else 0
        safe_pct = (safe/total_students*100) if total_students > 0 else 0
        
        insights = [
            KeyInsight(insight=f"Tier 4 Students: {tier4} students ({tier4_pct:.1f}%) have attendance below 70%"),
            KeyInsight(insight=f"Tier 3 Students: {tier3} students ({tier3_pct:.1f}%) have attendance between 70-75%"),
            KeyInsight(insight=f"Tier 2 Students: {tier2} students ({tier2_pct:.1f}%) have attendance between 75-80%"),
            KeyInsight(insight=f"Tier 1 Students: {tier1} students ({tier1_pct:.1f}%) have attendance between 80-85%"),
            KeyInsight(insight=f"Safe Students: {safe} students ({safe_pct:.1f}%) have attendance above 85%")
        ]
        
        # Generate recommendations based on risk levels
        risk_counts = risk_level_series.value_counts()
        critical = risk_counts.get('Critical', 0)
        high = risk_counts.get('High', 0)
        medium = risk_counts.get('Medium', 0)
        low = risk_counts.get('Low', 0)
        
        recommendations = [
            Recommendation(recommendation=f"Focus immediate interventions on {critical} students at Critical risk level"),
            Recommendation(recommendation=f"Implement targeted support for {high} students at High risk level"),
            Recommendation(recommendation=f"Monitor {medium} students at Medium risk level"),
            Recommendation(recommendation=f"Develop preventive measures for {low} students at Low risk level")
        ]
        
        # Note: Priority districts and schools calculation code has been removed
        
        # Log processing time for analysis
        logger.info(f"Analysis completed in {time.time() - start_time:.4f} seconds")
        
        return AnalysisResponse(
            summary_statistics=summary,
            key_insights=insights,
            recommendations=recommendations,
            priority_districts=[],  # Return empty lists instead of calculated values
            priority_schools=[]
        )
            
    except Exception as e:
        logger.error(f"Error in get_analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/global-analysis", response_model=AnalysisResponse)
async def get_global_analysis():
    """
    Get comprehensive analysis of attendance data for the entire dataset without filters.
    Returns the same response structure as the /api/analysis endpoint but always includes all records.
    """
    if not data_store.is_ready:
        raise HTTPException(status_code=503, 
                           detail="Data is still being loaded. Please try again shortly.")
    try:
        logger.info("Starting global analysis for entire dataset")
        start_time = time.time()
        # Use the entire dataset without any filters
        df = data_store.df
        if len(df) == 0:
            raise HTTPException(
                status_code=404, 
                detail="No data available in the dataset"
            )
        # Calculate basic statistics
        total_students = len(df)
        # Use efficient vectorized operations
        attendance_series = df['Attendance_2024']
        risk_series = df['RISK_SCORE']
        tier_series = df['TIER']
        risk_level_series = df['RISK_LEVEL']
        # Calculate metrics
        below_85_mask = attendance_series < 85
        below_85_students = below_85_mask.sum()
        critical_risk_mask = risk_level_series == 'Critical'
        critical_risk_students = critical_risk_mask.sum()
        # Calculate tier distributions using value_counts (more efficient)
        tier_counts = tier_series.value_counts()
        tier4 = tier_counts.get('Tier 4', 0)
        tier3 = tier_counts.get('Tier 3', 0)
        tier2 = tier_counts.get('Tier 2', 0)
        tier1 = tier_counts.get('Tier 1', 0)
        safe = tier_counts.get('Safe', 0)
        # Create summary statistics
        summary = SummaryStatistics(
            total_students=total_students,
            at_risk_students=below_85_students,
            at_risk_percentage=(below_85_students / total_students) * 100 if total_students > 0 else 0,
            below_85_students=below_85_students,
            below_85_percentage=(below_85_students / total_students) * 100 if total_students > 0 else 0,
            critical_risk_students=critical_risk_students,
            critical_risk_percentage=(critical_risk_students / total_students) * 100 if total_students > 0 else 0,
            tier4_students=tier4,
            tier4_percentage=(tier4 / total_students) * 100 if total_students > 0 else 0,
            tier3_students=tier3,
            tier3_percentage=(tier3 / total_students) * 100 if total_students > 0 else 0,
            tier2_students=tier2,
            tier2_percentage=(tier2 / total_students) * 100 if total_students > 0 else 0,
            tier1_students=tier1,
            tier1_percentage=(tier1 / total_students) * 100 if total_students > 0 else 0,
            safe_students=safe,
            safe_percentage=(safe / total_students) * 100 if total_students > 0 else 0
        )
        # Generate insights
        tier4_pct = (tier4/total_students*100) if total_students > 0 else 0
        tier3_pct = (tier3/total_students*100) if total_students > 0 else 0
        tier2_pct = (tier2/total_students*100) if total_students > 0 else 0
        tier1_pct = (tier1/total_students*100) if total_students > 0 else 0
        safe_pct = (safe/total_students*100) if total_students > 0 else 0
        insights = [
            KeyInsight(insight=f"Tier 4 Students: {tier4} students ({tier4_pct:.1f}%) have attendance below 70%"),
            KeyInsight(insight=f"Tier 3 Students: {tier3} students ({tier3_pct:.1f}%) have attendance between 70-75%"),
            KeyInsight(insight=f"Tier 2 Students: {tier2} students ({tier2_pct:.1f}%) have attendance between 75-80%"),
            KeyInsight(insight=f"Tier 1 Students: {tier1} students ({tier1_pct:.1f}%) have attendance between 80-85%"),
            KeyInsight(insight=f"Safe Students: {safe} students ({safe_pct:.1f}%) have attendance above 85%")
        ]
        # Generate recommendations based on risk levels
        risk_counts = risk_level_series.value_counts()
        critical = risk_counts.get('Critical', 0)
        high = risk_counts.get('High', 0)
        medium = risk_counts.get('Medium', 0)
        low = risk_counts.get('Low', 0)
        recommendations = [
            Recommendation(recommendation=f"Focus immediate interventions on {critical} students at Critical risk level"),
            Recommendation(recommendation=f"Implement targeted support for {high} students at High risk level"),
            Recommendation(recommendation=f"Monitor {medium} students at Medium risk level"),
            Recommendation(recommendation=f"Develop preventive measures for {low} students at Low risk level")
        ]
        
        # Note: Priority districts and schools calculation code has been removed
        
        # Log processing time for analysis
        logger.info(f"Global analysis completed in {time.time() - start_time:.4f} seconds")
        return AnalysisResponse(
            summary_statistics=summary,
            key_insights=insights,
            recommendations=recommendations,
            priority_districts=[],  # Return empty lists instead of calculated values
            priority_schools=[]
        )
    except Exception as e:
        logger.error(f"Error in get_global_analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
        
#working
# @app.get("/api/filter-options", response_model=FilterOptions)
# async def get_filter_options():
#     """Get unique values for dropdowns (districts, schools, grades)"""
#     if not data_store.is_ready:
#         raise HTTPException(status_code=503, 
#                           detail="Data is still being loaded. Please try again shortly.")
    
#     try:
#         df = data_store.df
        
#         # Get unique district names
#         districts = df['DISTRICT_NAME'].unique().tolist()
#         districts = [{"value": d, "label": d} for d in sorted(districts)]
        
#         # Get unique school names if available
#         schools = []
#         if 'SCHOOL_NAME' in df.columns:
#             schools = df['SCHOOL_NAME'].unique().tolist()
#             schools = [{"value": s, "label": s} for s in sorted(schools)]
        
#         # Get unique grade levels
#         grades = df['STUDENT_GRADE_LEVEL_2024'].unique().tolist()
#         # Convert any numeric grades to strings
#         grades = [str(g) for g in grades]
#         grades = [{"value": g, "label": g} for g in sorted(grades)]
        
#         return FilterOptions(
#             districts=districts,
#             schools=schools,
#             grades=grades
#         )
        
#     except Exception as e:
#         logger.error(f"Error retrieving filter options: {str(e)}")
#         logger.error(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Error retrieving filter options: {str(e)}")


@app.get("/api/filter-options", response_model=FilterOptions)
async def get_filter_options():
    """Get hierarchical filter options for districts, schools, and grades"""
    if not data_store.is_ready:
        raise HTTPException(status_code=503, 
                          detail="Data is still being loaded. Please try again shortly.")
    
    try:
        df = data_store.df
        
        # Create hierarchical structure for districts, schools, and grades
        district_map = {}
        
        # Group by district, then school, then grade
        for district_name, district_df in df.groupby('DISTRICT_NAME'):
            schools_in_district = []
            
            for school_name, school_df in district_df.groupby('SCHOOL_NAME'):
                # Get grades for this school
                grades_in_school = school_df['STUDENT_GRADE_LEVEL_2024'].unique().tolist()
                # Convert any numeric grades to strings
                grades_in_school = [str(g) for g in grades_in_school]
                grades_in_school = [{"value": g, "label": g} for g in sorted(grades_in_school)]
                
                # Add school with its grades
                schools_in_district.append({
                    "value": school_name,
                    "label": school_name,
                    "district": district_name,
                    "grades": grades_in_school
                })
            
            # Add district with its schools
            district_map[district_name] = {
                "value": district_name,
                "label": district_name,
                "schools": sorted(schools_in_district, key=lambda x: x["label"])
            }
        
        # Convert to list and sort
        districts = [district_map[d] for d in sorted(district_map.keys())]
        
        # Also create flat lists for initial loading
        flat_districts = [{"value": d, "label": d} for d in sorted(district_map.keys())]
        
        flat_schools = []
        for district in districts:
            for school in district["schools"]:
                flat_schools.append({
                    "value": school["value"],
                    "label": school["label"],
                    "district": school["district"]
                })
        flat_schools = sorted(flat_schools, key=lambda x: x["label"])
        
        # Get all unique grades across all schools
        all_grades = set()
        for district in districts:
            for school in district["schools"]:
                for grade in school["grades"]:
                    all_grades.add(grade["value"])
        flat_grades = [{"value": g, "label": g} for g in sorted(all_grades)]
        
        return FilterOptions(
            districts=flat_districts,
            schools=flat_schools,
            grades=flat_grades,
            hierarchical_data=districts
        )
        
    except Exception as e:
        logger.error(f"Error retrieving filter options: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving filter options: {str(e)}")


@app.get("/api/download/{report_type}")
async def download_report(
    report_type: str,
    district_name: Optional[str] = None,
    grade_level: Optional[str] = None,
    school_name: Optional[str] = None
):
    """Download various types of reports with filtering"""
    if not data_store.is_ready:
        raise HTTPException(status_code=503, 
                           detail="Data is still being loaded. Please try again shortly.")
    
    try:
        start_time = time.time()
        df = data_store.df
        
        # Convert grade_level to appropriate type if provided
        grade_level_param = None
        if grade_level is not None:
            grade_level_param = grade_level
        
        # Apply filters if provided
        if district_name or grade_level_param is not None or school_name:
            df = filter_data(df, district_name, grade_level_param, school_name)
            
        if len(df) == 0:
            raise HTTPException(status_code=404, detail="No data found for the selected filters")
        
        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add filter info to filename
        filter_info = ""
        if district_name:
            filter_info += f"_dist_{district_name}"
        if grade_level:
            filter_info += f"_grade_{grade_level}"
        if school_name:
            filter_info += f"_school_{school_name}"
            
        # Generate appropriate filename
        filename = f"attendance_report_{timestamp}{filter_info}"
        
        # Generate the appropriate report based on report_type
        if report_type.lower() == "summary":
            # Create summary report with aggregated data
            summary_df = generate_summary_report(df)
            output = io.BytesIO()
            summary_df.to_excel(output, index=False)
            output.seek(0)
            
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"{filename}_summary.xlsx"
            
        elif report_type.lower() == "detailed":
            # Create detailed student-level report
            detailed_df = generate_detailed_report(df)
            output = io.BytesIO()
            detailed_df.to_excel(output, index=False)
            output.seek(0)
            
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"{filename}_detailed.xlsx"
            
        elif report_type.lower() == "below_85":
            # Filter students with attendance below 85%
            below_85_df = df[df['Attendance_2024'] < 85].copy()
            
            if len(below_85_df) == 0:
                raise HTTPException(status_code=404, detail="No students found with attendance below 85%")
                
            # Generate detailed report for these students
            below_85_report = generate_detailed_report(below_85_df)
            output = io.BytesIO()
            below_85_report.to_excel(output, index=False)
            output.seek(0)
            
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"{filename}_below_85.xlsx"
            
        elif report_type.lower() == "csv":
            # Export raw data as CSV
            output = io.StringIO()
            df.to_csv(output, index=False)
            output_bytes = io.BytesIO(output.getvalue().encode())
            
            content_type = "text/csv"
            filename = f"{filename}_raw.csv"
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid report type: {report_type}")
        
        logger.info(f"Report generation completed in {time.time() - start_time:.4f} seconds")
        
        # Stream the file to the client
        headers = {
            "Content-Disposition": f"attachment; filename={filename}"
        }
        
        if report_type.lower() == "csv":
            return StreamingResponse(output_bytes, media_type=content_type, headers=headers)
        else:
            return StreamingResponse(output, media_type=content_type, headers=headers)
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

def generate_summary_report(df):
    """Generate a summary report with aggregated data"""
    # Group by district, school, and grade level
    group_cols = ['DISTRICT_NAME', 'STUDENT_GRADE_LEVEL_2024']
    if 'SCHOOL_NAME' in df.columns:
        group_cols.insert(1, 'SCHOOL_NAME')
    
    summary = df.groupby(group_cols).agg({
        'STUDENT_ID': 'count',
        'Attendance_2024': ['mean', 'min', 'max', 'std'],
        'RISK_SCORE': ['mean', 'min', 'max']
    }).reset_index()
    
    # Flatten multi-level columns
    summary.columns = [' '.join(col).strip() for col in summary.columns.values]
    
    # Add tier counts
    tiers = ['Tier 4', 'Tier 3', 'Tier 2', 'Tier 1', 'Safe']
    for tier in tiers:
        tier_counts = df.groupby(group_cols)['TIER'].apply(
            lambda x: (x == tier).sum()
        ).reset_index(name=f'{tier} Count')
        summary = pd.merge(summary, tier_counts, on=group_cols)
    
    # Calculate percentages
    for tier in tiers:
        summary[f'{tier} %'] = (summary[f'{tier} Count'] / summary['STUDENT_ID count'] * 100).round(2)
    
    # Format column names for better readability
    summary.rename(columns={
        'STUDENT_ID count': 'Total Students',
        'Attendance_2024 mean': 'Avg Attendance %',
        'Attendance_2024 min': 'Min Attendance %',
        'Attendance_2024 max': 'Max Attendance %',
        'Attendance_2024 std': 'Std Dev Attendance',
        'RISK_SCORE mean': 'Avg Risk Score',
        'RISK_SCORE min': 'Min Risk Score',
        'RISK_SCORE max': 'Max Risk Score'
    }, inplace=True)
    
    return summary

def generate_detailed_report(df):
    """Generate a detailed student-level report with risk factors and recommendations"""
    # Select and rename columns for the report
    report_df = df.copy()
    
    # Calculate risk factors (for demonstration - in production these would be more sophisticated)
    risk_factors = []
    recommendations = []
    
    for _, row in report_df.iterrows():
        # Generate risk factors based on attendance
        student_risk_factors = []
        student_recommendations = []
        
        attendance = row['Attendance_2024']
        if attendance < 70:
            student_risk_factors.append("Severe chronic absenteeism")
            student_recommendations.append("Immediate intervention required")
            student_recommendations.append("Family engagement specialist referral")
        elif attendance < 80:
            student_risk_factors.append("Chronic absenteeism")
            student_recommendations.append("Attendance improvement plan")
        elif attendance < 85:
            student_risk_factors.append("At risk of chronic absenteeism")
            student_recommendations.append("Early warning monitoring")
        
        risk_factors.append("|".join(student_risk_factors) if student_risk_factors else "None")
        recommendations.append("|".join(student_recommendations) if student_recommendations else "Continue monitoring")
    
    # Add calculated columns to the dataframe
    report_df['Risk Factors'] = risk_factors
    report_df['Recommendations'] = recommendations
    
    # Select and rename relevant columns
    columns_to_include = [
        'STUDENT_ID', 'DISTRICT_NAME', 'STUDENT_GRADE_LEVEL_2024', 
        'Attendance_2024', 'TIER', 'RISK_SCORE', 'RISK_LEVEL',
        'Risk Factors', 'Recommendations'
    ]
    
    if 'SCHOOL_NAME' in report_df.columns:
        columns_to_include.insert(2, 'SCHOOL_NAME')
    
    report_df = report_df[columns_to_include]
    
    # Rename columns for better readability
    column_renames = {
        'STUDENT_ID': 'Student ID',
        'DISTRICT_NAME': 'District',
        'SCHOOL_NAME': 'School',
        'STUDENT_GRADE_LEVEL_2024': 'Grade',
        'Attendance_2024': 'Attendance %',
        'RISK_SCORE': 'Risk Score',
        'RISK_LEVEL': 'Risk Level'
    }
    
    # Only rename columns that exist in the DataFrame
    rename_dict = {k: v for k, v in column_renames.items() if k in report_df.columns}
    report_df.rename(columns=rename_dict, inplace=True)
    
    return report_df


@app.get("/api/download/below85")
async def download_below_85_report(
    district_name: Optional[str] = None,
    grade_level: Optional[str] = None,
    school_name: Optional[str] = None,
    format: str = Query("xlsx", description="Output format: xlsx or csv")
):
    """
    Download report of students with attendance below 85% with filtering
    
    Args:
        district_name: Optional filter by district name
        grade_level: Optional filter by grade level
        school_name: Optional filter by school name
        format: Output format (xlsx or csv)
        
    Returns:
        Streaming response with the requested report
    """
    if not data_store.is_ready:
        raise HTTPException(status_code=503, 
                           detail="Data is still being loaded. Please try again shortly.")
    
    try:
        start_time = time.time()
        df = data_store.df
        
        # Convert grade_level to appropriate type if provided
        grade_level_param = None
        if grade_level is not None:
            grade_level_param = grade_level
        
        # Apply filters if provided
        if district_name or grade_level_param is not None or school_name:
            df = filter_data(df, district_name, grade_level_param, school_name)
            
        if len(df) == 0:
            raise HTTPException(status_code=404, detail="No data found for the selected filters")
        
        # Filter for students below 85% attendance
        below_85_df = df[df['Attendance_2024'] < 85].copy()
        
        if len(below_85_df) == 0:
            raise HTTPException(status_code=404, detail="No students with below 85% attendance found for the selected filters")
        
        # Create a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add filter info to filename
        filter_info = ""
        if district_name:
            filter_info += f"_dist_{district_name}"
        if grade_level:
            filter_info += f"_grade_{grade_level}"
        if school_name:
            filter_info += f"_school_{school_name}"
            
        # Generate the detailed report for below 85% attendance students
        detailed_df = generate_detailed_report(below_85_df)
        
        # Add extra risk analysis columns specific to attendance issues
        detailed_df['Days Below Expected'] = detailed_df['Attendance %'].apply(
            lambda x: round((85 - x) / 100 * 180) if x < 85 else 0  # Assuming 180 school days
        )
        
        # Sort by attendance percentage (ascending)
        detailed_df.sort_values(by='Attendance %', ascending=True, inplace=True)
        
        # Output based on format
        if format.lower() == "xlsx":
            output = io.BytesIO()
            detailed_df.to_excel(output, index=False)
            output.seek(0)
            
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"below_85_attendance_{timestamp}{filter_info}.xlsx"
            
        elif format.lower() == "csv":
            output = io.StringIO()
            detailed_df.to_csv(output, index=False)
            output_bytes = io.BytesIO(output.getvalue().encode())
            
            content_type = "text/csv"
            filename = f"below_85_attendance_{timestamp}{filter_info}.csv"
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid format: {format}. Use 'xlsx' or 'csv'")
        
        logger.info(f"Below 85% report generation completed in {time.time() - start_time:.4f} seconds")
        
        # Stream the file to the client
        headers = {
            "Content-Disposition": f"attachment; filename={filename}"
        }
        
        if format.lower() == "csv":
            return StreamingResponse(output_bytes, media_type=content_type, headers=headers)
        else:
            return StreamingResponse(output, media_type=content_type, headers=headers)
        
    except Exception as e:
        logger.error(f"Error generating below 85% report: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@app.get("/api/attendance/trends")
async def get_attendance_trends(
    district_name: Optional[str] = None,
    grade_level: Optional[str] = None,
    school_name: Optional[str] = None
):
    """Get attendance trends with filtering support"""
    if not data_store.is_ready:
        raise HTTPException(status_code=503, 
                          detail="Data is still being loaded. Please try again shortly.")
    
    try:
        df = data_store.df
        
        # Apply filters if provided
        if district_name or grade_level is not None or school_name:
            df = filter_data(df, district_name, grade_level, school_name)
            
        if len(df) == 0:
            raise HTTPException(status_code=404, detail="No data found for the selected filters")
        
        # Calculate attendance distribution by tiers
        tier_distribution = df['TIER'].value_counts().to_dict()
        
        # Calculate histogram data for attendance percentages
        hist_data = np.histogram(
            df['Attendance_2024'].dropna(),
            bins=[0, 50, 60, 70, 75, 80, 85, 90, 95, 100]
        )
        
        histogram = {
            'bins': [f"{hist_data[1][i]}-{hist_data[1][i+1]}" for i in range(len(hist_data[1])-1)],
            'counts': hist_data[0].tolist()
        }
        
        grade_distribution = {}
        if grade_level is None:
            grade_distribution = df.groupby('STUDENT_GRADE_LEVEL_2024')['Attendance_2024'].mean().to_dict()
            grade_distribution = {str(k): float(v) for k, v in grade_distribution.items()}

        district_comparison = {}
        if district_name is None:
            district_comparison = df.groupby('DISTRICT_NAME')['Attendance_2024'].mean().to_dict()
            # Convert to list of objects for easier frontend consumption
            district_comparison = [
                {"district": k, "attendance": float(v)} 
                for k, v in district_comparison.items()
            ]
        
        return {
            "tier_distribution": tier_distribution,
            "attendance_histogram": histogram,
            "grade_distribution": grade_distribution,
            "district_comparison": district_comparison
        }
        
    except Exception as e:
        logger.error(f"Error retrieving attendance trends: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving attendance trends: {str(e)}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    # Enable debug logging to help diagnose 500 errors
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("uvicorn")
    logger.setLevel(logging.DEBUG)
    
    # Run with debug mode enabled
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=8001, 
        reload=True,
        log_level="debug"
    )
