//working latest tested
import React, { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ChevronDown, ChevronUp, Globe, AlertCircle, Download } from "lucide-react";

const AlertsDashboard = () => {
  const [district, setDistrict] = useState("");
  const [school, setSchool] = useState("");
  const [grade, setGrade] = useState("");

  const [districtOptions, setDistrictOptions] = useState([]);
  const [schoolOptions, setSchoolOptions] = useState([]);
  const [gradeOptions, setGradeOptions] = useState([]);

  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isGlobalView, setIsGlobalView] = useState(false);
  const [allSchoolOptions, setAllSchoolOptions] = useState([]);

  const [showFilters, setShowFilters] = useState(true);


  const fetchFilterOptions = async () => {
    try {
      const res = await fetch(`http://127.0.0.1:8001/api/filter-options`);
      if (!res.ok) throw new Error("Failed to fetch filter options");
      const data = await res.json();
  
      setDistrictOptions(data.districts || []);
      setAllSchoolOptions(data.schools || []);
      setGradeOptions(data.grades || []);
      setSchoolOptions([]); 
    } catch (err) {
      console.error("Error fetching filter options:", err);
      setError("Failed to load filter options");
    }
  };

  useEffect(() => {
    fetchFilterOptions();
  }, []);

  // useEffect(() => {
  //   if (district) {
  //     const trimmedDistrict = district.trim();
  //     const filteredSchools = allSchoolOptions.filter(
  //       (s) => s.district.trim() === trimmedDistrict
  //     );
  //     setSchoolOptions(filteredSchools);
  //     setSchool(""); // reset school & grade
  //     setGrade("");
  //   } else {
  //     setSchoolOptions([]);
  //     setSchool("");
  //     setGrade("");
  //   }
  // }, [district]);

  useEffect(() => {
    const trimmedDistrict = district.trim();
    if (trimmedDistrict) {
      const filteredSchools = allSchoolOptions.filter(
        (s) => s.district.trim() === trimmedDistrict
      );
      setSchoolOptions(filteredSchools);
    } else {
      setSchoolOptions(allSchoolOptions); // Show all schools if no district selected
    }
  
    // Don't reset selected school unless it becomes invalid
    if (
      school &&
      trimmedDistrict &&
      !allSchoolOptions.find(
        (s) => s.district.trim() === trimmedDistrict && s.value === school
      )
    ) {
      setSchool("");
    }
  
    // Always reset grade when district changes
    setGrade("");
  }, [district, allSchoolOptions]);
  


  const fetchAnalysis = async () => {
    setLoading(true);
    setError(null);
    setIsGlobalView(false);
  
    try {
      const params = new URLSearchParams();
      if (district) params.append("district_name", district.trimEnd() + " ");
      if (school) params.append("school_name", school);
      if (grade) params.append("grade_level", grade);
  
      const res = await fetch(`http://127.0.0.1:8001/api/analysis?${params.toString()}`, {
        method: "GET",
        headers: { Accept: "application/json" },
      });
  
      const contentType = res.headers.get("content-type");
      if (!res.ok) throw new Error("API returned an error");
      if (!contentType || !contentType.includes("application/json")) throw new Error("Response was not JSON");
  
      const data = await res.json();
      setAnalysisData(data);
    } catch (err) {
      console.error("Error fetching analysis:", err);
      window.alert("Selected filter data does not exist.");
    } finally {
      setLoading(false);
    }
  };
  
  

  const fetchGlobalAnalysis = async () => {
    setLoading(true);
    setError(null);
    setIsGlobalView(true);
    
    try {
      const res = await fetch(`http://127.0.0.1:8001/api/global-analysis`, {
        method: "GET",
        headers: { Accept: "application/json" },
      });

      const contentType = res.headers.get("content-type");
      if (!res.ok) throw new Error("API returned an error");
      if (!contentType || !contentType.includes("application/json")) throw new Error("Response was not JSON");

      const data = await res.json();
      setAnalysisData(data);
    } catch (err) {
      console.error("Error fetching global analysis:", err);
      setError(err.message || "An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };
  
  const resetFilters = () => {
    setDistrict("");
    setSchool("");
    setGrade("");
    setAnalysisData(null);
    setIsGlobalView(false);
    fetchFilterOptions();
    fetchGlobalAnalysis();
  };

  useEffect(() => {
    fetchFilterOptions();
    fetchGlobalAnalysis(); // Auto-load on mount
  }, []);

  const handleDownloadReport = async (reportType = "summary") => {
    try {
      const queryParams = new URLSearchParams();
      if (grade) queryParams.append("grade_level", grade);
      if (school) queryParams.append("school_name", school);
      if (district) queryParams.append("district_name", district.trimEnd() + " ");
  
      const res = await fetch(`http://127.0.0.1:8001/api/download/${reportType}?${queryParams}`, {
        method: "GET",
      });
  
      if (!res.ok) throw new Error(`Failed to download ${reportType} report`);
  
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
  
      const a = document.createElement("a");
      a.href = url;
      a.download = `${reportType}_report.xlsx`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Download error:", err);
      alert(`Failed to download ${reportType} report: ${err.message}`);
    }
  };
  
  const handleDownloadBelow85Report = () => {
    handleDownloadReport("below_85");
  };

  return (
    <div className="min-h-screen bg-gray-50/50">
      <div className="container mx-auto px-4 py-8 max-w-full">
        <div className="mb-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold">Alerts Dashboard</h1>
              <p className="text-muted-foreground">Monitor alerts and notifications</p>
            </div>
            <div className="flex items-center gap-2">
            {/* <button 
  onClick={() => handleDownloadReport("summary")}
  className="flex items-center gap-0.5 px-2 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-500"
>
  <Download size={12} />
  Summary
</button>

<button 
  onClick={handleDownloadBelow85Report}
  className="flex items-center gap-0.5 px-2 py-1 text-sm bg-orange-500 text-white rounded hover:bg-orange-600"
>
  <AlertCircle size={12} />
  Below 85%
</button> */}

              <button
                onClick={() => setShowFilters(!showFilters)}
                className="flex items-center gap-1 text-sm text-blue-600 hover:underline ml-4"
              >
                {showFilters ? "Hide Filters" : "Show Filters"}
                {showFilters ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              </button>
            </div>
          </div>
          <div className="w-full h-0.5 bg-gray-300 mt-1"></div>
        </div>

        <div className="flex w-full min-h-[calc(100vh-6rem)]">
          {showFilters && (
            <div className="w-64 p-4 bg-white shadow rounded-md mr-4 h-[calc(100vh-6rem)] overflow-y-auto">
              <div className="mb-4">
                <label className="block font-semibold mb-1">District</label>
                <select
                  value={district}
                  onChange={(e) => setDistrict(e.target.value)}
                  className="w-full p-2 border rounded"
                >
                  <option value="">Select District</option>
                  {districtOptions.map((d) => (
                    <option key={d.value} value={d.value}>
                      {d.label}
                    </option>
                  ))}
                </select>
              </div>
              
              <div className="mb-4">
                <label className="block font-semibold mb-1">School</label>
                <select
                  value={school}
                  onChange={(e) => setSchool(e.target.value)}
                  className="w-full p-2 border rounded"
                  disabled={!schoolOptions.length}
                >
                  <option value="">Select School</option>
                  {schoolOptions.map((s) => (
                    <option key={s.value} value={s.value}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="mb-4">
                <label className="block font-semibold mb-1">Grade</label>
                <select
                  value={grade}
                  onChange={(e) => setGrade(e.target.value)}
                  className="w-full p-2 border rounded"
                  disabled={!gradeOptions.length}
                >
                  <option value="">Select Grade</option>
                  {gradeOptions.map((g) => (
                    <option key={g.value} value={g.value}>
                      {g.label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={fetchAnalysis}
                  className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 w-full"
                >
                  Search
                </button>
                <button
                  onClick={resetFilters}
                  className="bg-gray-300 text-gray-800 px-4 py-2 rounded hover:bg-gray-400 w-full"
                >
                  Reset
                </button>
              </div>
            </div>
          )}

          {/* Main Dashboard */}
          <div className="flex-1 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4 p-4">
            {isGlobalView && analysisData && (
              <div className="col-span-full mb-4">
                <div className="bg-blue-50 border-l-4 border-blue-500 p-4">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <Globe className="h-5 w-5 text-blue-500" />
                    </div>
                    <div className="ml-3">
                      <p className="text-sm text-blue-700">
                        Viewing Global Analysis - Showing data for all districts, schools, and grades
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
            {loading && (
              <div className="flex justify-center items-center col-span-full">
                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500" />
              </div>
            )}

            {error && <p className="text-red-500 col-span-full">{error}</p>}
          
            {analysisData && (
              <>
                <Card className="bg-orange-50 border border-orange-200">
  <CardHeader className="pb-2">
    <CardTitle className="flex items-center gap-2">
      Total Students
    </CardTitle>
  </CardHeader>
  <CardContent className="flex justify-between items-center">
    <span className="text-xl font-semibold">
      {analysisData.summary_statistics.total_students}
    </span>
    <button 
      onClick={() => handleDownloadReport("summary")}
      className="text-xs bg-orange-500 text-white p-1 rounded flex items-center gap-1"
      title="Download Summary Report"
    >
      <Download size={12} />
      Export
    </button>
  </CardContent>
</Card>
                <Card>
                  <CardHeader><CardTitle>At Risk Students</CardTitle></CardHeader>
                  <CardContent>{analysisData.summary_statistics.at_risk_students}</CardContent>
                </Card>
                <Card className="bg-orange-50 border-orange-200">
                  <CardHeader className="pb-2">
                    <CardTitle className="flex items-center gap-2">
                      <AlertCircle size={16} className="text-orange-500" />
                      Below 85% Attendance
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="flex justify-between items-center">
                    <span className="text-xl font-semibold">{analysisData.summary_statistics.below_85_students}</span>
                    <button
                      onClick={handleDownloadBelow85Report}
                      className="text-xs bg-orange-500 text-white p-1 rounded flex items-center gap-1"
                      title="Download Below 85% Report"
                    >
                      <Download size={12} />
                      Export
                    </button>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader><CardTitle>Critical Risk Students</CardTitle></CardHeader>
                  <CardContent>{analysisData.summary_statistics.critical_risk_students}</CardContent>
                </Card>
                <Card>
                  <CardHeader><CardTitle>Tier 4 Students</CardTitle></CardHeader>
                  <CardContent>{analysisData.summary_statistics.tier4_students}</CardContent>
                </Card>

                {/* Key Insights */}
                {analysisData.key_insights?.length > 0 && (
                  <div className="col-span-full mt-0">
                     <Card className="bg-white shadow">
      <CardHeader><CardTitle>Key Insights</CardTitle></CardHeader>
      <CardContent>
        <ul className="list-disc list-inside space-y-1">
          {analysisData.key_insights.map((item, index) => (
            <li key={index} dangerouslySetInnerHTML={{
              __html: item.insight.replace(/(\d+(\.\d+)?%?)/g, "<strong>$1</strong>")
            }} />
          ))}
        </ul>
      </CardContent>
    </Card>
                  </div>
                )}
                {/* Recommendations */}
                {analysisData.recommendations?.length > 0 && (
                  <div className="col-span-full mt-0">
                    {/* <Card className="bg-white shadow">
                      <CardHeader><CardTitle>Recommendations</CardTitle></CardHeader>
                      <CardContent>
                        <ul className="list-disc list-inside space-y-1">
                          {analysisData.recommendations.map((item, index) => (
                            <li key={index}>{item.recommendation}</li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card> */}
                    <Card className="bg-white shadow">
      <CardHeader><CardTitle>Recommendations</CardTitle></CardHeader>
      <CardContent>
        <ul className="list-disc list-inside space-y-1">
          {analysisData.recommendations.map((item, index) => (
            <li key={index} dangerouslySetInnerHTML={{
              __html: item.recommendation.replace(/(\d+(\.\d+)?%?)/g, "<strong>$1</strong>")
            }} />
          ))}
        </ul>
      </CardContent>
    </Card>
                  </div>
                )}
              </>
            )}
          </div>

        </div>
      </div>
      
    </div>
  );
};

export default AlertsDashboard;


