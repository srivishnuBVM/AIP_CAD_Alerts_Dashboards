import React, { useEffect, useState } from "react";
import { Student, AttendanceData, RiskCategory } from "@/types";
import { StudentSelector } from "@/components/StudentSelector";
import { AttendanceTrend } from "@/components/AttendanceTrend";
import { RiskIndicator } from "@/components/RiskIndicator";
import { AttendanceHistory } from "@/components/AttendanceHistory";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { CalendarCheck2, Filter } from "lucide-react";
import { Download } from "lucide-react";
import { Separator } from "@/components/ui/separator";

const API = "http://localhost:8000";

const Index: React.FC = () => {
  const [students, setStudents] = useState<Student[]>([]);
  const [selected, setSelected] = useState<Student | null>(null);
  const [history, setHistory] = useState<AttendanceData[]>([]);
  const [pred, setPred] = useState<AttendanceData | null>(null);
  const [risk, setRisk] = useState<RiskCategory | null>(null);
  const [trend, setTrend] = useState<any[]>([]);
  const [probability, setProbability] = useState<number | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  useEffect(() => {
    fetch(`${API}/students`)
      .then((r) => r.json())
      .then((d) => {
        setStudents(d);
        setSelected(d[0]);
      });
  }, []);

  useEffect(() => {
    if (!selected) return;
    const id = selected.id;
    fetch(`${API}/students/${id}/details`)
      .then((r) => r.json())
      .then((d) => {
        setRisk(d.risk);
        setPred(d.predictedAttendance);
        setProbability(d.probability2025)
        console.log("Probability 2025:", d.probability2025);
      });
    fetch(`${API}/students/${id}/metrics`)
      .then((r) => r.json())
      .then(setHistory);
    fetch(`${API}/students/${id}/trend`)
      .then((r) => r.json())
      .then(setTrend);
  }, [selected]);

  const prev = history[history.length - 2] ?? null;
  const curr = history[history.length - 1] ?? null;

  return (
    <div className="min-h-screen bg-gray-50/50">
      {/* Top Header Bar */}
      <header className="w-full bg-white border-b border-gray-200 shadow-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-gray-800">
                AI-Driven Attendance Analytics
              </h1>
              <p className="text-gray-500 mt-1">
                Track and analyze student attendance patterns
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="w-full flex">
        {/* Left sidebar with filters - collapsible */}
        <aside className={`bg-white border-r border-gray-200 shadow-sm transition-all duration-300 ${sidebarCollapsed ? "w-14" : "w-80"} h-[calc(100vh-73px)] sticky top-[73px]`}>
          {!sidebarCollapsed ? (
            <div className="p-5">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-lg font-bold text-gray-700">Filters</h2>
                <button 
                  onClick={() => setSidebarCollapsed(true)}
                  className="text-gray-500 hover:text-gray-700 focus:outline-none"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
              {selected && (
                <StudentSelector
                  students={students}
                  selectedStudent={selected}
                  onSelect={setSelected}
                />
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center pt-5">
              <button 
                onClick={() => setSidebarCollapsed(false)}
                className="flex flex-col items-center justify-center p-2 text-gray-500 hover:text-gray-700 focus:outline-none"
              >
                <Filter className="h-6 w-6 mb-1" />
                <span className="text-xs rotate-90 mt-2">Filters</span>
              </button>
            </div>
          )}
        </aside>
        
        {/* Right side with main content */}
        <main className="flex-1 p-6 bg-gray-50">
          {/* Student Info Header */}
          {selected && (
            <div className="bg-white rounded-lg shadow-sm p-4 mb-6">
              <div className="flex justify-between items-center">
                <div>
                  <h2 className="text-xl font-bold">Student: {selected.id}</h2>
                  <div className="flex gap-4 text-sm text-gray-500 mt-1">
                    <span>Grade: {selected.grade}</span>
                    <span>School: {selected.schoolName}</span>
                    <span>District: {selected.districtName}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Metrics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
            {/* Attendance Rate */}
            <Card className="shadow-sm hover:shadow-md transition-shadow">
              <CardHeader className="pb-2">
                <CardTitle className="text-base text-gray-600">
                  Attendance Rate (2024)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-baseline justify-between">
                  <div className="flex items-center">
                    <CalendarCheck2 className="h-5 w-5 text-green-500 mr-2" />
                    <div className="text-3xl font-bold">
                      {curr ? `${curr.attendanceRate}%` : "--"}
                    </div>
                  </div>
                  <div className="text-right">
                    {curr && prev ? (
                      <>
                        <div
                          className={
                            curr.attendanceRate - prev.attendanceRate >= 0
                              ? "text-green-600 font-semibold"
                              : "text-red-600 font-semibold"
                          }
                        >
                          {curr.attendanceRate - prev.attendanceRate >= 0 ? "▲" : "▼"}
                          {Math.abs(curr.attendanceRate - prev.attendanceRate)}%
                        </div>
                        <p className="text-xs text-gray-500">
                          vs {prev.year}
                        </p>
                      </>
                    ) : (
                      <div className="text-sm">N/A</div>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
            
            {/* Predicted Attendance */}
            <Card className="shadow-sm hover:shadow-md transition-shadow">
              <CardHeader className="pb-2">
                <CardTitle className="text-base text-gray-600">
                  AI Predicted Attendance (2025)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-baseline justify-between">
                  <div className="flex items-center">
                    <CalendarCheck2 className="h-5 w-5 text-amber-500 mr-2" />
                    <div className="text-3xl font-bold">
                      {pred ? `${pred.attendanceRate}%` : "--"}
                    </div>
                  </div>
                  <div className="text-right">
                    {pred && curr ? (
                      <>
                        <div
                          className={
                            pred.attendanceRate - curr.attendanceRate >= 0
                              ? "text-green-600 font-semibold"
                              : "text-red-600 font-semibold"
                          }
                        >
                          {pred.attendanceRate - curr.attendanceRate >= 0 ? "▲" : "▼"}
                          {Math.abs(pred.attendanceRate - curr.attendanceRate)}%
                        </div>
                        <p className="text-xs text-gray-500">
                          vs {curr.year}
                        </p>
                      </>
                    ) : (
                      <div className="text-sm">N/A</div>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Trend Chart */}
          <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
            <h3 className="text-lg font-bold mb-4">Attendance Trend</h3>
            <AttendanceTrend data={trend} />
          </div>

          {/* Attendance History */}
          {curr && pred && (
            <div className="bg-white rounded-lg shadow-sm p-6">
              <h3 className="text-lg font-bold mb-4">Attendance History</h3>
              <AttendanceHistory history={history} predicted={pred} />
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default Index;