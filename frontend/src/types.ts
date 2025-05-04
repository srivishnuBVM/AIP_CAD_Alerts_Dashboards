export interface AttendanceData {
  year: string;
  attendanceRate: number | null;
  absences: number | null;
  lates: number | null;
  excused: number | null;
  total: number;
  isPredicted?: boolean;
}

export interface Student {
  id: string;
  grade: string;
}

export interface RiskCategory {
  level: "Low" | "Medium" | "High";
  color: string;
  description: string;
}
