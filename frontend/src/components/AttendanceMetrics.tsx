// import React, { useEffect, useState } from "react";
// import { AttendanceData } from "@/types";
// import { CalendarCheck2, CalendarX2, Clock } from "lucide-react";
// import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// interface Props {
//   studentId: string;
// }

// export const AttendanceMetrics: React.FC<Props> = ({ studentId }) => {
//   const [metrics, setMetrics] = useState<AttendanceData[]>([]);

//   useEffect(() => {
//     fetch(`http://localhost:8000/students/${studentId}/metrics`)
//       .then((res) => res.json())
//       .then(setMetrics)
//       .catch(console.error);
//   }, [studentId]);

//   if (metrics.length < 2) return null;

//   const previous = metrics[metrics.length - 2];
//   const current = metrics[metrics.length-1];
//   const change = (current.attendanceRate ?? 0) - (previous.attendanceRate ?? 0);
//   const positive = change >= 0;

//   return (
//     <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
//       <Card>
//         <CardHeader>
//           <CardTitle className="text-sm font-medium text-muted-foreground">
//             Attendance Rate
//           </CardTitle>
//         </CardHeader>
//         <CardContent>
//           <div className="flex items-baseline justify-between">
//             <div className="flex items-center">
//               <CalendarCheck2 className="h-5 w-5 text-green-500 mr-2" />
//               <div className="text-2xl font-bold">
//                 {current.attendanceRate ?? "--"}%
//               </div>
//             </div>
//             <div
//               className={`text-sm ${
//                 positive ? "text-green-500" : "text-red-500"
//               }`}
//             >
//               {positive ? "↑" : "↓"} {Math.abs(change)}%
//             </div>
//           </div>
//           <p className="text-xs text-muted-foreground mt-1">
//             Compared to {previous.year}
//           </p>
//         </CardContent>
//       </Card>

//       <Card>
//         <CardHeader>
//           <CardTitle className="text-sm font-medium text-muted-foreground">
//             Absences
//           </CardTitle>
//         </CardHeader>
//         <CardContent>
//           <div className="flex items-baseline justify-between">
//             <div className="flex items-center">
//               <CalendarX2 className="h-5 w-5 text-red-400 mr-2" />
//               <div className="text-2xl font-bold">
//                 {current.absences ?? "--"}
//               </div>
//             </div>
//             <div className="text-xs">
//               <span className="font-medium">{current.excused ?? "--"}</span>{" "}
//               excused
//             </div>
//           </div>
//           <p className="text-xs text-muted-foreground mt-1">
//             {current.absences != null
//               ? ((current.absences / current.total) * 100).toFixed(1) +
//                 "% of school days"
//               : "--"}
//           </p>
//         </CardContent>
//       </Card>

//       <Card>
//         <CardHeader>
//           <CardTitle className="text-sm font-medium text-muted-foreground">
//             Late Arrivals
//           </CardTitle>
//         </CardHeader>
//         <CardContent>
//           <div className="flex items-baseline justify-between">
//             <div className="flex items-center">
//               <Clock className="h-5 w-5 text-amber-400 mr-2" />
//               <div className="text-2xl font-bold">{current.lates ?? "--"}</div>
//             </div>
//           </div>
//           <p className="text-xs text-muted-foreground mt-1">
//             {current.lates != null
//               ? ((current.lates / current.total) * 100).toFixed(1) +
//                 "% of school days"
//               : "--"}
//           </p>
//         </CardContent>
//       </Card>
//     </div>
//   );
// };
