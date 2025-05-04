import React from "react";
import { AttendanceData } from "@/types";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import {
  Table,
  TableHeader,
  TableRow,
  TableHead,
  TableBody,
  TableCell,
} from "@/components/ui/table";
import { Progress } from "@/components/ui/progress";

interface Props {
  history: AttendanceData[];
  predicted: AttendanceData;
}

export const AttendanceHistory: React.FC<Props> = ({ history, predicted }) => {
  const rows = [...history, { ...predicted, isPredicted: true }];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Year-by-Year Breakdown</CardTitle>
        <CardDescription>Detailed attendance metrics</CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Year</TableHead>
              <TableHead>Rate</TableHead>
              <TableHead className="text-center">Absences</TableHead>
              <TableHead className="text-center">Excused</TableHead>
              <TableHead className="text-center">Total Enrolled</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((r) => (
              <TableRow
                key={r.year}
                className={r.isPredicted ? "bg-amber-300/30" : ""}
              >
                <TableCell>
                  {r.year}{" "}
                  {r.isPredicted && (
                    <span className="text-xs text-amber-500">(Predicted)</span>
                  )}
                </TableCell>
                <TableCell>
                  <div className="flex flex-col gap-1">
                    <span>
                      {r.attendanceRate != null
                        ? `${r.attendanceRate}%`
                        : "N/A"}
                    </span>
                    <Progress value={r.attendanceRate ?? 0} className="h-2" />
                  </div>
                </TableCell>
                <TableCell className="text-center">
                  {r.absences ?? "N/A"}
                </TableCell>
                <TableCell className="text-center">
                  {r.excused ?? "N/A"}
                </TableCell>
                <TableCell className="text-center">{r.total}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
};
