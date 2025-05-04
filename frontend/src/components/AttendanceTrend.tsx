import React from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";

interface Point {
  year: string;
  value: number;
  isPredicted: boolean;
}

interface Props {
  data: Point[];
}


export const AttendanceTrend: React.FC<Props> = ({ data }) => (
  <Card className="mb-6">
    <CardHeader>
      <CardTitle>Attendance Trend</CardTitle>
      <CardDescription>Historical + 2025 Prediction</CardDescription>
    </CardHeader>
    <CardContent>
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis dataKey="year" />
            <YAxis domain={[75, 100]} tickFormatter={(v) => `${v}%`} />
            <Tooltip
              formatter={(v) => [`${v}%`, "Rate"]}
              labelFormatter={(l) => `Year: ${l}`}
            />
            <Legend />
            <ReferenceLine y={90} stroke="#FF8A65" strokeDasharray="3 3" />
            <Line
              type="monotone"
              dataKey="value"
              name="Attendance"
              stroke="#4CAF50"
              strokeWidth={2}
              label={{
                formatter: (value: number) => `${value}%`,
                position: "top",
                fontSize: 12,
                fill: "#4CAF50",
              }}
              
              dot={{
                r: 5,
                stroke: "var(--background)",
                strokeWidth: 2,
                fill: (d) => (d.isPredicted ? "#FFB547" : "#4CAF50"),
              }}
              activeDot={{
                r: 8,
                fill: (d) => (d.isPredicted ? "#FFB547" : "#4CAF50"),
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flex mt-2 text-xs text-muted-foreground justify-between">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-[#4CAF50]"></div>
          <span>Historical</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-[#FFB547]"></div>
          <span>Predicted</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 border-t border-[#FF8A65] border-dashed"></div>
          <span>90% Threshold</span>
        </div>
      </div>
    </CardContent>
  </Card>
);
