import React from "react";
import { RiskCategory } from "@/types";
import { ShieldCheck, AlertTriangle, ShieldAlert } from "lucide-react";

interface Props {
  risk: RiskCategory;
  probability?: number | null;
  compact?: boolean;
}


export const RiskIndicator: React.FC<Props> = ({ risk, compact = false, probability=null }) => {
  const icon = {
    Low: <ShieldCheck className="h-6 w-6 text-green-500" />,
    Medium: <AlertTriangle className="h-6 w-6 text-amber-500" />,
    High: <ShieldAlert className="h-6 w-6 text-red-500" />,
  }[risk.level];  

  if (compact) {
    return (
      <div className="flex items-center p-2 bg-muted/30 rounded-lg">
        {icon}
        <div className="ml-2">
          <h3 className="text-sm font-semibold" style={{ color: risk.color }}>
            {risk.level} Risk
          </h3>
          <p className="text-xs text-muted-foreground line-clamp-2">
            {risk.description}
          </p>
          {probability !== null && (
            <p className="text-xs mt-1 text-muted-foreground">
              Probability (2025): <strong>{(probability * 100).toFixed(2)}%</strong>
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center p-4 bg-muted/30 rounded-lg">
      {icon}
      <h3 className="mt-3 text-lg font-semibold" style={{ color: risk.color }}>
        {risk.level} Risk
      </h3>
      <p className="mt-2 text-center text-sm text-muted-foreground">
        {risk.description}
      </p>
      {probability !== null && (
        <p className="mt-2 text-sm text-muted-foreground">
          Probability (2025): <strong>{(probability * 100).toFixed(2)}%</strong>
        </p>
      )}
    </div>
  );
};
