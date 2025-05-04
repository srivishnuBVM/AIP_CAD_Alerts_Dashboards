import React, { useEffect, useState } from "react";
import { Student } from "@/types";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Check, ChevronDown } from "lucide-react";

interface Props {
  students: Student[];
  selectedStudent: Student;
  onSelect: (s: Student) => void;
}

export const StudentSelector: React.FC<Props> = ({
  students,
  selectedStudent,
  onSelect,
}) => {
  const gradeOrder = [
    "Pre-Kindergarten",
    "Kindergarten",
    "1st Grade",
    "2nd Grade",
    "3rd Grade",
    "4th Grade",
    "5th Grade",
    "6th Grade",
    "7th Grade",
    "8th Grade",
    "9th Grade",
    "10th Grade",
    "11th Grade",
    "12th Grade",
  ];

  const [selectedDistrict, setSelectedDistrict] = useState<string | null>(null);
  const [selectedSchool, setSelectedSchool] = useState<string | null>(null);
  const [selectedGrade, setSelectedGrade] = useState<string | null>(null);

  // Filter students by District Name
  const filteredByDistrict = selectedDistrict
    ? students.filter((s) => s.districtName === selectedDistrict)
    : students;

  // Get unique District Names for the dropdown
  const districtNames = Array.from(new Set(students.map((s) => s.districtName))).sort();

  // Filter students by Location ID (School)
  const filteredByLocation = selectedSchool
    ? filteredByDistrict.filter((s) => s.schoolName === selectedSchool)
    : filteredByDistrict;

  // Get unique Location IDs for the dropdown
  const locationIds = Array.from(new Set(filteredByDistrict.map((s) => s.schoolName))).sort();

  // Get unique grades available for the selected Location ID
  const availableGrades = Array.from(
    new Set(filteredByLocation.map((s) => s.grade))
  ).sort((a, b) => {
    const gradeAIndex = gradeOrder.indexOf(a);
    const gradeBIndex = gradeOrder.indexOf(b);
    return gradeAIndex - gradeBIndex;
  });

  // Filter students further by Grade
  const filteredStudents = selectedGrade
    ? filteredByLocation
        .filter((s) => s.grade === selectedGrade)
        .sort((a, b) => {
          // Sort by Location_ID first
          if (a.schoolName !== b.schoolName) {
            return a.schoolName.localeCompare(b.schoolName);
          }
          // Sort by Grade
          const gradeAIndex = gradeOrder.indexOf(a.grade);
          const gradeBIndex = gradeOrder.indexOf(b.grade);
          if (gradeAIndex !== gradeBIndex) {
            return gradeAIndex - gradeBIndex;
          }
          // Sort by Student ID last
          return a.id.localeCompare(b.id);
        })
    : filteredByLocation;

  useEffect(() => {
    console.log("Filtered students:", filteredStudents);
  }, [filteredStudents]);

  const DropdownLabel = ({ text, count }: { text: string; count?: number }) => (
    <div className="flex justify-between items-center w-full">
      <span className="font-medium">{text}</span>
      {count !== undefined && (
        <span className="bg-gray-100 text-gray-600 text-xs px-2 py-1 rounded-full">
          {count}
        </span>
      )}
    </div>
  );

  return (
    <div className="flex flex-col space-y-5">
      {/* District Dropdown */}
      <div className="space-y-2">
        <label className="block text-base font-semibold text-gray-700">
          District
        </label>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button 
              variant="outline" 
              className="w-full justify-between text-base font-normal h-10 bg-white border-gray-300"
            >
              <span className="truncate">
                {selectedDistrict || "Select district"}
              </span>
              <ChevronDown className="h-4 w-4 ml-2 opacity-50" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent className="max-h-60 overflow-y-auto w-full min-w-[240px]">
            {districtNames.map((district) => (
              <DropdownMenuItem
                key={district}
                onClick={() => {
                  setSelectedDistrict(district);
                  setSelectedSchool(null);
                  setSelectedGrade(null);
                }}
                className="text-base py-2 cursor-pointer"
              >
                <div className="flex items-center w-full">
                  <span className="mr-2 w-4">
                    {selectedDistrict === district && (
                      <Check className="h-4 w-4" />
                    )}
                  </span>
                  {district}
                </div>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* School Dropdown */}
      <div className="space-y-2">
        <label className="block text-base font-semibold text-gray-700">
          School
        </label>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button 
              variant="outline" 
              className="w-full justify-between text-base font-normal h-10 bg-white border-gray-300"
              disabled={!selectedDistrict}
            >
              <span className="truncate">
                {selectedSchool || "Select school"}
              </span>
              <ChevronDown className="h-4 w-4 ml-2 opacity-50" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent className="max-h-60 overflow-y-auto w-full min-w-[240px]">
            {locationIds.length === 0 ? (
              <DropdownMenuItem disabled className="text-base py-2">
                No schools available
              </DropdownMenuItem>
            ) : (
              locationIds.map((location) => (
                <DropdownMenuItem
                  key={location}
                  onClick={() => {
                    setSelectedSchool(location);
                    setSelectedGrade(null);
                  }}
                  className="text-base py-2 cursor-pointer"
                >
                  <div className="flex items-center w-full">
                    <span className="mr-2 w-4">
                      {selectedSchool === location && (
                        <Check className="h-4 w-4" />
                      )}
                    </span>
                    {location}
                  </div>
                </DropdownMenuItem>
              ))
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Grade Dropdown */}
      <div className="space-y-2">
        <label className="block text-base font-semibold text-gray-700">
          Grade
        </label>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button 
              variant="outline" 
              className="w-full justify-between text-base font-normal h-10 bg-white border-gray-300"
              disabled={!selectedSchool}
            >
              <span className="truncate">
                {selectedGrade || "Select grade"}
              </span>
              <ChevronDown className="h-4 w-4 ml-2 opacity-50" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent className="max-h-60 overflow-y-auto w-full min-w-[240px]">
            {availableGrades.length === 0 ? (
              <DropdownMenuItem disabled className="text-base py-2">
                No grades available
              </DropdownMenuItem>
            ) : (
              availableGrades.map((grade) => (
                <DropdownMenuItem
                  key={grade}
                  onClick={() => {
                    setSelectedGrade(grade);
                  }}
                  className="text-base py-2 cursor-pointer"
                >
                  <div className="flex items-center w-full">
                    <span className="mr-2 w-4">
                      {selectedGrade === grade && <Check className="h-4 w-4" />}
                    </span>
                    {grade}
                  </div>
                </DropdownMenuItem>
              ))
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Student Dropdown */}
      <div className="space-y-2">
        <label className="block text-base font-semibold text-gray-700">
          Student
        </label>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="outline"
              className="w-full justify-between text-base font-normal h-10 bg-white border-gray-300"
              disabled={!selectedGrade || filteredStudents.length === 0}
            >
              <span className="truncate">
                {selectedStudent?.id || "Select student"}
              </span>
              <ChevronDown className="h-4 w-4 ml-2 opacity-50" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent className="max-h-60 overflow-y-auto w-full min-w-[240px]">
            {filteredStudents.length === 0 ? (
              <DropdownMenuItem disabled className="text-base py-2">
                No students available
              </DropdownMenuItem>
            ) : (
              filteredStudents.map((s) => (
                <DropdownMenuItem
                  key={s.id}
                  onClick={() => {
                    onSelect(s);
                  }}
                  className="text-base py-2 cursor-pointer"
                >
                  <div className="flex items-center justify-between w-full">
                    <div className="flex items-center">
                      <span className="mr-2 w-4">
                        {selectedStudent?.id === s.id && (
                          <Check className="h-4 w-4" />
                        )}
                      </span>
                      {s.id}
                    </div>
                    <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                      {s.grade}
                    </span>
                  </div>
                </DropdownMenuItem>
              ))
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
};