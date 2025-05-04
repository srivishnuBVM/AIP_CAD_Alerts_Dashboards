import React from 'react';

const DownloadButton = ({ filters }) => {
  const handleDownload = async () => {
    const baseUrl = "http://127.0.0.1:8001/api/download/summary";
    const params = new URLSearchParams();

    // Add filters if they are present
    if (filters.district_name) params.append("district_name", filters.district_name);
    if (filters.grade_level) params.append("grade_level", filters.grade_level);
    if (filters.school_name) params.append("school_name", filters.school_name);

    const fullUrl = `${baseUrl}?${params.toString()}`;

    try {
      const response = await fetch(fullUrl);
      if (!response.ok) {
        throw new Error("Failed to download the report");
      }

      const blob = await response.blob();
      const contentDisposition = response.headers.get("Content-Disposition");
      const filename = contentDisposition
        ? contentDisposition.split("filename=")[1]
        : "report_summary.xlsx";

      const link = document.createElement("a");
      link.href = window.URL.createObjectURL(blob);
      link.download = filename.replace(/"/g, ''); // clean filename
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error("Download error:", error);
      alert("Error downloading the report. Please try again.");
    }
  };

  return (
    <button onClick={handleDownload} className="px-4 py-2 bg-blue-600 text-white rounded">
      Download Summary Report
    </button>
  );
};

export default DownloadButton;
