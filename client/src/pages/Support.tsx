import { useState } from "react";
import { useSupabase } from "@/database/SupabaseProvider";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { useNavigate } from "react-router-dom";

const Support: React.FC = () => {
  const { supabase } = useSupabase();
  const [issueType, setIssueType] = useState("");
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  // Validate that both fields are filled out

  const submitSupportRequest = async () => {
    if (!issueType.trim() || !summary.trim()) {
      toast.error("Both issue type and summary are required!");
      return;
    }

    setLoading(true);

    try {
      // Insert the support request into the "Support" table in the database

      const { error } = await supabase
        .from("Support")
        .insert([{ issue_type: issueType, summary }]);

      if (error) throw error;
      // Show a success message and reset the form fields

      toast.success("Your support request has been submitted!");
      setIssueType("");
      setSummary("");
      navigate("/");
    } catch (error: unknown) {
      // Handle any errors that occur during the submission

      if (error instanceof Error) {
        toast.error(error.message || "An unexpected error occurred");
        console.error("Support submission error:", error.message);
      } else {
        toast.error("An unexpected error occurred");
        console.error("Support submission error:", error);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-transparent text-gray-900 dark:text-white px-6 py-10">
      {/* Card component to display the support form */}

      <Card className="max-w-4xl w-full shadow-lg bg-white dark:bg-black border border-gray-300 dark:border-gray-700 rounded-2xl p-6">
        <CardHeader className="text-center">
          <CardTitle className="text-4xl font-bold tracking-tight text-gray-900 dark:text-white">
            Support
          </CardTitle>
        </CardHeader>
        <CardContent className="text-left text-gray-700 dark:text-gray-300 flex flex-wrap gap-4">
          <p className="w-full">
            Need help? Please select the type of issue you're facing and provide
            a brief summary. Our support team will get back to you as soon as
            possible.
          </p>
          {/* Dropdown to select the issue type */}

          <div className="w-full">
            <label htmlFor="issueType" className="block mb-2 font-medium">
              Issue Type
            </label>
            <select
              id="issueType"
              value={issueType}
              onChange={(e) => setIssueType(e.target.value)}
              className="border border-gray-300 dark:border-gray-600 p-2 rounded-md w-full bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white"
              disabled={loading}
            >
              <option value="" disabled>
                Select an issue type
              </option>
              <option value="Bug">Bug</option>
              <option value="Security">Security</option>
              <option value="Privacy Concerns">Privacy Concerns</option>
              <option value="Account & Login Issues">
                Account & Login Issues
              </option>
              <option value="Feature Request">Feature Request</option>
              <option value="Performance Issue">Performance Issue</option>
              <option value="UI/UX Feedback">UI/UX Feedback</option>
              <option value="Data Sync & Integration">
                Data Sync & Integration
              </option>
              <option value="Notification Issues">Notification Issues</option>
              <option value="Other">Other</option>
            </select>
          </div>
          {/* Textarea to enter a summary of the issue */}

          <div className="w-full">
            <label htmlFor="summary" className="block mb-2 font-medium">
              Summary
            </label>
            <textarea
              id="summary"
              value={summary}
              onChange={(e) => setSummary(e.target.value)}
              placeholder="Enter a brief summary of your issue..."
              className="min-h-[120px] resize-none border border-gray-300 dark:border-gray-600 p-2 rounded-md w-full bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white"
              disabled={loading}
            />
          </div>
          {/* Submit button */}

          <div className="w-full flex justify-end mt-4">
            <Button
              onClick={submitSupportRequest}
              disabled={loading}
              className="bg-primary hover:bg-primary/90 text-white px-8 py-4 text-lg transition-colors duration-200"
            >
              {loading ? "Submitting..." : "Submit Request"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Support;
