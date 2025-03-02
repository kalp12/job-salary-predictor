async function predictSalary() {
    const jobTitle = document.getElementById("jobTitle").value.trim();  // Corrected ID
    const experience = document.getElementById("experience").value.trim();

    if (!jobTitle || !experience) {
        alert("Please enter both job title and experience.");
        return;
    }

    const url = `http://127.0.0.1:8000/predict/?job_title=${encodeURIComponent(jobTitle)}&experience=${encodeURIComponent(experience)}`;

    try {
        const response = await fetch(url, {
            method: "POST",  // Using GET to match FastAPI expectations
            headers: {
                "Content-Type": "application/json"
            }
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        document.getElementById("salary_output").innerText = `Predicted Salary: ${data.predicted_salary}`;
    } catch (error) {
        console.error("Error fetching salary prediction:", error);
        alert("Failed to fetch salary data. Check console for details.");
    }
}
