document.getElementById('churnForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // UI Elements
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.getElementById('loadingSpinner');
    const riskRing = document.getElementById('riskRing');
    const riskPercentage = document.getElementById('riskPercentage');
    const riskBadge = document.getElementById('riskBadge');
    const insightText = document.getElementById('insightText');

    // Show loading state
    submitBtn.disabled = true;
    btnText.textContent = "Analyzing...";
    spinner.classList.remove('hidden');

    // Gather form data
    const formData = new FormData(e.target);
    const dataObj = Object.fromEntries(formData.entries());

    // Convert numerics
    dataObj.SeniorCitizen = parseInt(dataObj.SeniorCitizen);
    dataObj.tenure = parseInt(dataObj.tenure);
    dataObj.MonthlyCharges = parseFloat(dataObj.MonthlyCharges);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(dataObj)
        });

        if (!response.ok) {
            throw new Error(`Server error: ${await response.text()}`);
        }

        const result = await response.json();
        
        // Calculate UI values
        const probPct = Math.round(result.churn_probability * 100);
        
        // Update Gauge Animation
        // stroke-dasharray = "value, 100" where value is out of 100
        setTimeout(() => {
            riskRing.setAttribute('stroke-dasharray', `${probPct}, 100`);
            
            // Count up animation for text
            let current = 0;
            const step = Math.ceil(probPct / 20) || 1;
            const timer = setInterval(() => {
                current += step;
                if (current >= probPct) {
                    current = probPct;
                    clearInterval(timer);
                }
                riskPercentage.textContent = `${current}%`;
            }, 30);
            
            // Color updates based on risk
            let color, badgeText, insight;
            
            if (probPct < 35) {
                color = "var(--risk-low)";
                badgeText = "Low Risk";
                insight = "This customer is highly likely to stay. Ensure standard service levels are maintained to keep them satisfied.";
            } else if (probPct < 65) {
                color = "var(--risk-medium)";
                badgeText = "Medium Risk";
                insight = "This customer is showing signs of potential churn. Consider offering a small discount or a courtesy check-in call.";
            } else {
                color = "var(--risk-high)";
                badgeText = "High Risk";
                insight = "CRITICAL: This customer is highly likely to leave. Deploy retention strategies immediately, such as a promotional contract renewal.";
            }

            riskRing.style.stroke = color;
            riskBadge.textContent = badgeText;
            riskBadge.style.background = color;
            riskBadge.style.color = "#000";
            insightText.textContent = insight;

        }, 100);

    } catch (error) {
        console.error(error);
        alert("An error occurred while making the prediction. Check the console for details.");
    } finally {
        // Reset button
        submitBtn.disabled = false;
        btnText.textContent = "Analyze Customer Risk";
        spinner.classList.add('hidden');
    }
});
