document.addEventListener('DOMContentLoaded', () => {
    // Initialize the page by fetching configuration
    initializeConfig();

    // Setup button listeners
    document.getElementById('train-btn').addEventListener('click', handleTrain);
    document.getElementById('predict-btn').addEventListener('click', handlePredict);
});

/**
 * Populate the UI
 * Fetch the features/targets and create the HTML elements.
 */
async function initializeConfig() {
    const response = await fetch('/api/config');
    const data = await response.json();

    const configContainer = document.getElementById('config-container');
    const targetSelect = document.getElementById('target-select');

    // Populate Checkboxes for Features
    data.features.forEach(feature => {
        const id = `feat-${feature}`;
        configContainer.innerHTML += `
            <div class="form-check me-3 text-start">
                <input class="form-check-input" type="checkbox" value="${feature}" id="${id}" checked>
                <label class="form-check-label small" for="${id}">${feature}</label>
            </div>
        `;
    });

    // Populate Dropdown for Target
    data.targets.forEach(target => {
        targetSelect.innerHTML += `<option value="${target}">${target}</option>`;
    });
}

/**
 * Collect selected features, tell the server to train, 
 * and then build the prediction inputs.
 */
async function handleTrain() {
    const trainBtn = document.getElementById('train-btn');
    const statusDiv = document.getElementById('train-status');
    
    // Get all checked boxes
    const selectedFeatures = Array.from(document.querySelectorAll('.form-check-input:checked'))
                                  .map(cb => cb.value);
    const target = document.getElementById('target-select').value;

    trainBtn.disabled = true;
    statusDiv.innerText = "Initializing Engine... please wait.";

    // TODO: Actually POST these to /api/train in the next step
    // For now, we simulate success to show you the UI generation:
    setTimeout(() => {
        setupPredictionUI(selectedFeatures, target);
        statusDiv.innerText = "Engine Ready.";
        document.getElementById('predict-btn').disabled = false;
        
        // Close the accordion automatically after training
        const configAccordion = document.getElementById('collapseConfig');
        const bsCollapse = bootstrap.Collapse.getInstance(configAccordion) || new bootstrap.Collapse(configAccordion);
        bsCollapse.hide();
    }, 1000);
}

/**
 * EXERCISE 3: Dynamic Input Generation
 * This builds the "Revenue", "Employees" etc. fields 
 * ONLY for the features the user selected.
 */
function setupPredictionUI(features, target) {
    const inputContainer = document.getElementById('prediction-inputs');
    const targetDisplay = document.getElementById('target-display');
    
    inputContainer.innerHTML = ''; // Clear previous
    targetDisplay.innerHTML = `<span class="output-pill">${target.replace(/_/g, ' ')} <small>&times;</small></span>`;

    features.forEach(feat => {
        // Skip 'business_summary' because it has its own big textarea already
        if (feat === 'business_summary') return;

        inputContainer.innerHTML += `
            <div class="col-md-6">
                <div class="form-group-custom">
                    <label class="form-label">${feat.replace(/_/g, ' ')}</label>
                    <input type="number" class="form-control predict-input" data-feature="${feat}" placeholder="0.00">
                </div>
            </div>
        `;
    });
}

/**
 * EXERCISE 4: Repeated Prediction
 */
async function handlePredict() {
    const resultDiv = document.getElementById('prediction-result');
    resultDiv.classList.remove('d-none');
    resultDiv.innerText = "Calculating Valuation...";

    // 1. Gather values from all '.predict-input' elements
    const inputs = {};
    document.querySelectorAll('.predict-input').forEach(el => {
        inputs[el.dataset.feature] = parseFloat(el.value) || 0;
    });
    
    // 2. Add the summary
    inputs['business_summary'] = document.getElementById('business_summary').value;

    console.log("Prediction Inputs:", inputs);

    // TODO: POST to /api/predict
    // Mock result for now
    setTimeout(() => {
        resultDiv.innerText = "$1,250,000,000";
    }, 800);
}
