document.addEventListener('DOMContentLoaded', () => {
    // Initialize the page by fetching configuration
    initializeConfig();

    // Setup button listeners
    document.getElementById('train-btn').addEventListener('click', handleTrain);
    document.getElementById('predict-btn').addEventListener('click', handlePredict);
    
    // Listen for summary changes for validation
    document.getElementById('business_summary').addEventListener('input', checkInputs);
});

let configData = null;

/**
 * Helper to turn raw column names into professional labels
 */
function getDisplayLabel(feature) {
    const mapping = {
        'estimated_revenue': 'Revenue (TTM)',
        'ebitda': 'EBITDA (TTM)',
        'total_cash': 'Total Cash',
        'total_debt': 'Total Debt',
        'forwardPE': 'Forward P/E',
        'ev_to_ebitda': 'EV / EBITDA',
        'enterprise_value': 'Enterprise Value',
        'employee_count': 'Employee Count',
        'sector': 'Sector',
        'business_summary': 'Business Summary'
    };
    return mapping[feature] || feature.replace(/_/g, ' ');
}

/**
 * Populate the UI
 * Fetch the features/targets and create the HTML elements.
 */
async function initializeConfig() {
    const response = await fetch('/api/config');
    configData = await response.json();

    const configContainer = document.getElementById('config-container');
    const targetSelect = document.getElementById('target-select');

    // Combine all features for the checkbox list
    const allFeatures = [
        ...configData.financial_features,
        ...configData.categorical_features,
        ...configData.nlp_features
    ];

    allFeatures.forEach(feature => {
        const id = `feat-${feature}`;
        configContainer.innerHTML += `
            <div class="form-check me-3 text-start">
                <input class="form-check-input" type="checkbox" value="${feature}" id="${id}" checked>
                <label class="form-check-label small" for="${id}">${getDisplayLabel(feature)}</label>
            </div>
        `;
    });

    // Populate Dropdown for Target
    configData.targets.forEach(target => {
        targetSelect.innerHTML += `<option value="${target}">${getDisplayLabel(target)}</option>`;
    });
}

/**
 * The "Train Once" Logic 
 */
async function handleTrain() {
    const trainBtn = document.getElementById('train-btn');
    const statusDiv = document.getElementById('train-status');
    const predictBtn = document.getElementById('predict-btn');
    
    // Get all checked boxes
    const selectedFeatures = Array.from(document.querySelectorAll('.form-check-input:checked'))
                                  .map(cb => cb.value);
    const target = document.getElementById('target-select').value;

    // VALIDATION: Prevent target from being an input feature
    if (selectedFeatures.includes(target)) {
        statusDiv.innerHTML = `<span class="text-danger fw-bold">Error:</span> Target variable ("${getDisplayLabel(target)}") cannot be selected as an input feature. Please uncheck it and try again.`;
        return;
    }

    trainBtn.disabled = true;
    predictBtn.disabled = true; // Disable until new inputs are filled
    statusDiv.innerHTML = `<span class="text-primary">Initializing Engine...</span>`;

    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                features: selectedFeatures,
                target: target
            })
        });

        const result = await response.json();

        if (result.status === 'success') {
            setupPredictionUI(selectedFeatures, target);
            statusDiv.innerHTML = `<span class="text-success">Engine Ready.</span>`;
            
            // Display MAPE below the accordion
            const metricsDisplay = document.getElementById('metrics-display');
            const mapeScore = document.getElementById('mape-score');
            
            // Format MAPE as a percentage (assuming it's a decimal like 0.15)
            const mapePercent = (result.metrics.mape * 100).toFixed(1);
            mapeScore.innerText = `MAPE: ${mapePercent}%`;
            metricsDisplay.classList.remove('d-none');

            // Close the accordion automatically
            const configAccordion = document.getElementById('collapseConfig');
            const bsCollapse = bootstrap.Collapse.getInstance(configAccordion) || new bootstrap.Collapse(configAccordion);
            bsCollapse.hide();
        } else {
            statusDiv.innerHTML = `<span class="text-danger">Error: ${result.message}</span>`;
        }
    } catch (err) {
        statusDiv.innerHTML = `<span class="text-danger">Failed to connect to server.</span>`;
    } finally {
        trainBtn.disabled = false;
    }
}

/**
 * Dynamic Input Generation & Validation
 */
function setupPredictionUI(features, target) {
    const inputContainer = document.getElementById('prediction-inputs');
    const targetDisplay = document.getElementById('target-display');
    
    inputContainer.innerHTML = ''; // Clear previous inputs
    targetDisplay.innerHTML = `<span class="output-pill">${getDisplayLabel(target)} <small>&times;</small></span>`;

    features.forEach(feat => {
        if (feat === 'business_summary') return;

        const isCategorical = configData.categorical_features.includes(feat);
        let inputHtml = '';

        if (feat === 'sector') {
            // Render a dropdown for sector
            inputHtml = `
                <select class="form-select predict-input" data-feature="sector" required>
                    <option value="" selected disabled>Choose Sector...</option>
                    ${configData.sector_options.map(s => `<option value="${s}">${s}</option>`).join('')}
                </select>
            `;
        } else {
            const inputType = isCategorical ? 'text' : 'number';
            const placeholder = isCategorical ? 'Enter value...' : '0.00';
            inputHtml = `<input type="${inputType}" class="form-control predict-input" data-feature="${feat}" placeholder="${placeholder}" required>`;
        }

        const col = document.createElement('div');
        col.className = 'col-md-6';
        col.innerHTML = `
            <div class="form-group-custom">
                <label class="form-label">${getDisplayLabel(feat)}</label>
                ${inputHtml}
            </div>
        `;
        
        col.querySelector('.predict-input').addEventListener('input', checkInputs);
        col.querySelector('.predict-input').addEventListener('change', checkInputs); // For select change
        inputContainer.appendChild(col);
    });
    
    checkInputs();
}

/**
 * Validation Check
 * Enables the Predict button ONLY if all feature fields are filled.
 */
function checkInputs() {
    const inputs = document.querySelectorAll('.predict-input');
    const summary = document.getElementById('business_summary');
    const predictBtn = document.getElementById('predict-btn');
    
    const allInputsFilled = Array.from(inputs).every(input => input.value.trim() !== "");
    const summaryFilled = summary.value.trim().length > 10;

    predictBtn.disabled = !(allInputsFilled && summaryFilled);
}

/**
 * Repeated Prediction
 */
async function handlePredict() {
    const resultDiv = document.getElementById('prediction-result');
    const summary = document.getElementById('business_summary');
    const inputs = document.querySelectorAll('.predict-input');
    
    resultDiv.classList.remove('d-none');
    resultDiv.innerText = "Calculating Valuation...";

    // 1. Gather Payload
    const payload = {
        business_summary: summary.value.trim()
    };

    inputs.forEach(input => {
        const feat = input.dataset.feature;
        const val = input.value.trim();
        
        // Convert numbers if not a categorical field
        const isFinancial = configData.financial_features.includes(feat);
        payload[feat] = isFinancial ? parseFloat(val) : val;
    });

    try {
        // 2. POST to API
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        // 3. Display Result
        if (result.status === 'success') {
            resultDiv.innerHTML = `<span class="fw-bold">${result.valuation}</span>`;
        } else {
            resultDiv.innerHTML = `<span class="text-danger small">Error: ${result.message}</span>`;
        }
    } catch (err) {
        resultDiv.innerHTML = `<span class="text-danger small">Connection error.</span>`;
    }
}
