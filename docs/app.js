//
// app.js
//

// CONSTANTS for paths (adjust these as needed)
const CSV_BASE_PATH = "https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/refs/heads/master/data/all_csv/";
const TABLE_TO_PAGE_JSON_PATH = "https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/refs/heads/master/data/table_to_page.json";
const TOTAL_EXAMPLES_PATH = "https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/refs/heads/master/tokenized_data/total_examples.json";
const R1_TRAINING_ALL_PATH = "https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/refs/heads/master/collected_data/r1_training_all.json";
const R2_TRAINING_ALL_PATH = "https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/refs/heads/master/collected_data/r2_training_all.json";
const FULL_CLEANED_PATH = "https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/refs/heads/master/tokenized_data/full_cleaned.json";
const MANIFEST_JSON_PATH = "results/manifest.json"; // Updated path

// Global variables for precomputed results
let allResults = [];               
let tableIdToResultsMap = {};      
let availableOptions = {
  models: new Set(),
  datasets: new Set(),
  learningTypes: new Set(),
  nValues: new Set(),
  formatTypes: new Set()
};
let globalAbortController = null;
let tableToPageMap = {};  // csv filename -> [title, link]
let resultsChartInstance = null;

// For live inference
let deepSeekPipeline = null;
let currentModelId = null;
// Global variable for table->claims mapping from total_examples.json
let tableIdToClaimsMap = {};
let tableEntityLinkingMap = {}; // mapping for full_cleaned.json (i.e., table_id -> entity linking)

let manifestOptions = []; // Array of manifest options for filtering


// DOM element references
const modelLoadingStatusEl = document.getElementById("modelLoadingStatus");
const liveThinkOutputEl = document.getElementById("liveThinkOutput");
const liveStreamOutputEl = document.getElementById("liveStreamOutput");

document.addEventListener("DOMContentLoaded", async () => {
  try {
    // Attempt to fetch the table-to-page mapping.
    try {
      tableToPageMap = await fetchTableToPage();
    } catch (e) {
      console.warn("Failed to fetch table_to_page.json. Continuing without it.", e);
      tableToPageMap = {};
    }

    // Attempt to fetch and process the manifest and fetch the claims mapping.
    let manifest;
    try {
      manifest = await fetchManifest();
      if (!manifest.results_files || !Array.isArray(manifest.results_files)) {
        console.warn("Manifest does not contain results_files. Using an empty list.");
        manifest.results_files = [];
      }
      parseManifest(manifest); // Populate manifestOptions

      const globalModels    = Array.from(new Set(manifestOptions.map(o => o.model))).sort();
      const globalDatasets  = Array.from(new Set(manifestOptions.map(o => o.dataset))).sort();
      const globalLearningTypes = Array.from(new Set(manifestOptions.map(o => o.learningType))).sort();
      const globalNValues   = Array.from(new Set(manifestOptions.map(o => o.nValue))).sort((a, b) => {
        if (a === "all") return 1;
        if (b === "all") return -1;
        return parseInt(a) - parseInt(b);
      });
      const globalFormatTypes = Array.from(new Set(manifestOptions.map(o => o.formatType))).sort();
      
      populateAllDropdowns(); // Populate dropdowns with all possible values
      populateAllDropdowns();
      ["modelSelect", "datasetSelect", "learningTypeSelect", "nValueSelect", "formatTypeSelect"].forEach(id => {
        document.getElementById(id).addEventListener("change", updateDropdownsAndDisableInvalidOptions);
      });
      updateDropdownsAndDisableInvalidOptions();

    } catch (manifestError) {
      console.warn("Failed to fetch or parse manifest.json. Continuing without manifest.", manifestError);
    }

    await fetchTotalExamplesClaims();
    await fetchFullCleaned();

    // Continue with the rest of the initialization.
    populateExistingTableDropdown();
    addLoadButtonListener();
    setupTabSwitching();
    setupLiveCheckEvents();

    // Set up the performance metrics toggle
    const toggleMetricsEl = document.getElementById("performanceMetricsToggle");
    const toggleArrow = document.getElementById("toggleArrow");
    toggleMetricsEl.addEventListener("click", function() {
      const metricsContent = document.getElementById("metricsContent");
      if (metricsContent.style.display === "none") {
        metricsContent.style.display = "block";
        toggleArrow.textContent = "▼";
        updateNativeMetrics();
      } else {
        metricsContent.style.display = "none";
        toggleArrow.textContent = "►";
      }
    });

  } catch (error) {
    console.error("Initialization failed:", error);
    document.getElementById("infoPanel").innerHTML = `<p style="color:red;">Failed to initialize the app: ${error}</p>`;
  }
});

async function fetchTableToPage() {
  const response = await fetch(TABLE_TO_PAGE_JSON_PATH);
  if (!response.ok) {
    console.warn("Failed to fetch table_to_page.json. Titles/links won't be shown.");
    return {};
  }
  return response.json();
}

async function fetchManifest() {
  const response = await fetch(MANIFEST_JSON_PATH);
  if (!response.ok) {
    throw new Error(`Failed to fetch manifest.json: ${response.status} ${response.statusText}`);
  }
  const manifest = await response.json();
  if (!manifest.results_files || !Array.isArray(manifest.results_files)) {
    throw new Error("Invalid manifest.json format. Missing 'results_files' array.");
  }
  return manifest;
}

function parseManifest(manifest) {
  manifest.results_files.forEach(filename => {
    const shortName = filename.replace(/^results\//, "");
    const regex = /^results_with_cells_(.+?)_(test_examples|val_examples)_(\d+|all)_(zero_shot|one_shot|few_shot|chain_of_thought)_(naturalized|markdown|json|html)\.json$/;
    const match = shortName.match(regex);
    if (match) {
      const [_, model, dataset, nValue, learningType, formatType] = match;
      manifestOptions.push({ model, dataset, nValue, learningType, formatType, filename });
    } else {
      console.warn(`Filename "${filename}" does not match expected pattern; ignoring.`);
    }
  });
}

function populateAllDropdowns() {
  // Extract all values from the manifestOptions
  const models = Array.from(new Set(manifestOptions.map(opt => opt.model))).sort();
  const datasets = Array.from(new Set(manifestOptions.map(opt => opt.dataset))).sort();
  const learningTypes = Array.from(new Set(manifestOptions.map(opt => opt.learningType))).sort();
  const nValues = Array.from(new Set(manifestOptions.map(opt => opt.nValue))).sort((a, b) => {
    if (a === "all") return 1;
    if (b === "all") return -1;
    return parseInt(a) - parseInt(b);
  });
  const formatTypes = Array.from(new Set(manifestOptions.map(opt => opt.formatType))).sort();

  // Populate each select with all possible values plus the "Any" option (empty value)
  populateSelect("modelSelect", models, "", true);
  populateSelect("datasetSelect", datasets, "", true);
  populateSelect("learningTypeSelect", learningTypes, "", true);
  populateSelect("nValueSelect", nValues, "", true);
  populateSelect("formatTypeSelect", formatTypes, "", true);
}

function isValidCombination(model, dataset, learningType, nValue, formatType) {
  // Each parameter is either a value or "" meaning "Any"
  return manifestOptions.some(opt => {
    if (model && opt.model !== model) return false;
    if (dataset && opt.dataset !== dataset) return false;
    if (learningType && opt.learningType !== learningType) return false;
    if (nValue && opt.nValue !== nValue) return false;
    if (formatType && opt.formatType !== formatType) return false;
    return true;
  });
}

function updateDropdownDisabledState(dropdownId, isValidCandidate) {
  const selectEl = document.getElementById(dropdownId);
  Array.from(selectEl.options).forEach(option => {
    // Always allow the "Any" option (assumed to have an empty string as its value)
    if (option.value === "") {
      option.disabled = false;
    } else {
      option.disabled = !isValidCandidate(option.value);
    }
  });
}

function updateDropdownsAndDisableInvalidOptions() {
  const currentModel = document.getElementById("modelSelect").value;
  const currentDataset = document.getElementById("datasetSelect").value;
  const currentLearningType = document.getElementById("learningTypeSelect").value;
  const currentNValue = document.getElementById("nValueSelect").value;
  const currentFormatType = document.getElementById("formatTypeSelect").value;

  // For modelSelect: candidate value is the model while using current values from the other dropdowns.
  updateDropdownDisabledState("modelSelect", candidate =>
    isValidCombination(candidate, currentDataset, currentLearningType, currentNValue, currentFormatType)
  );

  updateDropdownDisabledState("datasetSelect", candidate =>
    isValidCombination(currentModel, candidate, currentLearningType, currentNValue, currentFormatType)
  );

  updateDropdownDisabledState("learningTypeSelect", candidate =>
    isValidCombination(currentModel, currentDataset, candidate, currentNValue, currentFormatType)
  );

  updateDropdownDisabledState("nValueSelect", candidate =>
    isValidCombination(currentModel, currentDataset, currentLearningType, candidate, currentFormatType)
  );

  updateDropdownDisabledState("formatTypeSelect", candidate =>
    isValidCombination(currentModel, currentDataset, currentLearningType, currentNValue, candidate)
  );

  // Check if any selected value is "Select", if so, disable the "Load" button
  const loadBtn = document.getElementById("loadBtn");
  const allValues = [currentModel, currentDataset, currentLearningType, currentNValue, currentFormatType];
  if (allValues.some(v => v === "")) {
    loadBtn.disabled = true;
    loadBtn.style.cursor = "not-allowed";
    loadBtn.style.opacity = "0.5";
    loadBtn.style.pointerEvents = "auto";
  } else {
    loadBtn.disabled = false;
    loadBtn.style.cursor = "pointer";
    loadBtn.style.opacity = "1";
    loadBtn.style.pointerEvents = "auto";
  }
}


async function fetchTotalExamplesClaims() {
  try {
    const response1 = await fetch(R1_TRAINING_ALL_PATH);
    const response2 = await fetch(R2_TRAINING_ALL_PATH);

    if (!response1.ok || !response2.ok) {
      console.warn("Failed to fetch one or both training data files.");
      return;
    }

    const r1Data = await response1.json();
    const r2Data = await response2.json();

    // Combine the two objects.
    // If the same table_id appears in both, the r2Data value will override r1Data's.
    tableIdToClaimsMap = { ...r1Data, ...r2Data };

  } catch (err) {
    console.warn("Could not load training data:", err);
    tableIdToClaimsMap = {};  // fallback
  }
}



async function fetchFullCleaned() {
  try {
    const response = await fetch(FULL_CLEANED_PATH);
    if (!response.ok) {
      console.warn("Failed to fetch full_cleaned.json");
      return;
    }
    tableEntityLinkingMap = await response.json();
  } catch (err) {
    console.warn("Error fetching full_cleaned.json", err);
  }
}


function populateSelect(selectId, values, currentSelection = "", includeAny = true) {
  const sel = document.getElementById(selectId);
  if (!sel) return;
  sel.innerHTML = "";
  if (includeAny) {
    const anyOption = document.createElement("option");
    anyOption.value = "";
    anyOption.textContent = "Select";
    sel.appendChild(anyOption);
  }
  values.forEach(v => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    sel.appendChild(opt);
  });
  // Preserve the selection if possible.
  if (currentSelection && values.includes(currentSelection)) {
    sel.value = currentSelection;
  } else {
    sel.value = includeAny ? "" : (values.length > 0 ? values[0] : "");
  }
}

function populateDropdowns() {
  populateSelect("modelSelect", Array.from(availableOptions.models).sort());
  populateSelect("datasetSelect", Array.from(availableOptions.datasets).sort());
  populateSelect("learningTypeSelect", Array.from(availableOptions.learningTypes).sort());
  populateSelect("nValueSelect", Array.from(availableOptions.nValues).sort((a, b) => {
    if (a === "all") return 1;
    if (b === "all") return -1;
    return parseInt(a) - parseInt(b);
  }));
  populateSelect("formatTypeSelect", Array.from(availableOptions.formatTypes).sort());
}

function addLoadButtonListener() {
  const loadBtn = document.getElementById("loadBtn");
  if (loadBtn) loadBtn.addEventListener("click", loadResults);
}

async function loadResults() {
  const modelName = document.getElementById("modelSelect").value;
  const datasetName = document.getElementById("datasetSelect").value;
  const learningType = document.getElementById("learningTypeSelect").value;
  const nValue = document.getElementById("nValueSelect").value;
  const formatType = document.getElementById("formatTypeSelect").value;
  const resultsFileName = `results/results_with_cells_${modelName}_${datasetName}_${nValue}_${learningType}_${formatType}.json`;
  
  const infoPanel = document.getElementById("infoPanel");
  infoPanel.innerHTML = `<p>Loading results ...</p>`;

  try {
    const response = await fetch(resultsFileName);
    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
    allResults = await response.json();
    infoPanel.innerHTML = `<p>Loaded <strong>${allResults.length}</strong> results for the <strong>${modelName}</strong> model (dataset: ${datasetName}, learning type: ${learningType}, n-value: ${nValue}, format: ${formatType}).</p>`;
    buildTableMap();
    populateTableSelect();
    document.getElementById("tableDropDown").style.display = "block";
    document.getElementById("tableMetaInfo").style.display = "block";
    document.getElementById("performanceMetrics").style.display = "block";
    updateNativeMetrics();
  } catch (err) {
    console.error(`Failed to load ${resultsFileName}:`, err);
    infoPanel.innerHTML = `<p style="color:red;">Failed to load results: ${err}</p>`;
    allResults = [];
    tableIdToResultsMap = {};
    document.getElementById("tableSelect").innerHTML = "";
    document.getElementById("tableSelect").disabled = true;
    document.getElementById("claimList").innerHTML = "";
    document.getElementById("table-container").innerHTML = "";
  }
}

function buildTableMap() {
  tableIdToResultsMap = {};
  allResults.forEach(item => {
    const tid = item.table_id;
    if (!tableIdToResultsMap[tid]) tableIdToResultsMap[tid] = [];
    tableIdToResultsMap[tid].push(item);
  });
}

function populateTableSelect() {
  const tableSelect = document.getElementById("tableSelect");
  tableSelect.innerHTML = "";
  const tableIds = Object.keys(tableIdToResultsMap);
  if (tableIds.length === 0) {
    tableSelect.disabled = true;
    tableSelect.innerHTML = `<option value="">No tables available</option>`;
    return;
  }
  tableSelect.disabled = false;
  tableIds.forEach(tid => {
    const option = document.createElement("option");
    option.value = tid;
    // Use the table title if available
    let title = tableToPageMap[tid] ? tableToPageMap[tid][0] : "";
    option.textContent = title ? `${tid} - ${title}` : tid;
    tableSelect.appendChild(option);
  });
  tableSelect.removeEventListener("change", onTableSelectChange);
  tableSelect.addEventListener("change", onTableSelectChange);
  // Set an initial value and trigger an update
  tableSelect.value = tableIds[0];
  onTableSelectChange();

  // Initialize (or reinitialize) Choices.js on the tableSelect
  if (window.tableSelectChoices) {
    window.tableSelectChoices.destroy();
  }
  window.tableSelectChoices = new Choices('#tableSelect', {
    searchEnabled: true,
    itemSelectText: '',  // Remove the “Press to select” text if you prefer
    shouldSort: false     // (Optional) Prevent Choices from sorting the options
  });
}


function onTableSelectChange() {
  const tableSelect = document.getElementById("tableSelect");
  const selectedTid = tableSelect.value;
  showClaimsForTable(selectedTid);
  updateResultsChart(selectedTid);
}

function showClaimsForTable(tableId) {
  const claimListDiv = document.getElementById("claimList");
  claimListDiv.innerHTML = "";
  const container = document.getElementById("table-container");
  container.innerHTML = "";
  const itemsForTable = tableIdToResultsMap[tableId] || [];
  itemsForTable.forEach((res, idx) => {
    const div = document.createElement("div");
    div.className = "claim-item";
    
    // Determine if the claim was answered correctly.
    const isCorrect = res.predicted_response === res.true_response;
    // Add extra class based on correctness.
    div.classList.add(isCorrect ? "correct" : "incorrect");

    // Create a symbol element: green check or red cross.
    const symbolSpan = document.createElement("span");
    symbolSpan.className = "result-symbol";
    symbolSpan.textContent = isCorrect ? "✓" : "✕";

    // Build the claim text.
    const claimText = document.createTextNode(`Claim #${idx + 1}: ${res.claim}`);
    
    // Append the symbol and text to the div.
    div.appendChild(symbolSpan);
    div.appendChild(claimText);

    // Click event to show full claim details and table.
    div.addEventListener("click", () => {
      document.querySelectorAll(".claim-item").forEach(item => item.classList.remove("selected"));
      div.classList.add("selected");
      renderClaimAndTable(res);
    });
    claimListDiv.appendChild(div);
  });
  if (itemsForTable.length > 0) {
    claimListDiv.firstChild.click();
  }
}


async function renderClaimAndTable(resultObj) {
  document.getElementById("full-highlight-legend-precomputed").style.display = "none";
  document.getElementById("full-entity-highlight-legend-precomputed").style.display = "none";
  const container = document.getElementById("table-container");
  container.innerHTML = "";
  const infoDiv = document.createElement("div");
  infoDiv.className = "info-panel";
  infoDiv.innerHTML = `
    <p><strong>Claim:</strong> ${resultObj.claim}</p>
    <p><strong>Predicted:</strong> ${resultObj.predicted_response ? "TRUE" : "FALSE"}</p>
    <p><strong>Raw Output:</strong> ${resultObj.resp}</p>
    <p><strong>Ground Truth:</strong> ${resultObj.true_response ? "TRUE" : "FALSE"}</p>
  `;
  container.appendChild(infoDiv);

  const metaDiv = document.getElementById("tableMetaInfo");
  metaDiv.innerHTML = "";
  const meta = tableToPageMap[resultObj.table_id];
  if (meta) {
    const [tableTitle, wikipediaUrl] = meta;
    metaDiv.innerHTML = `
      <p><strong>Table Title:</strong> ${tableTitle}</p>
      <p><strong>Wikipedia Link:</strong> <a href="${wikipediaUrl}" target="_blank">${wikipediaUrl}</a></p>
    `;
  } else {
    metaDiv.innerHTML = `<p><em>No title/link found for this table</em></p>`;
  }

  const csvFileName = resultObj.table_id;
  const csvUrl = CSV_BASE_PATH + csvFileName;
  let csvText = "";
  try {
    const resp = await fetch(csvUrl);
    if (!resp.ok) throw new Error(`Status: ${resp.status}`);
    csvText = await resp.text();
  } catch (err) {
    const errMsg = document.createElement("p");
    errMsg.style.color = "red";
    errMsg.textContent = `Failed to load CSV from ${csvUrl}: ${err}`;
    container.appendChild(errMsg);
    return;
  }
  
  const lines = csvText.split(/\r?\n/).filter(line => line.trim().length > 0);
  const tableData = lines.map(line => line.split("#"));
  if (!tableData.length) {
    container.appendChild(document.createElement("p")).textContent = "Table is empty or could not be parsed.";
    return;
  }
  
  const columns = tableData[0];
  const dataRows = tableData.slice(1);
  
  const tableEl = document.createElement("table");
  tableEl.classList.add("styled-table")
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  columns.forEach(col => {
    const th = document.createElement("th");
    th.textContent = col;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  tableEl.appendChild(thead);
  
  const tbody = document.createElement("tbody");
  dataRows.forEach((rowVals, rowIndex) => {
    const tr = document.createElement("tr");
    rowVals.forEach((cellVal, colIndex) => {
      const td = document.createElement("td");
      td.textContent = cellVal;
      const columnName = columns[colIndex];
      const shouldHighlight = resultObj.relevant_cells.some(
        hc => hc.row_index === rowIndex &&
              hc.column_name.toLowerCase() === columnName.toLowerCase()
      );
      if (shouldHighlight) {
        td.classList.add("highlight");
        document.getElementById("full-highlight-legend-precomputed").style.display = "block";
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  tableEl.appendChild(tbody);
  container.appendChild(tableEl);


  // --- Highlight entity-linked cells from full_cleaned.json ---
  if (tableEntityLinkingMap[resultObj.table_id]) {
    // Get the entity linking info. In full_cleaned.json, the first element is an array of statements.
    const entityStatements = tableEntityLinkingMap[resultObj.table_id][0];
    let entityCoords = [];
    const regex = /#([^#]+);(-?\d+),(-?\d+)#/g;
    entityStatements.forEach(statement => {
      let match;
      while ((match = regex.exec(statement)) !== null) {
        // Note: match[2] is the row and match[3] the column, as numbers.
        const row = Number(match[2]);
        const col = Number(match[3]);
        entityCoords.push({ row, col });
      }
    });
    // Now loop over the tbody rows of the table and add the class "entity-highlight" if the cell is in the list.
    const tbody = tableEl.querySelector("tbody");
    if (tbody) {
      Array.from(tbody.rows).forEach((tr, rowIndex) => {
        Array.from(tr.cells).forEach((td, colIndex) => {
          if (entityCoords.some(coord => coord.row === rowIndex && coord.col === colIndex)) {
            td.classList.add("entity-highlight");
          }
        });
      });
    }
    document.getElementById("full-entity-highlight-legend-precomputed").style.display = "block";
  }
}



/**
 * Compute overall performance metrics from allResults and render native plots.
 * Assumes that each result object contains:
 *   - predicted_response (0 or 1)
 *   - true_response (0 or 1)
 */
function updateNativeMetrics() {
  if (!allResults || allResults.length === 0) {
    console.warn("No results available for metrics calculation.");
    return;
  }
  
  // Initialize counts.
  let TP = 0, TN = 0, FP = 0, FN = 0;
  
  allResults.forEach(item => {
    // Skip results where predicted_response is null.
    if (item.predicted_response === null) return;
    if (item.true_response === 1 && item.predicted_response === 1) {
      TP++;
    } else if (item.true_response === 0 && item.predicted_response === 0) {
      TN++;
    } else if (item.true_response === 0 && item.predicted_response === 1) {
      FP++;
    } else if (item.true_response === 1 && item.predicted_response === 0) {
      FN++;
    }
  });
  
  // Create a confusion matrix: rows = actual, columns = predicted.
  const matrix = [
    [TN, FP],
    [FN, TP]
  ];
  
  // Render confusion matrix as a heatmap using Plotly.
  const heatmapData = [{
      z: matrix,
      x: ['Pred. Neg.', 'Pred. Pos.'],
      y: ['Act. Neg.', 'Act. Pos.'],
      type: 'heatmap',
      colorscale: 'Inferno',  // Alternative options: 'Cividis', 'Inferno', 'YlGnBu'
      showscale: false
  }];
  
  const heatmapLayout = {
    title: 'Confusion Matrix',
    annotations: [],
    xaxis: { title: 'Predicted' },
    yaxis: { title: 'Actual' }
  };
  
  // Add annotations (text labels) for each cell.
  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      heatmapLayout.annotations.push({
        x: heatmapData[0].x[j],
        y: heatmapData[0].y[i],
        text: String(matrix[i][j]),
        showarrow: false,
        font: {
          color: 'white'
        }
      });
    }
  }
  
  Plotly.newPlot('confusionMatrixPlot', heatmapData, heatmapLayout);
  
  // Calculate summary statistics.
  const precision = (TP + FP) > 0 ? TP / (TP + FP) : 0;
  const recall    = (TP + FN) > 0 ? TP / (TP + FN) : 0;
  const accuracy  = (TP + TN) / (TP + TN + FP + FN);
  const f1        = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  
  // Render summary statistics as a bar chart.
  const summaryData = [{
    x: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    y: [accuracy, precision, recall, f1],
    type: 'bar',
    //marker: {color: ['#1976d2', '#4caf50', '#ff9800', '#9c27b0']}
  }];
  
  const summaryLayout = {
    title: 'Performance Summary',
    yaxis: { range: [0, 1], title: 'Score' }
  };
  
  Plotly.newPlot('performanceSummaryPlot', summaryData, summaryLayout);
}


function updateResultsChart(tableId) {
  const resultsHeader = document.getElementById("resultsHeader");
  const chartContainer = document.getElementById("chartContainer");
  if (!resultsHeader) return; // safety check if header isn't in the DOM

  // Get results for the selected table
  const results = tableIdToResultsMap[tableId] || [];
  let correctCount = 0;
  let incorrectCount = 0;
  results.forEach(result => {
    if (result.predicted_response === result.true_response) {
      correctCount++;
    } else {
      incorrectCount++;
    }
  });

  // If there are no results, hide the header (which includes the chart)
  if (results.length === 0) {
    resultsHeader.style.display = "none";
    return;
  } else {
    resultsHeader.style.display = "flex";
  }

  // Prepare the data for a bar chart
  const data = {
    labels: ["Correct", "Incorrect"],
    datasets: [{
      label: "Number of Claims",
      data: [correctCount, incorrectCount],
      backgroundColor: ["#4caf50", "#f44336"],
      hoverBackgroundColor: ["#66bb6a", "#ef5350"],
      barThickness: 30  // Adjust this for a more compact look
    }]
  };

  const ctx = document.getElementById("resultsChart").getContext("2d");

  if (resultsChartInstance) {
    resultsChartInstance.data = data;
    resultsChartInstance.update();
  } else {
    resultsChartInstance = new Chart(ctx, {
      type: "bar",
      data: data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function(context) {
                // context.parsed.y contains the count when using a vertical bar chart.
                return `${context.label}: ${context.parsed.y !== undefined ? context.parsed.y : context.parsed}`;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              stepSize: 1  // Makes sure counts are shown as whole numbers
            }
          }
        }
      }
    });
  }
}





//
// LIVE CHECK functions
//
async function populateExistingTableDropdown() {
  const existingTableSelect = document.getElementById("existingTableSelect");
  existingTableSelect.innerHTML = `<option value="">-- Select a Table --</option>`;
  try {
    const response = await fetch("https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/refs/heads/master/data/all_csv_ids.json");
    if (!response.ok) throw new Error(`Failed to fetch all_csv_ids.json: ${response.statusText}`);
    const csvIds = await response.json();
    if (!csvIds || !Array.isArray(csvIds)) throw new Error("Invalid format for all_csv_ids.json.");
    csvIds.sort().forEach(csvFile => {
      const option = document.createElement("option");
      option.value = csvFile;
      // Look up the title from tableToPageMap, if available.
      let meta = tableToPageMap[csvFile];
      option.textContent = meta ? `${csvFile} - ${meta[0]}` : csvFile;
      existingTableSelect.appendChild(option);
    });

    // Initialize (or reinitialize) Choices.js on the existingTableSelect
    if (window.existingTableSelectChoices) {
      window.existingTableSelectChoices.destroy();
    }
    window.existingTableSelectChoices = new Choices('#existingTableSelect', {
      searchEnabled: true,
      itemSelectText: '',
      shouldSort: false
    });

    existingTableSelect.addEventListener("change", async () => {
      const selectedFile = existingTableSelect.value;
      if (!selectedFile) return;
      // 1. Load the CSV for the preview as you already do:
      await fetchAndFillTable(selectedFile);
      // 2. Check if we have claims for this table:
      populateClaimsDropdown(selectedFile);
    });
  } catch (error) {
    console.error("Error loading CSV list:", error);
    alert("Failed to fetch available tables. Please try again later.");
  }
}


async function fetchAndFillTable(tableId) {
  const inputTableEl = document.getElementById("inputTable");
  const previewContainer = document.getElementById("livePreviewTable");
  const liveTableMetaInfo = document.getElementById("liveTableMetaInfo");
  const includeTableNameOption = document.getElementById("includeTableNameOption");

  inputTableEl.value = "";
  previewContainer.innerHTML = "";
  if (liveTableMetaInfo) {
    liveTableMetaInfo.style.display = "none";
    liveTableMetaInfo.innerHTML = "";
  }
  includeTableNameOption.style.display = "none";

  const csvUrl = CSV_BASE_PATH + tableId;
  try {
    const response = await fetch(csvUrl);
    if (!response.ok) throw new Error(`Failed to fetch CSV: ${response.statusText}`);
    const csvText = await response.text();
    inputTableEl.value = csvText;
    renderLivePreviewTable(csvText, []);
    validateLiveCheckInputs();
    const meta = tableToPageMap[tableId];
    if (meta) {
      const [tableTitle, wikipediaUrl] = meta;
      if (liveTableMetaInfo) {
        liveTableMetaInfo.style.display = "block";
        liveTableMetaInfo.innerHTML = `
          <p><strong>Table Title:</strong> ${tableTitle}</p>
          <p><strong>Wikipedia Link:</strong> <a href="${wikipediaUrl}" target="_blank">${wikipediaUrl}</a></p>
        `;
      }
      includeTableNameOption.style.display = "block";
    }
  } catch (error) {
    console.error("Error loading table CSV:", error);
    alert("Failed to load table from dataset.");
  }
}


function populateClaimsDropdown(tableId) {
  const claimsWrapperEl = document.getElementById("existingClaimsWrapper");
  const claimsSelectEl = document.getElementById("existingClaimsSelect");
  
  // Clear out previous content:
  claimsSelectEl.innerHTML = `<option value="">-- Select a Claim --</option>`;
  
  // If we don't have data for this table, hide the wrapper and return
  if (!tableIdToClaimsMap[tableId]) {
    claimsWrapperEl.style.display = "none";
    return;
  }
  
  // The structure in total_examples.json is an array:
  //   [ [claims array], [label array], [some other array], "table title" ]
  // We'll just need the first array for the claim text, 
  // and the second array for whether it's correct(1)/incorrect(0).
  const tableData = tableIdToClaimsMap[tableId];
  if (!Array.isArray(tableData) || tableData.length < 2) {
    claimsWrapperEl.style.display = "none";
    return;
  }
  
  const claimsList = tableData[0];  // the array of claims
  const labelsList = tableData[1];  // the array of 0/1 correctness (optional usage)
  
  // Show the claims dropdown
  claimsWrapperEl.style.display = "block";

  // Fill it with the claims
  claimsList.forEach((claim, idx) => {
    const isCorrect = labelsList[idx] === 1; // or 0
    // We can optionally show correctness in text
    const optionEl = document.createElement("option");
    optionEl.value = idx;  // store index
    optionEl.textContent = `Claim #${idx+1} (${isCorrect ? "TRUE" : "FALSE"}) - ${claim.slice(0,60)}...`;
    claimsSelectEl.appendChild(optionEl);
  });

  // Add change listener to the claims select
  claimsSelectEl.onchange = function() {
    const selectedIndex = claimsSelectEl.value; // index in the array
    if (!selectedIndex) return;

    // The actual claim text is:
    const chosenClaim = claimsList[selectedIndex];
    // Put that into the #inputClaim
    document.getElementById("inputClaim").value = chosenClaim;
    validateLiveCheckInputs(); // re-validate so the "Run" button is enabled
  };
}


function validateLiveCheckInputs() {
  const tableInput = document.getElementById("inputTable").value.trim();
  const claimInput = document.getElementById("inputClaim").value.trim();
  const runLiveCheckBtn = document.getElementById("runLiveCheck");
  const stopLiveCheckBtn = document.getElementById("stopLiveCheck");

  if (tableInput && claimInput && deepSeekPipeline) {
    runLiveCheckBtn.disabled = false;
    runLiveCheckBtn.style.opacity = "1";
    runLiveCheckBtn.style.cursor = "pointer";
    stopLiveCheckBtn.disabled = false;
    stopLiveCheckBtn.style.opacity = "1";
    stopLiveCheckBtn.style.cursor = "pointer";
  } else {
    runLiveCheckBtn.disabled = true;
    runLiveCheckBtn.style.opacity = "0.6";
    runLiveCheckBtn.style.cursor = "not-allowed";
  }
}

function setupTabSwitching() {
  document.querySelectorAll(".mode-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".mode-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById("resultsSection").style.display = "none";
      document.getElementById("liveCheckSection").style.display = "none";
      document.getElementById("reportSection").style.display = "none";
      if (tab.dataset.mode === "precomputed") {
        document.getElementById("resultsSection").style.display = "block";
      } else if (tab.dataset.mode === "live") {
        document.getElementById("liveCheckSection").style.display = "block";
      } else if (tab.dataset.mode === "report") {
        document.getElementById("reportSection").style.display = "block";
        const pdfViewer = document.getElementById("pdfViewer");
        if (!pdfViewer.src || pdfViewer.src === "about:blank") {
          pdfViewer.src = "report.pdf";
        }
      }
    });
  });
}

function setupLiveCheckEvents() {
  const loadModelBtn = document.getElementById("loadLiveModel");
  loadModelBtn.addEventListener("click", async () => {
    const modelId = document.getElementById("liveModelSelect").value;
    const modelName = document.getElementById("liveModelSelect").selectedOptions[0].textContent;
    await initLivePipeline(modelId, modelName);
    validateLiveCheckInputs(); // re-validate so the "Run" button is enabled
  });

  const inputTableEl = document.getElementById("inputTable");
  const inputClaimEl = document.getElementById("inputClaim");
  const runLiveCheckBtn = document.getElementById("runLiveCheck");

  inputTableEl.addEventListener("input", () => {
    const csvText = inputTableEl.value;
    renderLivePreviewTable(csvText, []);
    validateLiveCheckInputs();
  });

  inputClaimEl.addEventListener("input", () => {
    validateLiveCheckInputs();
  });

  inputClaimEl.addEventListener("keypress", (event) => {
      if (event.key === "Enter") {
          event.preventDefault();
          runLiveCheckBtn.click();
      }
  });

  runLiveCheckBtn.addEventListener("click", async () => {
    if (!deepSeekPipeline) {
      console.warn("No pipeline loaded. Please load a model first.");
      modelLoadingStatusEl.textContent = "Model not ready. Please load a model first.";
      return;
    }
    const startTime = performance.now();

    document.getElementById("liveStreamOutput").style.display = "none";
    document.getElementById("liveResults").style.display = "none";

    // Instead of manually setting width, let the flex layout handle it
    // runLiveCheckBtn.style.width = "calc(100% - 40px)";  // removed
    const stopLiveCheckBtn = document.getElementById("stopLiveCheck");
    stopLiveCheckBtn.style.display = "flex";
    stopLiveCheckBtn.classList.add("loading");
    runLiveCheckBtn.disabled = true;
    runLiveCheckBtn.style.opacity = "0.6";
    runLiveCheckBtn.style.cursor = "not-allowed";
  
    const tableInput = inputTableEl.value.trim();
    const claimInput = inputClaimEl.value.trim();
    if (!tableInput || !claimInput) {
      alert("Please provide both a table and a claim before running the live check.");
      return;
    }
  
    const selectedFile = document.getElementById("existingTableSelect").value;
    const includeTitleChecked = document.getElementById("includeTableNameCheck").checked;
    //const tablePrompt = csvToJson(tableInput);
    const tablePrompt = csvToMarkdown(tableInput);
  
    let optionalTitleSection = "";
    if (selectedFile && includeTitleChecked) {
      const meta = tableToPageMap[selectedFile];
      if (meta) {
        const [title, wikiLink] = meta;
        optionalTitleSection = `
The table is titled: "${title}".
Link: ${wikiLink}
`;
      }
    }
  
    const userPrompt = `
You are tasked with determining whether a claim about the following table (in markdown format) is TRUE or FALSE.
Before providing your final answer, explain step-by-step your reasoning process by referring to the relevant parts of the table.
    
Output ONLY valid JSON with:
- "answer" as "TRUE" or "FALSE".
- "relevant_cells" as a list of {row_index, column_name} pairs.
- **Row indices must be numerical** and match the corresponding row.
- **Column names must match exactly** the headers in the table.

${optionalTitleSection}

#### Table (markdown Format):
${tablePrompt}

#### Claim:
"${claimInput}"

Instructions:
- First, list your reasoning steps in a clear and logical order.
- After your explanation, output a final answer in a valid JSON object with the following format:
{{
  "answer": "TRUE" or "FALSE",
  "relevant_cells": [ list of relevant cells as objects with "row_index" and "column_name" ]
}}

Make sure that your output is strictly in this JSON format and nothing else.
"""`.trim();
  
    liveThinkOutputEl.textContent = "";
    liveThinkOutputEl.style.display = "none";
    liveStreamOutputEl.textContent = "";
  
    globalAbortController = new AbortController();
    const signal = globalAbortController.signal;
  
    const selectedModelId = document.getElementById("liveModelSelect").value;
    const isDeepSeek = selectedModelId.includes("DeepSeek");
  
    let buffer = "";
    let inThinkBlock = false;
    let finalText = "";
    let thinkText = "";
  
    const customCallback = (token) => {
      if (signal.aborted) {
        throw new Error("User aborted generation.");
      }
      buffer += token;
      while (true) {
        if (!inThinkBlock) {
          const startIdx = buffer.indexOf("<think>");
          if (startIdx === -1) {
            finalText += buffer;
            buffer = "";
            break;
          } else {
            finalText += buffer.slice(0, startIdx);
            inThinkBlock = true;
            buffer = buffer.slice(startIdx + "<think>".length);
          }
        } else {
          const endIdx = buffer.indexOf("</think>");
          if (endIdx === -1) {
            thinkText += buffer;
            buffer = "";
            break;
          } else {
            thinkText += buffer.slice(0, endIdx);
            inThinkBlock = false;
            buffer = buffer.slice(endIdx + "</think>".length);
          }
        }
      }
      if (isDeepSeek) {
        if (thinkText.trim().length > 0) {
          if (!liveThinkOutputEl.style.display || liveThinkOutputEl.style.display === "none") {
            liveThinkOutputEl.style.display = "block";
          }
          if (!document.getElementById("thinkingLabel")) {
            liveThinkOutputEl.innerHTML = `
              <div class="thinking-overlay">
                <span id="thinkingLabel" class="thinking-label">Thinking...</span>
                <button id="toggleThinkingBtn" class="toggle-thinking">▲</button>
              </div>
              <div id="thinkContent"></div>
            `;
            // Add event listener for the toggle button:
            document.getElementById("toggleThinkingBtn").addEventListener("click", function() {
              const thinkContent = document.getElementById("thinkContent");
              if (thinkContent.style.display === "none") {
                thinkContent.style.display = "block";
                this.textContent = "▲";
              } else {
                thinkContent.style.display = "none";
                this.textContent = "▼";
              }
            });
          }          
          const thinkContentDiv = document.getElementById("thinkContent");
          if (thinkContentDiv) {
            thinkContentDiv.innerHTML = DOMPurify.sanitize(marked.parse(thinkText.trim()));
          }
        }
        // Do NOT update the answer area during generation in deep-seek mode.
      } else {
        finalText += token;
        liveStreamOutputEl.style.display = "block";
        liveStreamOutputEl.innerHTML = `
          <div class="answer-overlay">Answer</div>
          <div id="answerContent">${DOMPurify.sanitize(marked.parse(finalText))}</div>
        `;
      }
      document.querySelectorAll('#thinkContent pre code, #answerContent pre code').forEach((block) => {
        hljs.highlightElement(block);
      });
    };
    
  
    const { TextStreamer } = window._transformers;
    const streamer = new TextStreamer(deepSeekPipeline.tokenizer, {
      skip_prompt: true,
      callback_function: customCallback
    });
  
    let result;
    try {
      const messages = [{ role: "user", content: userPrompt }];
      result = await deepSeekPipeline(messages, {
        max_new_tokens: 1024,
        do_sample: false,
        streamer,
        signal
      });
    } catch (err) {
      runLiveCheckBtn.style.opacity = "1";
      runLiveCheckBtn.disabled = false;
      runLiveCheckBtn.style.cursor = "pointer";
      stopLiveCheckBtn.style.display = "none";
      stopLiveCheckBtn.classList.remove("loading");
  
      if (err.name === "AbortError" || err.message.includes("aborted")) {
        console.warn("Generation was aborted.");
        liveStreamOutputEl.textContent += "\n\n[Generation Aborted]";
        if (isDeepSeek) {
          const endTime = performance.now();
          const secs = ((endTime - startTime) / 1000).toFixed(1);
          const thinkingLabel = document.getElementById("thinkingLabel");
          if (thinkingLabel) {
            thinkingLabel.textContent = `Aborted after ${secs}s.`;
            thinkingLabel.classList.add("done");
          }
        }
        return;
      }
      console.error("Error running pipeline:", err);
      liveStreamOutputEl.textContent += `\n\n[Error: ${err}]`;
      return;
    }
  
    runLiveCheckBtn.style.opacity = "1";
    runLiveCheckBtn.disabled = false;
    runLiveCheckBtn.style.cursor = "pointer";
  
    stopLiveCheckBtn.style.display = "none";
    stopLiveCheckBtn.classList.remove("loading");
  
    if (buffer.length > 0) {
      if (inThinkBlock) {
        thinkText += buffer;
      } else {
        finalText += buffer;
      }
    }
    
    const endTime = performance.now();
    const secs = ((endTime - startTime) / 1000).toFixed(1);
  
    if (isDeepSeek) {
      const thinkingLabel = document.getElementById("thinkingLabel");
      if (thinkingLabel) {
        thinkingLabel.textContent = `Thought for ${secs} seconds.`;
        thinkingLabel.classList.add("done");
      }
      // Now show the answer box once generation is done.
      liveStreamOutputEl.style.display = "block";
      liveStreamOutputEl.innerHTML = `
          <div class="answer-overlay">Answer</div>
          <div id="answerContent">${DOMPurify.sanitize(marked.parse(finalText))}</div>
      `;
    }
  
    const rawResponse = finalText.trim();
    const parsed = extractJsonFromResponse(rawResponse);
    const finalAnswer = parsed.answer || "UNKNOWN";
    const relevantCells = parsed.relevant_cells || [];
  
    displayLiveResults(tableInput, claimInput, finalAnswer, relevantCells);
  });
  
  runLiveCheckBtn.disabled = true;
  runLiveCheckBtn.style.opacity = "0.6";
  runLiveCheckBtn.style.cursor = "not-allowed";
  
  const stopLiveCheckBtn = document.getElementById("stopLiveCheck");
  stopLiveCheckBtn.addEventListener("click", () => {
      if (globalAbortController) {
          globalAbortController.abort();
          console.log("Generation aborted by user.");
      }
      runLiveCheckBtn.style.opacity = "1";
      runLiveCheckBtn.disabled = false;
      runLiveCheckBtn.style.cursor = "pointer";
  
      stopLiveCheckBtn.style.display = "none";
      stopLiveCheckBtn.classList.remove("loading");
  });
}

async function initLivePipeline(modelId, modelName) {
  // If the requested model is already loaded, do nothing.
  if (deepSeekPipeline && currentModelId === modelId) {
    console.log("Model already loaded.");
    return;
  }
  currentModelId = modelId;
  if (deepSeekPipeline) {
    console.log("Clearing previous model from memory...");
    deepSeekPipeline = null;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  const modelLoadingStatusEl = document.getElementById("modelLoadingStatus");
  const loadingTextEl = document.getElementById("modelLoadingText");
  const progressText = document.getElementById("modelLoadingProgress");
  const progressContainer = document.getElementById("modelProgressBarContainer");
  const progressBar = document.getElementById("modelProgressBar");
  const loadModelBtn = document.getElementById("loadLiveModel");
  
  loadModelBtn.disabled = true;
  loadModelBtn.style.opacity = "0.6";
  loadingTextEl.textContent = `Loading model: ${modelName}`;
  loadingTextEl.insertAdjacentHTML("beforeend", `<span class="spinner"></span>`);
  progressBar.style.width = "0%";
  progressText.textContent = `0%`;
  modelLoadingStatusEl.style.display = "block";
  progressContainer.style.display = "block";
  
  console.log(`Initializing pipeline with: ${modelId}`);
  const { pipeline } = window._transformers;
  
  try {
    const progressCallback = (progress) => {
      const percent = Math.round(progress.progress);
      const loadedSizeMB = (progress.loaded / (1024 * 1024)).toFixed(2);
      const totalSizeMB = (progress.total / (1024 * 1024)).toFixed(2);
      progressBar.style.width = `${percent}%`;
      if (!isNaN(loadedSizeMB) && !isNaN(totalSizeMB)) {
        progressText.textContent = `${loadedSizeMB}MB / ${totalSizeMB}MB (${percent}%)`;
      }
    };
  
    let generator;
    try {
      if (modelId.includes("DeepSeek") || modelId.includes("Llama")) {
        generator = await pipeline("text-generation", modelId, {
          dtype: "q4f16",
          device: "webgpu",
          progress_callback: progressCallback
        });
      } else if (modelId.includes("Phi")) {
        generator = await pipeline("text-generation", modelId, {
          dtype: "q4f16",
          device: "webgpu",
          use_external_data_format: true,
          progress_callback: progressCallback
        });
      } 
      else {
        generator = await pipeline("text-generation", modelId, {
          device: "webgpu",
          dtype: "fp32",
          progress_callback: progressCallback
        });
      }
    } catch (gpuErr) {
      console.warn("GPU init failed, falling back to CPU...", gpuErr);
      generator = await pipeline("text-generation", modelId, {
        backend: "wasm",
        progress_callback: progressCallback,
      });
    }
  
    deepSeekPipeline = generator;
    loadingTextEl.textContent = `Model loaded: ${modelName}`;
    progressText.textContent = `100%`;
    progressBar.style.width = `100%`;
    setTimeout(() => {
      progressContainer.style.display = "none";
    }, 1000);
  
  } catch (err) {
    console.error("Failed to init pipeline:", err);
    loadingTextEl.textContent = `Failed to load model: ${err}`;
  } finally {
    loadModelBtn.disabled = false;
    loadModelBtn.style.opacity = "1";
  }
}

/**
 * Render the CSV as an HTML table (live preview).
 */
function renderLivePreviewTable(csvText, relevantCells) {
  const previewContainer = document.getElementById("livePreviewTable");
  previewContainer.innerHTML = "";
  const lines = csvText.split(/\r?\n/).filter(line => line.trim().length > 0);
  if (!lines.length) return;
  const tableData = lines.map(line => line.split("#"));
  if (!tableData.length) return;
  const columns = tableData[0];
  const dataRows = tableData.slice(1);
  const tableEl = document.createElement("table");
  tableEl.classList.add("styled-table")
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  columns.forEach(col => {
    const th = document.createElement("th");
    th.textContent = col;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  tableEl.appendChild(thead);
  const tbody = document.createElement("tbody");
  dataRows.forEach((rowVals, rowIndex) => {
    const tr = document.createElement("tr");
    rowVals.forEach((cellVal, colIndex) => {
      const td = document.createElement("td");
      td.textContent = cellVal;
      const colName = columns[colIndex];
      const shouldHighlight = relevantCells.some(
        hc => hc.row_index === rowIndex &&
              hc.column_name?.toLowerCase() === colName.toLowerCase()
      );
      if (shouldHighlight) td.classList.add("highlight");
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  tableEl.appendChild(tbody);


  // --- If an existing table is selected, highlight entity-linked cells ---
  const existingTableSelect = document.getElementById("existingTableSelect");
  if (existingTableSelect && existingTableSelect.value && tableEntityLinkingMap[existingTableSelect.value]) {
    const entityStatements = tableEntityLinkingMap[existingTableSelect.value][0];
    let entityCoords = [];
    const regex = /#([^#]+);(-?\d+),(-?\d+)#/g;
    entityStatements.forEach(statement => {
      let match;
      while ((match = regex.exec(statement)) !== null) {
        const row = Number(match[2]);
        const col = Number(match[3]);
        entityCoords.push({ row, col });
      }
    });
    const tbody = tableEl.querySelector("tbody");
    if (tbody) {
      Array.from(tbody.rows).forEach((tr, rowIndex) => {
        Array.from(tr.cells).forEach((td, colIndex) => {
          if (entityCoords.some(coord => coord.row === rowIndex && coord.col === colIndex)) {
            td.classList.add("entity-highlight");
          }
        });
      });
    }
  }


  previewContainer.appendChild(tableEl);

  // --- Update the dynamic legend based on highlighted cells ---
  const legendModel = document.getElementById("full-highlight-legend-live");
  const legendEntity = document.getElementById("full-entity-highlight-legend-live");

  // Check if any cells in the table have the model highlighting (".highlight")
  if (tableEl.querySelectorAll("td.highlight").length > 0) {
    legendModel.style.display = "block"; // or "block", depending on your styling
  } else {
    legendModel.style.display = "none";
  }

  // Check if any cells in the table have the entity linking highlight (".entity-highlight")
  if (tableEl.querySelectorAll("td.entity-highlight").length > 0) {
    legendEntity.style.display = "block"; // or "block"
  } else {
    legendEntity.style.display = "none";
  }
}

function displayLiveResults(csvText, claim, answer, relevantCells) {
  document.getElementById("liveResults").style.display = "block";
  const liveClaimList = document.getElementById("liveClaimList");
  liveClaimList.innerHTML = "";
  const claimDiv = document.createElement("div");
  claimDiv.className = "claim-item selected";
  claimDiv.textContent = `Claim: "${claim}" => Model says: ${answer}`;
  liveClaimList.appendChild(claimDiv);
  renderLivePreviewTable(csvText, relevantCells);
}

function csvToMarkdown(csvStr) {
  const lines = csvStr.trim().split(/\r?\n/);
  if (!lines.length) return "";
  const tableData = lines.map(line => line.split("#"));
  if (!tableData.length) return "";
  const headers = tableData[0];
  const rows = tableData.slice(1);
  let md = `| ${headers.join(" | ")} |\n`;
  md += `| ${headers.map(() => "---").join(" | ")} |\n`;
  rows.forEach(row => {
    md += `| ${row.join(" | ")} |\n`;
  });
  return md;
}

function csvToJson(csvStr) {
  const lines = csvStr.trim().split(/\r?\n/);
  if (!lines.length) return "{}";
  const headers = lines[0].split("#");
  const rows = lines.slice(1).map(line => line.split("#"));
  return JSON.stringify({ columns: headers, data: rows }, null, 2);
}

function extractJsonFromResponse(rawResponse) {
  let jsonText = rawResponse.trim();
  const fencePattern = /```json\s*([\s\S]*?)\s*```/i;
  const fenceMatch = jsonText.match(fencePattern);
  if (fenceMatch) {
    jsonText = fenceMatch[1].trim();
  }
  jsonText = jsonText.replace(/'/g, '"');
  jsonText = jsonText.replace(/(\{|,)\s*([\w\d_]+)\s*:/g, '$1 "$2":');
  jsonText = jsonText.replace(/:\s*(TRUE|FALSE)([\s,\}])/gi, ': "$1"$2');
  let parsed;
  try {
    parsed = JSON.parse(jsonText);
  } catch (err) {
    console.warn("[extractJsonFromResponse] Could not parse JSON:", err);
    console.log("Attempted JSON Text:", jsonText);
    return {};
  }
  if (Array.isArray(parsed.relevant_cells)) {
    parsed.relevant_cells = parsed.relevant_cells.map(cell => {
      if (Array.isArray(cell) && cell.length === 2) {
        return {
          row_index: Number(cell[0]),
          column_name: String(cell[1])
        };
      } else if (typeof cell === "object" && cell.row_index !== undefined && cell.column_name) {
        return cell;
      }
      return null;
    }).filter(Boolean);
  }
  return parsed;
}

function separateThinkFromResponse(rawText) {
  const thinkRegex = /<think>([\s\S]*?)<\/think>/i;
  const match = rawText.match(thinkRegex);
  let thinkContent = "";
  let remainder = rawText;
  if (match) {
    thinkContent = match[1].trim();
    remainder = rawText.replace(thinkRegex, "").trim();
  }
  return { think: thinkContent, noThink: remainder };
}
