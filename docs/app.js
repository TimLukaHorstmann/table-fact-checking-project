// app.js

// Base URL or relative path to your CSV folder
const CSV_BASE_PATH = "https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/refs/heads/master/data/all_csv/";

// Path to your manifest.json
const MANIFEST_JSON_PATH = "manifest.json"; // Adjust if it's in a different location

let allResults = []; // Array of results objects
let tableIdToResultsMap = {};
let availableOptions = {
  models: new Set(),
  datasets: new Set(),
  learningTypes: new Set(),
  nValues: new Set(),
  formatTypes: new Set()
};

// 1) On page load, fetch manifest.json and populate dropdowns
document.addEventListener("DOMContentLoaded", () => {
  initializeApp();
});

// Initialize the application
async function initializeApp() {
  try {
    const manifest = await fetchManifest();
    parseManifest(manifest);
    populateDropdowns();
    addLoadButtonListener();
  } catch (error) {
    console.error("Initialization failed:", error);
    const infoPanel = document.getElementById("infoPanel");
    infoPanel.innerHTML = `<p style="color:red;">Failed to initialize the app: ${error}</p>`;
  }
}

// Fetch manifest.json
async function fetchManifest() {
  const response = await fetch(MANIFEST_JSON_PATH);
  if (!response.ok) {
    throw new Error(`Failed to fetch manifest.json: ${response.status} ${response.statusText}`);
  }
  const manifest = await response.json();
  if (!manifest.results_files || !Array.isArray(manifest.results_files)) {
    throw new Error("Invalid manifest.json format.");
  }
  return manifest;
}

// Parse manifest.json and extract options
function parseManifest(manifest) {
  manifest.results_files.forEach(filename => {
    // Expected filename format:
    // results_with_cells_{MODEL}_{DATASET}_{N}_{LEARNING_TYPE}_{FORMAT_TYPE}.json
    const regex = /^results_with_cells_(.+?)_(test_set|val_set)_(\d+|all)_(zero_shot|one_shot|few_shot)_(naturalized|markdown)\.json$/;
    const match = filename.match(regex);
    if (match) {
      const [_, model, dataset, nValue, learningType, formatType] = match;
      availableOptions.models.add(model);
      availableOptions.datasets.add(dataset);
      availableOptions.learningTypes.add(learningType);
      availableOptions.nValues.add(nValue);
      availableOptions.formatTypes.add(formatType);
    } else {
      console.warn(`Filename "${filename}" does not match the expected pattern and will be ignored.`);
    }
  });
}

// Populate dropdowns based on available options
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

// A helper to populate a <select> with an array of strings
function populateSelect(selectId, values) {
  const sel = document.getElementById(selectId);
  sel.innerHTML = ""; // clear
  values.forEach(v => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    sel.appendChild(opt);
  });
}

// Add event listener to "Load Results" button
function addLoadButtonListener() {
  const loadBtn = document.getElementById("loadBtn");
  loadBtn.addEventListener("click", loadResults);
}

// Load the selected results JSON file
async function loadResults() {
  // Grab user selections
  const modelName = document.getElementById("modelSelect").value;
  const datasetName = document.getElementById("datasetSelect").value;
  const learningType = document.getElementById("learningTypeSelect").value;
  const nValue = document.getElementById("nValueSelect").value;
  const formatType = document.getElementById("formatTypeSelect").value;

  // Construct the filename
  // E.g. "results_with_cells_mistral_test_set_2_zero_shot_naturalized.json"
  const resultsFileName = `results_with_cells_${modelName}_${datasetName}_${nValue}_${learningType}_${formatType}.json`;

  // Info panel
  const infoPanel = document.getElementById("infoPanel");
  infoPanel.innerHTML = `<p>Loading <strong>${resultsFileName}</strong> ...</p>`;

  try {
    const response = await fetch(resultsFileName);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    allResults = await response.json();
    infoPanel.innerHTML = `<p>Loaded <strong>${allResults.length}</strong> results from <strong>${resultsFileName}</strong>.</p>`;
    buildTableMap();
    populateTableSelect();
  } catch (err) {
    console.error(`Failed to load or parse ${resultsFileName}:`, err);
    infoPanel.innerHTML = `<p style="color:red;">Failed to load <strong>${resultsFileName}</strong>: ${err}</p>`;
    allResults = [];
    tableIdToResultsMap = {};
    document.getElementById("tableSelect").innerHTML = "";
    document.getElementById("tableSelect").disabled = true;
    document.getElementById("claimList").innerHTML = "";
    document.getElementById("table-container").innerHTML = "";
  }
}

// Build a map: table_id -> array of results
function buildTableMap() {
  tableIdToResultsMap = {};
  allResults.forEach(item => {
    const tid = item.table_id;
    if (!tableIdToResultsMap[tid]) {
      tableIdToResultsMap[tid] = [];
    }
    tableIdToResultsMap[tid].push(item);
  });
}

// Populate the tableSelect dropdown
function populateTableSelect() {
  const tableSelect = document.getElementById("tableSelect");
  tableSelect.innerHTML = ""; // clear
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
    option.textContent = tid;
    tableSelect.appendChild(option);
  });

  tableSelect.removeEventListener("change", onTableSelectChange); // Prevent multiple bindings
  tableSelect.addEventListener("change", onTableSelectChange);

  // By default, pick the first
  tableSelect.value = tableIds[0];
  onTableSelectChange();
}

// Handle table selection change
function onTableSelectChange() {
  const tableSelect = document.getElementById("tableSelect");
  const selectedTid = tableSelect.value;
  showClaimsForTable(selectedTid);
}

// Display claims for the selected table
function showClaimsForTable(tableId) {
    const claimListDiv = document.getElementById("claimList");
    claimListDiv.innerHTML = "";
  
    const container = document.getElementById("table-container");
    container.innerHTML = "";
  
    if (!tableIdToResultsMap[tableId]) return;
  
    const itemsForTable = tableIdToResultsMap[tableId];
    itemsForTable.forEach((res, idx) => {
      const div = document.createElement("div");
      div.className = "claim-item";
      div.textContent = `Claim #${idx + 1}: ${res.claim}`;
  
      // On click, select this claim and render the table
      div.addEventListener("click", () => {
        // Remove 'selected' from all .claim-item elements
        document.querySelectorAll(".claim-item").forEach(item => {
          item.classList.remove("selected");
        });
  
        // Add 'selected' to the one that was clicked
        div.classList.add("selected");
  
        // Render claim & table
        renderClaimAndTable(res);
      });
  
      claimListDiv.appendChild(div);
    });
  
    // Optionally auto-show first claim
    if (itemsForTable.length > 0) {
      // Simulate a click on the first claim
      claimListDiv.firstChild.click();
    }
  }

// Render the claim and the corresponding table with highlights
async function renderClaimAndTable(resultObj) {
  const container = document.getElementById("table-container");
  container.innerHTML = "";

  // Show claim info
  const infoDiv = document.createElement("div");
  infoDiv.className = "info-panel";
  infoDiv.innerHTML = `
    <p><strong>Claim:</strong> ${resultObj.claim}</p>
    <p><strong>Predicted Label:</strong> ${resultObj.predicted_response ? "TRUE" : "FALSE"}</p>
    <p><strong>Model Raw Output:</strong> ${resultObj.resp}</p>
    <p><strong>Ground Truth:</strong> ${resultObj.true_response ? "TRUE" : "FALSE"}</p>
  `;
  container.appendChild(infoDiv);

  // Build CSV URL from the table_id
  const csvFileName = resultObj.table_id;
  const csvUrl = CSV_BASE_PATH + csvFileName;

  // Fetch CSV
  let csvText = "";
  try {
    const resp = await fetch(csvUrl);
    if (!resp.ok) {
      throw new Error(`Status: ${resp.status}`);
    }
    csvText = await resp.text();
  } catch (err) {
    const errMsg = document.createElement("p");
    errMsg.style.color = "red";
    errMsg.textContent = `Failed to load CSV from ${csvUrl}: ${err}`;
    container.appendChild(errMsg);
    return;
  }

  // Parse CSV lines
  const lines = csvText.split(/\r?\n/).filter(line => line.trim().length > 0);
  const tableData = lines.map(line => line.split("#"));
  if (!tableData || tableData.length === 0) {
    const msg = document.createElement("p");
    msg.textContent = "Table is empty or couldn't parse properly.";
    container.appendChild(msg);
    return;
  }

  const columns = tableData[0];
  const dataRows = tableData.slice(1);

  // Create HTML table
  const tableEl = document.createElement("table");

  // table head
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  columns.forEach(col => {
    const th = document.createElement("th");
    th.textContent = col;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  tableEl.appendChild(thead);

  // table body
  const tbody = document.createElement("tbody");
  dataRows.forEach((rowValues, rowIndex) => {
    const tr = document.createElement("tr");
    rowValues.forEach((cellVal, colIndex) => {
      const td = document.createElement("td");
      td.textContent = cellVal;

      // Highlight if in resultObj.highlighted_cells
      const columnName = columns[colIndex];
      const highlight = resultObj.highlighted_cells.some(
        hc => hc.row_index === rowIndex && hc.column_name === columnName
      );
      if (highlight) {
        td.classList.add("highlight");
      }

      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  tableEl.appendChild(tbody);

  container.appendChild(tableEl);
}