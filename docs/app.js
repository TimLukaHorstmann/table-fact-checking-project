
Copy
// app.js

// Remove hard-coded arrays
const CSV_BASE_PATH = "https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/refs/heads/master/data/all_csv/";
let allResults = [];
let tableIdToResultsMap = {};

document.addEventListener("DOMContentLoaded", async () => {
  try {
    // 1. Fetch manifest with available JSON files
    const manifestResponse = await fetch('manifest.json');
    if (!manifestResponse.ok) throw new Error("Manifest load failed");
    const filenames = await manifestResponse.json();
    
    // 2. Extract parameters from filenames
    const { models, datasets, nValues, learningTypes, formatTypes } = parseManifest(filenames);

    // 3. Populate dropdowns dynamically
    populateSelect("modelSelect", models);
    populateSelect("datasetSelect", datasets);
    populateSelect("nValueSelect", nValues);
    populateSelect("learningTypeSelect", learningTypes);
    populateSelect("formatTypeSelect", formatTypes);

    document.getElementById("loadBtn").addEventListener("click", loadResults);
  } catch (error) {
    console.error("Initialization error:", error);
    document.getElementById("infoPanel").innerHTML = 
      `<p style="color:red;">Error initializing: ${error.message}</p>`;
  }
});

// Helper to parse filenames and extract parameters
function parseManifest(filenames) {
  const params = {
    models: new Set(),
    datasets: new Set(),
    nValues: new Set(),
    learningTypes: new Set(),
    formatTypes: new Set()
  };

  filenames.forEach(filename => {
    const parsed = parseFilename(filename);
    if (parsed) {
      params.models.add(parsed.model);
      params.datasets.add(parsed.dataset);
      params.nValues.add(parsed.n);
      params.learningTypes.add(parsed.learningType);
      params.formatTypes.add(parsed.formatType);
    }
  });

  // Convert sets to sorted arrays
  return {
    models: [...params.models].sort(),
    datasets: [...params.datasets].sort(),
    nValues: [...params.nValues].sort(),
    learningTypes: [...params.learningTypes].sort(),
    formatTypes: [...params.formatTypes].sort()
  };
}

// Filename parser (assumes specific structure)
function parseFilename(filename) {
  const pattern = /^results_with_cells_([^_]+)_(.+?)_([^_]+)_(.+?)_(.+?)\.json$/;
  const match = filename.match(pattern);
  
  if (!match) return null;

  return {
    model: match[1],
    dataset: match[2], // Allows underscores in dataset name
    n: match[3],
    learningType: match[4], // Allows underscores in learning type
    formatType: match[5]
  };
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
    infoPanel.innerHTML = `<p>Loaded <strong>${allResults.length}</strong> results from ${resultsFileName}</p>`;
    buildTableMap();
  } catch (err) {
    console.error(`Failed to load or parse ${resultsFileName}:`, err);
    infoPanel.innerHTML = `<p style="color:red;">Failed to load ${resultsFileName}: ${err}</p>`;
    allResults = [];
    tableIdToResultsMap = {};
  }

  // Now that we have loaded results, populate tableSelect
  populateTableSelect();
}

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

function populateTableSelect() {
  const tableSelect = document.getElementById("tableSelect");
  tableSelect.innerHTML = ""; // clear
  const tableIds = Object.keys(tableIdToResultsMap);

  if (tableIds.length === 0) {
    tableSelect.disabled = true;
    return;
  }

  tableSelect.disabled = false;
  tableIds.forEach(tid => {
    const option = document.createElement("option");
    option.value = tid;
    option.textContent = tid;
    tableSelect.appendChild(option);
  });

  tableSelect.addEventListener("change", onTableSelectChange);

  // By default, pick the first
  tableSelect.value = tableIds[0];
  onTableSelectChange();
}

function onTableSelectChange() {
  const tableSelect = document.getElementById("tableSelect");
  const selectedTid = tableSelect.value;
  showClaimsForTable(selectedTid);
}

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
    div.textContent = `Claim #${idx+1}: ${res.claim}`;
    div.addEventListener("click", () => {
      renderClaimAndTable(res);
    });
    claimListDiv.appendChild(div);
  });

  // Optionally auto-show first claim
  if (itemsForTable.length > 0) {
    renderClaimAndTable(itemsForTable[0]);
  }
}

async function renderClaimAndTable(resultObj) {
  const container = document.getElementById("table-container");
  container.innerHTML = "";

  // Show claim info
  const infoDiv = document.createElement("div");
  infoDiv.className = "info-panel";
  infoDiv.innerHTML = `
    <p><b>Claim:</b> ${resultObj.claim}</p>
    <p><b>Predicted Label:</b> ${resultObj.predicted_response ? "TRUE" : "FALSE"}</p>
    <p><b>Model Raw Output:</b> ${resultObj.resp}</p>
    <p><b>Ground Truth:</b> ${resultObj.true_response ? "TRUE" : "FALSE"}</p>
  `;
  container.appendChild(infoDiv);

  // Build CSV URL from the table_id
  const csvFileName = resultObj.table_id;
  const csvUrl = CSV_BASE_PATH + csvFileName;

  // fetch CSV
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

      // highlight if in resultObj.highlighted_cells
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