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

// We'll store the pipeline object here once loaded
let deepSeekPipeline = null;

// UI elements for convenience
const modelLoadingStatusEl = document.getElementById("modelLoadingStatus");
const liveStreamOutputEl = document.getElementById("liveStreamOutput");

// 1) On page load, fetch manifest.json and populate dropdowns
document.addEventListener("DOMContentLoaded", async () => {
  await initializeApp();
  await initDeepSeekPipeline();
});

// Initialize the application
async function initializeApp() {
  try {
    const manifest = await fetchManifest();
    parseManifest(manifest);
    populateDropdowns();
    addLoadButtonListener();

    // Tab switching
    document.querySelectorAll('.mode-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        document.getElementById('resultsSection').style.display = 
          tab.dataset.mode === 'precomputed' ? 'block' : 'none';
        document.getElementById('liveCheckSection').style.display = 
          tab.dataset.mode === 'live' ? 'block' : 'none';
      });
    });

  } catch (error) {
    console.error("Initialization failed:", error);
    const infoPanel = document.getElementById("infoPanel");
    infoPanel.innerHTML = `<p style="color:red;">Failed to initialize the app: ${error}</p>`;
  }
}

// Initialize the text-generation pipeline with a text streamer
async function initDeepSeekPipeline() {
  try {
    // Indicate loading
    modelLoadingStatusEl.textContent = "Loading model... ";
    modelLoadingStatusEl.innerHTML += `<span class="spinner"></span>`; // optional spinner

    console.log("🚀 Initializing DeepSeek pipeline...");

    // Import from the global scope
    const { pipeline, TextStreamer } = window._transformers;

    // Create a text-generation pipeline
    const generator = await pipeline(
      "text-generation",
      "onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX",
      { dtype: "q4f16" } // Possibly switch to { backend: 'wasm' } if WebGPU fails
    );

    // Mark pipeline as loaded
    deepSeekPipeline = generator;

    console.log("✅ Model loaded successfully!");
    modelLoadingStatusEl.textContent = "Model loaded successfully!";
  } catch (err) {
    console.error("🔥 Failed to initialize DeepSeek pipeline:", err);
    modelLoadingStatusEl.textContent = `Model failed to load: ${err}`;
  }
}

function csvToMarkdown(csvStr) {
  const lines = csvStr.trim().split(/\r?\n/);
  const tableData = lines.map(line => line.split("#")); 
  if (tableData.length === 0) return "";

  const headers = tableData[0];
  const rows = tableData.slice(1);

  let md = `| ${headers.join(" | ")} |\n`;
  md += `| ${headers.map(() => "---").join(" | ")} |\n`;
  rows.forEach(row => {
    md += `| ${row.join(" | ")} |\n`;
  });

  return md;
}

// Display final results in the UI
function displayLiveResults(csvText, claim, answer, highlightedCells) {
  const liveClaimList = document.getElementById("liveClaimList");
  const liveTableContainer = document.getElementById("liveTableContainer");

  // Clear old content
  liveClaimList.innerHTML = "";
  liveTableContainer.innerHTML = "";

  // Show the claim + answer
  const claimDiv = document.createElement("div");
  claimDiv.className = "claim-item selected";
  claimDiv.textContent = `Claim: ${claim} => Model says: ${answer}`;
  liveClaimList.appendChild(claimDiv);

  // Build the table
  const lines = csvText.split(/\r?\n/).filter(line => line.trim().length > 0);
  const tableData = lines.map(line => line.split("#")); 
  if (!tableData.length) return;

  const columns = tableData[0];
  const dataRows = tableData.slice(1);

  const tableEl = document.createElement("table");

  // Header
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  columns.forEach(col => {
    const th = document.createElement("th");
    th.textContent = col;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  tableEl.appendChild(thead);

  // Body
  const tbody = document.createElement("tbody");
  dataRows.forEach((rowValues, rowIndex) => {
    const tr = document.createElement("tr");
    rowValues.forEach((cellVal, colIndex) => {
      const td = document.createElement("td");
      td.textContent = cellVal;

      // Check if we should highlight
      const colName = columns[colIndex];
      const shouldHighlight = highlightedCells.some(
        hc => hc.row_index === rowIndex && hc.column_name?.toLowerCase() === colName.toLowerCase()
      );
      if (shouldHighlight) {
        td.classList.add("highlight");
      }

      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  tableEl.appendChild(tbody);

  liveTableContainer.appendChild(tableEl);
}

// Handler for the "Run Live Check" button
document.getElementById("runLiveCheck").addEventListener("click", async () => {
  if (!deepSeekPipeline) {
    console.error("DeepSeek pipeline not yet initialized!");
    modelLoadingStatusEl.textContent = "Model not ready. Please wait...";
    return;
  }

  // 1) Read user input
  const tableInput = document.getElementById("inputTable").value;
  const claimInput = document.getElementById("inputClaim").value;

  // 2) Convert CSV -> Markdown
  const tablePrompt = csvToMarkdown(tableInput);

  // 3) We'll create a single user "message"
  const userPrompt = `
You are given a table:
${tablePrompt}

Claim: "${claimInput}"

Return JSON:
{"answer": "TRUE" or "FALSE", "highlighted_cells": [{"row_index": number, "column_name": string}]}
  `.trim();

  // Clear any old streaming text
  liveStreamOutputEl.textContent = "";

  // 4) Prepare a TextStreamer to show partial tokens
  const { TextStreamer } = window._transformers;
  const streamer = new TextStreamer(deepSeekPipeline.tokenizer, {
    skip_prompt: true,
    // This callback gets called for every new token
    callback_function: (token) => {
      liveStreamOutputEl.textContent += token;
    }
  });

  let result;
  try {
    // 5) Call the pipeline with messages + streaming
    //    DeepSeek uses chat-like format (role + content).
    const messages = [
      { role: "user", content: userPrompt },
    ];

    result = await deepSeekPipeline(messages, {
      max_new_tokens: 128,
      do_sample: false,
      streamer
    });

    console.log("DeepSeek raw result:", result);

  } catch (err) {
    console.error("Error running DeepSeek pipeline:", err);
    liveStreamOutputEl.textContent += `\n\n[Error: ${err}]`;
    return;
  }

  // 6) The final text is in `result[0].generated_text` (an array of message objects).
  const rawResponse = Array.isArray(result) && result.length > 0 
    ? result[0].generated_text?.at(-1)?.content || ""
    : "";

  // 7) Try to parse JSON from the rawResponse
  let parsed = {};
  try {
    parsed = JSON.parse(rawResponse);
  } catch (e) {
    console.warn("Could not parse JSON from model output. Raw text:\n", rawResponse);
  }

  const finalAnswer = parsed.answer || "UNKNOWN";
  const highlightedCells = parsed.highlighted_cells || [];

  // 8) Update the UI with final answer + highlight
  displayLiveResults(tableInput, claimInput, finalAnswer, highlightedCells);
});

// -------------------------
// Manifest + precomputed results logic
// -------------------------

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

  tableSelect.removeEventListener("change", onTableSelectChange);
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
      document.querySelectorAll(".claim-item").forEach(item => {
        item.classList.remove("selected");
      });
      div.classList.add("selected");
      renderClaimAndTable(res);
    });

    claimListDiv.appendChild(div);
  });

  // Auto-show first claim
  if (itemsForTable.length > 0) {
    claimListDiv.firstChild.click();
  }
}

// Render the claim and the corresponding table with highlights
async function renderClaimAndTable(resultObj) {
  const container = document.getElementById("table-container");
  container.innerHTML = "";

  const infoDiv = document.createElement("div");
  infoDiv.className = "info-panel";
  infoDiv.innerHTML = `
    <p><strong>Claim:</strong> ${resultObj.claim}</p>
    <p><strong>Predicted Label:</strong> ${resultObj.predicted_response ? "TRUE" : "FALSE"}</p>
    <p><strong>Model Raw Output:</strong> ${resultObj.resp}</p>
    <p><strong>Ground Truth:</strong> ${resultObj.true_response ? "TRUE" : "FALSE"}</p>
  `;
  container.appendChild(infoDiv);

  const csvFileName = resultObj.table_id;
  const csvUrl = CSV_BASE_PATH + csvFileName;

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

  const tableEl = document.createElement("table");

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
  dataRows.forEach((rowValues, rowIndex) => {
    const tr = document.createElement("tr");
    rowValues.forEach((cellVal, colIndex) => {
      const td = document.createElement("td");
      td.textContent = cellVal;

      const columnName = columns[colIndex];
      const highlight = resultObj.highlighted_cells.some(
        hc =>
          hc.row_index === rowIndex &&
          hc.column_name.toLowerCase() === columnName.toLowerCase()
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
