//
// app.js
//

// Base URL for the CSVs
const CSV_BASE_PATH = "https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/refs/heads/master/data/all_csv/";

// Path to your manifest.json
const MANIFEST_JSON_PATH = "manifest.json";

//
// Global variables
//
let allResults = [];               // For precomputed results
let tableIdToResultsMap = {};      // table_id -> array of results
let availableOptions = {
  models: new Set(),
  datasets: new Set(),
  learningTypes: new Set(),
  nValues: new Set(),
  formatTypes: new Set()
};

// We'll store the pipeline object here for the live check
let deepSeekPipeline = null;

// Some quick references
const modelLoadingStatusEl = document.getElementById("modelLoadingStatus");
const liveStreamOutputEl = document.getElementById("liveStreamOutput");

// ---------------------
// On page load
// ---------------------
document.addEventListener("DOMContentLoaded", async () => {
  try {
    // 1) Initialize the precomputed side
    const manifest = await fetchManifest();
    parseManifest(manifest);
    populateDropdowns();
    addLoadButtonListener();

    // 2) Setup tab switching
    document.querySelectorAll(".mode-tab").forEach(tab => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".mode-tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");

        const showPrecomputed = (tab.dataset.mode === "precomputed");
        document.getElementById("resultsSection").style.display = showPrecomputed ? "block" : "none";
        document.getElementById("liveCheckSection").style.display = showPrecomputed ? "none" : "block";
      });
    });

    // 3) Setup events for the live check portion
    setupLiveCheckEvents();

  } catch (error) {
    console.error("Initialization failed:", error);
    const infoPanel = document.getElementById("infoPanel");
    if (infoPanel) {
      infoPanel.innerHTML = `<p style="color:red;">Failed to initialize the app: ${error}</p>`;
    }
  }
});

// ---------------------
// PRECOMPUTED RESULTS
// ---------------------

/**
 * Fetch the manifest.json.
 * It should contain something like:
 * {
 *   "results_files": [
 *     "results/results_with_cells_gpt4_test_set_1_zero_shot_naturalized.json",
 *     ...
 *   ]
 * }
 */
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

/**
 * Parse the manifest file, extracting model/dataset/learning/etc. info
 */
function parseManifest(manifest) {
  manifest.results_files.forEach(filename => {
    // If the file path starts with "results/", remove that so we can match the pattern easily
    const shortName = filename.replace(/^results\//, "");

    // We expect something like:
    //   results_with_cells_{MODEL}_{DATASET}_{N}_{LEARNING_TYPE}_{FORMAT_TYPE}.json
    const regex = /^results_with_cells_(.+?)_(test_set|val_set)_(\d+|all)_(zero_shot|one_shot|few_shot)_(naturalized|markdown)\.json$/;
    const match = shortName.match(regex);
    if (match) {
      const [_, model, dataset, nValue, learningType, formatType] = match;
      availableOptions.models.add(model);
      availableOptions.datasets.add(dataset);
      availableOptions.learningTypes.add(learningType);
      availableOptions.nValues.add(nValue);
      availableOptions.formatTypes.add(formatType);
    } else {
      console.warn(`Filename "${filename}" does not match the expected pattern; ignoring.`);
    }
  });
}

/**
 * Populate the 5 dropdowns for precomputed results
 */
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

/**
 * Helper to populate a <select> with an array of values
 */
function populateSelect(selectId, values) {
  const sel = document.getElementById(selectId);
  if (!sel) return;
  sel.innerHTML = "";
  values.forEach(v => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    sel.appendChild(opt);
  });
}

/**
 * Add event listener to the "Load Results" button in the precomputed section
 */
function addLoadButtonListener() {
  const loadBtn = document.getElementById("loadBtn");
  if (loadBtn) {
    loadBtn.addEventListener("click", loadResults);
  }
}

/**
 * Load the selected results JSON file based on user dropdowns
 */
async function loadResults() {
  const modelName = document.getElementById("modelSelect").value;
  const datasetName = document.getElementById("datasetSelect").value;
  const learningType = document.getElementById("learningTypeSelect").value;
  const nValue = document.getElementById("nValueSelect").value;
  const formatType = document.getElementById("formatTypeSelect").value;

  // Our JSON files live in the "results/" subfolder
  const resultsFileName = `results/results_with_cells_${modelName}_${datasetName}_${nValue}_${learningType}_${formatType}.json`;

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

/**
 * Build a map: table_id -> array of results
 */
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

/**
 * Populate the "tableSelect" dropdown with available table IDs
 */
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
    option.textContent = tid;
    tableSelect.appendChild(option);
  });

  tableSelect.removeEventListener("change", onTableSelectChange);
  tableSelect.addEventListener("change", onTableSelectChange);

  // Select the first one automatically
  tableSelect.value = tableIds[0];
  onTableSelectChange();
}

function onTableSelectChange() {
  const tableSelect = document.getElementById("tableSelect");
  const selectedTid = tableSelect.value;
  showClaimsForTable(selectedTid);
}

/**
 * Show claims for the chosen table
 */
function showClaimsForTable(tableId) {
  const claimListDiv = document.getElementById("claimList");
  claimListDiv.innerHTML = "";

  const container = document.getElementById("table-container");
  container.innerHTML = "";

  const itemsForTable = tableIdToResultsMap[tableId] || [];
  itemsForTable.forEach((res, idx) => {
    const div = document.createElement("div");
    div.className = "claim-item";
    div.textContent = `Claim #${idx + 1}: ${res.claim}`;

    // On click, highlight and render
    div.addEventListener("click", () => {
      document.querySelectorAll(".claim-item").forEach(item => {
        item.classList.remove("selected");
      });
      div.classList.add("selected");
      renderClaimAndTable(res);
    });

    claimListDiv.appendChild(div);
  });

  // Auto-click the first claim (if any)
  if (itemsForTable.length > 0) {
    claimListDiv.firstChild.click();
  }
}

/**
 * Render the chosen claim + the table with highlighted cells
 */
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

  // Fetch the CSV
  const csvFileName = resultObj.table_id; // e.g. "table_123.csv"
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
  if (!tableData.length) {
    const msg = document.createElement("p");
    msg.textContent = "Table is empty or could not be parsed.";
    container.appendChild(msg);
    return;
  }

  const columns = tableData[0];
  const dataRows = tableData.slice(1);

  // Build HTML table
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
  dataRows.forEach((rowVals, rowIndex) => {
    const tr = document.createElement("tr");
    rowVals.forEach((cellVal, colIndex) => {
      const td = document.createElement("td");
      td.textContent = cellVal;

      // Highlight if listed
      const columnName = columns[colIndex];
      const shouldHighlight = resultObj.highlighted_cells.some(
        hc => hc.row_index === rowIndex &&
              hc.column_name.toLowerCase() === columnName.toLowerCase()
      );
      if (shouldHighlight) {
        td.classList.add("highlight");
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  tableEl.appendChild(tbody);

  container.appendChild(tableEl);
}

// ---------------------
// LIVE CHECK LOGIC
// ---------------------

/**
 * Validate if both Table and Claim inputs are non-empty.
 * Enables or disables the "Run Live Check" button accordingly.
 */
function validateLiveCheckInputs() {
  const tableInput = document.getElementById("inputTable").value.trim();
  const claimInput = document.getElementById("inputClaim").value.trim();
  const runLiveCheckBtn = document.getElementById("runLiveCheck");

  if (tableInput && claimInput) {
    runLiveCheckBtn.disabled = false;
    runLiveCheckBtn.style.opacity = "1";
    runLiveCheckBtn.style.cursor = "pointer";
  } else {
    runLiveCheckBtn.disabled = true;
    runLiveCheckBtn.style.opacity = "0.6";
    runLiveCheckBtn.style.cursor = "not-allowed";
  }
}


function setupLiveCheckEvents() {
  // 1) "Load Model" button
  const loadModelBtn = document.getElementById("loadLiveModel");
  loadModelBtn.addEventListener("click", async () => {
    const modelId = document.getElementById("liveModelSelect").value;
    await initLivePipeline(modelId);
  });

  // 2) Whenever user edits the CSV, re-render the preview and validate inputs
  const inputTableEl = document.getElementById("inputTable");
  const inputClaimEl = document.getElementById("inputClaim");
  const runLiveCheckBtn = document.getElementById("runLiveCheck");

  inputTableEl.addEventListener("input", () => {
    const csvText = inputTableEl.value;
    renderLivePreviewTable(csvText, []);
    validateLiveCheckInputs();
  });

  // 3) Whenever user edits the Claim, validate inputs
  inputClaimEl.addEventListener("input", () => {
    validateLiveCheckInputs();
  });

  // 4) "Run Live Check" button
  runLiveCheckBtn.addEventListener("click", async () => {
    if (!deepSeekPipeline) {
      console.warn("No pipeline loaded. Please load a model first.");
      modelLoadingStatusEl.textContent = "Model not ready. Please load a model first.";
      return;
    }

    // Read user inputs
    const tableInput = inputTableEl.value.trim();
    const claimInput = inputClaimEl.value.trim();

    // Additional validation to ensure inputs are not empty
    if (!tableInput || !claimInput) {
      alert("Please provide both a table and a claim before running the live check.");
      return;
    }

    // Convert CSV to Markdown table for prompt
    const tablePrompt = csvToMarkdown(tableInput);

    // Construct user prompt
    const userPrompt = `
You are tasked with determining whether a claim about the following table (in Markdown) is TRUE or FALSE.

Table (Markdown):
${tablePrompt}

Claim: "${claimInput}"

Instructions:
- Carefully check each condition in the claim against the table and determine which cells are relevant to the claim. These are the "highlighted_cells".
- If fully supported, the 'answer' should be "TRUE". Otherwise "FALSE".
- Return only a valid JSON object with two keys:
"answer": must be "TRUE" or "FALSE" (all caps)
"highlighted_cells": a list of objects, each with "row_index" (int) and "column_name" (string)

For example:

{{
  "answer": "TRUE",
  "highlighted_cells": [
    {{"row_index": 0, "column_name": "Revenue"}},
    {{"row_index": 1, "column_name": "Employees"}}
  ]
}}

No extra keys, no extra text. Just that JSON.
`.trim();

    // Clear any old streaming text
    liveStreamOutputEl.textContent = "";

    // Prepare a TextStreamer for partial tokens
    const { TextStreamer } = window._transformers;
    const streamer = new TextStreamer(deepSeekPipeline.tokenizer, {
      skip_prompt: true,
      callback_function: (token) => {
        liveStreamOutputEl.textContent += token;
      }
    });

    let result;
    try {
      // Chat format
      const messages = [{ role: "user", content: userPrompt }];
      result = await deepSeekPipeline(messages, {
        max_new_tokens: 2048,
        do_sample: false,
        streamer
      });
    } catch (err) {
      console.error("Error running pipeline:", err);
      liveStreamOutputEl.textContent += `\n\n[Error: ${err}]`;
      return;
    }

    // Extract final text
    const rawResponse = (Array.isArray(result) && result.length > 0)
      ? result[0].generated_text?.at(-1)?.content || ""
      : "";

    // Attempt to parse the JSON robustly
    const parsed = extractJsonFromResponse(rawResponse);

    const finalAnswer = parsed.answer || "UNKNOWN";
    const highlightedCells = parsed.highlighted_cells || [];

    // Update UI
    displayLiveResults(tableInput, claimInput, finalAnswer, highlightedCells);
  });
   // Initially disable the "Run Live Check" button
  runLiveCheckBtn.disabled = true;
  runLiveCheckBtn.style.opacity = "0.6";
  runLiveCheckBtn.style.cursor = "not-allowed";
}



/**
 * Initialize (or re-initialize) the pipeline with a selected model
 */
async function initLivePipeline(modelId) {
  // Clear any old pipeline reference
  deepSeekPipeline = null;

  // Show user
  modelLoadingStatusEl.textContent = `Loading model: ${modelId} `;
  modelLoadingStatusEl.innerHTML += `<span class="spinner"></span>`;

  console.log(`Initializing pipeline with: ${modelId}`);
  const { pipeline } = window._transformers;

  try {
    // Try WebGPU + half-precision
    let generator;
    try {
      generator = await pipeline("text-generation", modelId, {
        dtype: "q4f16",
        device: "webgpu"
      });
    } catch (gpuErr) {
      console.warn("GPU init failed, falling back to CPU...", gpuErr);
      generator = await pipeline("text-generation", modelId, {
        backend: "wasm",
        // or dtype: 'float32'
      });
    }

    deepSeekPipeline = generator;
    modelLoadingStatusEl.textContent = `Model loaded: ${modelId}`;
    // Change colour to green
    modelLoadingStatusEl.style.color = "green";
  } catch (err) {
    console.error("Failed to init pipeline:", err);
    modelLoadingStatusEl.textContent = `Failed to load model: ${err}`;
  }
}

/**
 * Render the CSV as an HTML table (live preview).
 * `highlightedCells` is an array of {row_index, column_name} if we want to highlight cells.
 */
function renderLivePreviewTable(csvText, highlightedCells) {
  const previewContainer = document.getElementById("livePreviewTable");
  previewContainer.innerHTML = ""; // Clear old table

  const lines = csvText.split(/\r?\n/).filter(line => line.trim().length > 0);
  if (!lines.length) {
    return; // No data to show
  }

  // Convert lines into array-of-arrays
  const tableData = lines.map(line => line.split("#"));
  if (!tableData.length) return;

  const columns = tableData[0];
  const dataRows = tableData.slice(1);

  // Build table
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
  dataRows.forEach((rowVals, rowIndex) => {
    const tr = document.createElement("tr");
    rowVals.forEach((cellVal, colIndex) => {
      const td = document.createElement("td");
      td.textContent = cellVal;

      // Check highlight
      const colName = columns[colIndex];
      const shouldHighlight = highlightedCells.some(
        hc => hc.row_index === rowIndex &&
              hc.column_name?.toLowerCase() === colName.toLowerCase()
      );
      if (shouldHighlight) {
        td.classList.add("highlight");
      }

      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  tableEl.appendChild(tbody);

  previewContainer.appendChild(tableEl);
}

/**
 * Display final claim + result + highlight cells in the existing live preview table
 */
function displayLiveResults(csvText, claim, answer, highlightedCells) {
  // 1) Show the claim + model's final answer
  const liveClaimList = document.getElementById("liveClaimList");
  liveClaimList.innerHTML = ""; // Clear old
  const claimDiv = document.createElement("div");
  claimDiv.className = "claim-item selected";
  claimDiv.textContent = `Claim: "${claim}" => Model says: ${answer}`;
  liveClaimList.appendChild(claimDiv);

  // 2) Re-render the same table with highlights
  renderLivePreviewTable(csvText, highlightedCells);
}

/**
 * Convert CSV (#-delimited) to a Markdown table for the prompt.
 */
function csvToMarkdown(csvStr) {
  const lines = csvStr.trim().split(/\r?\n/);
  if (!lines.length) return "";

  const tableData = lines.map(line => line.split("#"));
  if (!tableData.length) return "";

  const headers = tableData[0];
  const rows = tableData.slice(1);

  // Build Markdown
  let md = `| ${headers.join(" | ")} |\n`;
  md += `| ${headers.map(() => "---").join(" | ")} |\n`;
  rows.forEach(row => {
    md += `| ${row.join(" | ")} |\n`;
  });

  return md;
}

/**
 * Attempt to parse JSON from the model's raw output.
 * 1) Look for ```json code fence
 * 2) If that fails, try to parse entire rawResponse
 * 3) Otherwise return {}
 */
function extractJsonFromResponse(rawResponse) {
  // 1) Regex for code fence: ```json ... ```
  const fencePattern = /```json\s*([\s\S]*?)\s*```/i;
  const fenceMatch = rawResponse.match(fencePattern);
  if (fenceMatch) {
    const jsonText = fenceMatch[1].trim();
    try {
      return JSON.parse(jsonText);
    } catch (jsonErr) {
      console.warn("[extractJsonFromResponse] Could not parse code-fenced JSON:", jsonErr);
    }
  }

  // 2) Try parsing entire response
  try {
    return JSON.parse(rawResponse);
  } catch (err) {
    console.warn("[extractJsonFromResponse] Could not parse entire response as JSON:", err);
  }

  // 3) Fallback
  return {};
}