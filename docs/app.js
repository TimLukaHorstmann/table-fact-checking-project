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
let globalAbortController = null;

// We'll store the pipeline object here for the live check
let deepSeekPipeline = null;

// Some quick references
const modelLoadingStatusEl = document.getElementById("modelLoadingStatus");
const liveThinkOutputEl = document.getElementById("liveThinkOutput"); 
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
    populateExistingTableDropdown();
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
      const shouldHighlight = resultObj.relevant_cells.some(
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






/////////////////////////////////////////////////////////////////////////////////////




// ---------------------
// LIVE CHECK LOGIC
// ---------------------


async function populateExistingTableDropdown() {
  const existingTableSelect = document.getElementById("existingTableSelect");
  existingTableSelect.innerHTML = `<option value="">-- Select a Table --</option>`;

  try {
    // Fetch all CSV IDs from docs/all_csv_ids.json
    const response = await fetch("https://raw.githubusercontent.com/wenhuchen/Table-Fact-Checking/refs/heads/master/data/all_csv_ids.json");
    if (!response.ok) throw new Error(`Failed to fetch all_csv_ids.json: ${response.statusText}`);

    const csvIds = await response.json();
    if (!csvIds || !Array.isArray(csvIds)) throw new Error("Invalid format for all_csv_ids.json.");

    // Populate dropdown with CSVs
    csvIds.sort().forEach(csvFile => {
      const option = document.createElement("option");
      option.value = csvFile;
      option.textContent = csvFile;
      existingTableSelect.appendChild(option);
    });

    // Add event listener to fetch table when selected
    existingTableSelect.addEventListener("change", async () => {
      const selectedFile = existingTableSelect.value;
      if (!selectedFile) return;
      await fetchAndFillTable(selectedFile);
    });

  } catch (error) {
    console.error("Error loading CSV list:", error);
    alert("Failed to fetch available tables. Please try again later.");
  }
}


async function fetchAndFillTable(tableId) {
  const inputTableEl = document.getElementById("inputTable");
  const previewContainer = document.getElementById("livePreviewTable");

  inputTableEl.value = "";  // Clear previous input
  previewContainer.innerHTML = "";  // Clear previous preview

  const csvUrl = CSV_BASE_PATH + tableId;

  try {
    const response = await fetch(csvUrl);
    if (!response.ok) throw new Error(`Failed to fetch CSV: ${response.statusText}`);

    const csvText = await response.text();
    inputTableEl.value = csvText;  // Fill the textarea
    renderLivePreviewTable(csvText, []);  // Update live preview
    validateLiveCheckInputs(); // Ensure "Run Check" button is enabled
  } catch (error) {
    console.error("Error loading table CSV:", error);
    alert("Failed to load table from dataset.");
  }
}



/**
 * Validate if both Table and Claim inputs are non-empty.
 * Enables or disables the "Run Live Check" button accordingly.
 */
function validateLiveCheckInputs() {
  const tableInput = document.getElementById("inputTable").value.trim();
  const claimInput = document.getElementById("inputClaim").value.trim();
  const runLiveCheckBtn = document.getElementById("runLiveCheck");
  const stopLiveCheckBtn = document.getElementById("stopLiveCheck");

  if (tableInput && claimInput) {
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
  
    // Show the Stop icon
    document.getElementById("stopLiveCheck").style.display = "inline-flex";
  
    // Build userPrompt, etc. (same as before) ...
    const tableInput = inputTableEl.value.trim();
    const claimInput = inputClaimEl.value.trim();
    if (!tableInput || !claimInput) {
      alert("Please provide both a table and a claim before running the live check.");
      return;
    }
    const tablePrompt = csvToMarkdown(tableInput);
    const userPrompt = `
  You are checking if a claim about the following table is TRUE or FALSE ("answer") and which cells support it ("relevant_cells").
  Output ONLY valid JSON with keys "answer" ("TRUE" or "FALSE") and "relevant_cells" (list of {row_index, column_name} pairs, where the column_name must match EXACTLY one of the coumn names in the table).

  Markdown table:
  ${tablePrompt}
  
  Claim: "${claimInput}"
  `.trim();
    
  // Clear old output
    liveThinkOutputEl.textContent = "";
    liveThinkOutputEl.style.display = "none";
    liveStreamOutputEl.textContent = "";
  
    // Setup AbortController
    globalAbortController = new AbortController();
    const signal = globalAbortController.signal;
  
    // Check if we are using DeepSeek:
    const selectedModelId = document.getElementById("liveModelSelect").value;
    const isDeepSeek = selectedModelId.includes("DeepSeek");
  
    // We'll keep a buffer for partial tokens, so we can handle <think> or </think> crossing chunk boundaries
    let buffer = "";           
    let inThinkBlock = false;  
    let finalText = "";        // everything outside <think>...</think>
    let thinkText = "";        // everything inside <think>...</think>
  
    // 2) Create a custom callback
    const customCallback = (token) => {
      // If user clicked "STOP"
      if (signal.aborted) {
        throw new Error("User aborted generation.");
      }
      // Accumulate new chunk
      buffer += token;
  
      // We'll look for <think> or </think> in 'buffer'
      // and move text to the correct place.
      while (true) {
        // If we are NOT currently in the <think> block, check if <think> is found
        if (!inThinkBlock) {
          const startIdx = buffer.indexOf("<think>");
          if (startIdx === -1) {
            // No <think>, so all text belongs to final
            finalText += buffer;
            buffer = "";
            break;
          } else {
            // Everything before <think> belongs to final
            finalText += buffer.slice(0, startIdx);
            // Now we enter the think block
            inThinkBlock = true;
            // Remove that portion + <think> from the buffer
            buffer = buffer.slice(startIdx + "<think>".length);
          }
        } 
        // If we ARE in the <think> block, look for </think>
        else {
          const endIdx = buffer.indexOf("</think>");
          if (endIdx === -1) {
            // All text is inside the think block for now
            thinkText += buffer;
            buffer = "";
            break;
          } else {
            // Everything up to </think> is in the think block
            thinkText += buffer.slice(0, endIdx);
            // We exit the think block
            inThinkBlock = false;
            // Remove that portion + </think> from buffer
            buffer = buffer.slice(endIdx + "</think>".length);
            // And continue the while loop to see if there's more <think> in buffer
          }
        }
      }
  
      // Update the UI (in real-time)
      // Update UI elements
      if (isDeepSeek) {
        // Show thinking area if there's any text
        if (thinkText.trim().length > 0) {
          liveThinkOutputEl.style.display = "block";
          const thinkContentMarkdown = marked.parse(thinkText.trim());
          const safeThinkContent = DOMPurify.sanitize(thinkContentMarkdown);
          liveThinkOutputEl.innerHTML = `
            <div class="thinking-overlay">
              <span id="thinkingLabel" class="thinking-label">Thinking...</span>
            </div>
            <div id="thinkContent">${safeThinkContent}</div>
          `;
        }

        // Update main output (outside <think>)
        const answerContentMarkdown = marked.parse(finalText);
        const safeAnswerContent = DOMPurify.sanitize(answerContentMarkdown);
        liveStreamOutputEl.innerHTML = `
          <div class="answer-overlay">Answer</div>
          <div id="answerContent">${safeAnswerContent}</div>
        `;
      } else {
        // For non-deepseek models, treat all output as final
        finalText += token;
        const answerContentMarkdown = marked.parse(finalText);
        const safeAnswerContent = DOMPurify.sanitize(answerContentMarkdown);
        liveStreamOutputEl.innerHTML = `
          <div class="answer-overlay">Answer</div>
          <div id="answerContent">${safeAnswerContent}</div>
        `;
      }
      // After updating innerHTML
      document.querySelectorAll('#thinkContent pre code, #answerContent pre code').forEach((block) => {
        hljs.highlightElement(block);
      });
    };
  
    // 3) Construct the streamer using our custom callback
    const { TextStreamer } = window._transformers;
    const streamer = new TextStreamer(deepSeekPipeline.tokenizer, {
      skip_prompt: true,
      callback_function: customCallback
    });
  
    // 4) Invoke the pipeline
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
      // Hide stop icon
      document.getElementById("stopLiveCheck").style.display = "none";
  
      if (err.name === "AbortError" || err.message.includes("aborted")) {
        console.warn("Generation was aborted.");
        liveStreamOutputEl.textContent += "\n\n[Generation Aborted]";
        return;
      }
      console.error("Error running pipeline:", err);
      liveStreamOutputEl.textContent += `\n\n[Error: ${err}]`;
      return;
    }
  
    // 5) Done generating => hide stop icon
    document.getElementById("stopLiveCheck").style.display = "none";
  
    // Because the model might have final leftover text in 'buffer'
    // that we haven't assigned yet, handle that:
    if (buffer.length > 0) {
      // If we were inThinkBlock, add it to thinkText
      if (inThinkBlock) {
        thinkText += buffer;
      } else {
        finalText += buffer;
      }
    }
  
    // Finally, the text outside <think> is in finalText
    // We'll treat that as the "rawResponse" to parse for JSON
    // (The user specifically wants the JSON part AFTER the think block)
    const rawResponse = finalText.trim();
  
    // Attempt to parse JSON
    const parsed = extractJsonFromResponse(rawResponse);
    const finalAnswer = parsed.answer || "UNKNOWN";
    const relevantCells = parsed.relevant_cells || [];
  
    // Show final answer & highlight table
    displayLiveResults(tableInput, claimInput, finalAnswer, relevantCells);
  });
   // Initially disable the "Run Live Check" button
  runLiveCheckBtn.disabled = true;
  runLiveCheckBtn.style.opacity = "0.6";
  runLiveCheckBtn.style.cursor = "not-allowed";

  const stopLiveCheckBtn = document.getElementById("stopLiveCheck");
  stopLiveCheckBtn.addEventListener("click", () => {
    if (globalAbortController) {
      globalAbortController.abort();
      console.log("Generation aborted by user.");
      modelLoadingStatusEl.textContent = "Generation manually stopped.";
    }
  });
  stopLiveCheckBtn.disabled = true;
  stopLiveCheckBtn.style.opacity = "0.6";
  stopLiveCheckBtn.style.cursor = "not-allowed";

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
 * `relevantCells` is an array of {row_index, column_name} if we want to highlight cells.
 */
function renderLivePreviewTable(csvText, relevantCells) {
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
      const shouldHighlight = relevantCells.some(
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
function displayLiveResults(csvText, claim, answer, relevantCells) {
  // 1) Show the claim + model's final answer
  const liveClaimList = document.getElementById("liveClaimList");
  liveClaimList.innerHTML = ""; // Clear old
  const claimDiv = document.createElement("div");
  claimDiv.className = "claim-item selected";
  claimDiv.textContent = `Claim: "${claim}" => Model says: ${answer}`;
  liveClaimList.appendChild(claimDiv);

  // 2) Re-render the same table with highlights
  renderLivePreviewTable(csvText, relevantCells);
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

function extractJsonFromResponse(rawResponse) {
  let jsonText = rawResponse.trim();

  // 1) Extract JSON inside ```json ... ```
  const fencePattern = /```json\s*([\s\S]*?)\s*```/i;
  const fenceMatch = jsonText.match(fencePattern);
  if (fenceMatch) {
    jsonText = fenceMatch[1].trim();
  }

  // 2) Fix common formatting errors

  // - Replace {row_index, "column_name"} with {"row_index": row_index, "column_name": "column_name"}
  // Updated Regex to allow numeric row_index
  jsonText = jsonText.replace(
    /\{(\d+|[^"\s]+),\s*"([^"]+)"\}/g,
    '{"row_index": "$1", "column_name": "$2"}'
  );

  // - Ensure all keys are properly quoted
  jsonText = jsonText.replace(
    /(\{|,)\s*([\w\d_]+)\s*:/g,
    '$1 "$2":'
  );

  // - Fix unquoted values like TRUE/FALSE → "TRUE"/"FALSE"
  jsonText = jsonText.replace(
    /:\s*(TRUE|FALSE)([\s,\}])/gi,
    ': "$1"$2'
  );

  // - Handle single quotes (change to double quotes)
  jsonText = jsonText.replace(/'/g, '"');

  // 3) Attempt to parse JSON
  try {
    return JSON.parse(jsonText);
  } catch (err) {
    console.warn("[extractJsonFromResponse] Could not parse JSON:", err);
    // Print raw for debugging
    console.log("Attempted JSON Text:", jsonText);
    return {}; // Return empty object as a fallback
  }
}




function separateThinkFromResponse(rawText) {
  // Regex: capture everything between <think> and </think>
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
