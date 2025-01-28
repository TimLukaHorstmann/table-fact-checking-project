// Location of your results JSON
const RESULTS_JSON_PATH = "../Prompt_Engineering/results_20250128/results_with_cells_mistral_test_set_2_zero_shot_naturalized.json"; 

// Base URL or relative path to your CSV folder
// If you're hosting on GitHub Pages, you might need the full raw URL like:
//   https://raw.githubusercontent.com/YourUser/YourRepo/main/data/all_csv/
// or if you have the CSV files in the same domain, a relative path could be:
const CSV_BASE_PATH = "../original_repo/data/all_csv/"; 

let allResults = [];
let tableIdToResultsMap = {};

async function loadResultsJSON() {
  const response = await fetch(RESULTS_JSON_PATH);
  allResults = await response.json();

  // Build a map: table_id -> array of results
  tableIdToResultsMap = allResults.reduce((acc, item) => {
    const tId = item.table_id;
    if (!acc[tId]) {
      acc[tId] = [];
    }
    acc[tId].push(item);
    return acc;
  }, {});
}

function populateTableSelect() {
  const tableSelect = document.getElementById("tableSelect");
  const tableIds = Object.keys(tableIdToResultsMap);

  tableIds.forEach((tid) => {
    const opt = document.createElement("option");
    opt.value = tid;
    opt.textContent = tid;
    tableSelect.appendChild(opt);
  });

  tableSelect.addEventListener("change", () => {
    const selected = tableSelect.value;
    showClaimsForTable(selected);
  });

  // Auto-select the first table if present
  if (tableIds.length > 0) {
    tableSelect.value = tableIds[0];
    showClaimsForTable(tableIds[0]);
  }
}

function showClaimsForTable(tableId) {
  const claimListDiv = document.getElementById("claimList");
  claimListDiv.innerHTML = "";

  const resultsForTable = tableIdToResultsMap[tableId] || [];

  resultsForTable.forEach((res, idx) => {
    const div = document.createElement("div");
    div.className = "claim-item";
    div.textContent = `Claim #${idx+1}: ${res.claim}`;
    div.addEventListener("click", () => {
      // On click, fetch the CSV, parse it, then render & highlight
      renderClaimAndTable(res);
    });
    claimListDiv.appendChild(div);
  });

  // Optionally auto-show the first claim
  if (resultsForTable.length > 0) {
    renderClaimAndTable(resultsForTable[0]);
  }
}

async function renderClaimAndTable(resultObj) {
  const container = document.getElementById("table-container");
  container.innerHTML = "";

  // Show basic info
  const infoDiv = document.createElement("div");
  infoDiv.innerHTML = `
    <p><b>Claim:</b> ${resultObj.claim}</p>
    <p><b>Predicted Label:</b> ${resultObj.predicted_response ? "TRUE" : "FALSE"}</p>
    <p><b>Model Raw Output:</b> ${resultObj.resp}</p>
    <p><b>Ground Truth:</b> ${resultObj.true_response ? "TRUE" : "FALSE"}</p>
  `;
  container.appendChild(infoDiv);

  // Next, fetch the CSV from GitHub (or local) using the table_id
  const csvFileName = resultObj.table_id;  // e.g. "example_table.csv"
  const csvUrl = CSV_BASE_PATH + csvFileName;

  // fetch the CSV
  let csvText = "";
  try {
    const resp = await fetch(csvUrl);
    csvText = await resp.text();
  } catch (err) {
    const errMsg = document.createElement("p");
    errMsg.textContent = `Failed to load CSV from ${csvUrl}: ${err}`;
    container.appendChild(errMsg);
    return;
  }

  // parse CSV with '#' delimiter
  // a simple parser, ignoring quotes, etc.
  const lines = csvText.split(/\r?\n/);
  const tableData = lines.map(line => line.split("#"));

  if (!tableData || tableData.length === 0) {
    const msg = document.createElement("p");
    msg.textContent = "Table is empty or couldn't parse properly.";
    container.appendChild(msg);
    return;
  }

  // We'll assume tableData[0] is the header row:
  const columns = tableData[0];
  const dataRows = tableData.slice(1);

  // Create HTML table
  const tableEl = document.createElement("table");

  // thead
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  columns.forEach(colName => {
    const th = document.createElement("th");
    th.textContent = colName;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  tableEl.appendChild(thead);

  // tbody
  const tbody = document.createElement("tbody");
  dataRows.forEach((rowVals, rowIndex) => {
    const tr = document.createElement("tr");
    rowVals.forEach((cellVal, colIndex) => {
      const td = document.createElement("td");
      td.textContent = cellVal;

      // Check if this cell is highlighted
      // The model's row_index is based on data rows, so if model says row_index=0,
      // that corresponds to dataRows[0], which is lines[1] in CSV, etc.
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

(async function init() {
  await loadResultsJSON();
  populateTableSelect();
})();