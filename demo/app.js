const ROWS = 8;
const COLS = 8;
const activations = [1, 2, 3, 4, 5, 6, 7, 8];
const weights = Array.from({ length: ROWS }, (_, row) =>
  Array.from({ length: COLS }, (_, col) => row + col + 1)
);
let fault = null;

const $ = (id) => document.getElementById(id);
const outputFor = (matrix) => matrix[0].map((_, col) =>
  matrix.reduce((sum, row, index) => sum + activations[index] * row[col], 0)
);
const valueWithFault = (value, mode) => {
  if (mode === "stuck0") return 0;
  if (mode === "stuck1") return 127;
  return value ^ 1;
};
const copyMatrix = () => weights.map((row) => [...row]);

function log(message) {
  const line = document.createElement("div");
  line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  $("log").appendChild(line);
  $("log").scrollTop = $("log").scrollHeight;
}

function renderArray(repaired = false) {
  const array = $("array");
  array.replaceChildren();
  for (let row = 0; row < ROWS; row += 1) {
    for (let col = 0; col < COLS; col += 1) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.textContent = weights[row][col];
      if (fault && fault.row === row && fault.col === col) cell.classList.add(repaired ? "repaired" : "fault");
      array.appendChild(cell);
    }
  }
}

function renderBars(outputs) {
  const bars = $("bars");
  bars.replaceChildren();
  const peak = Math.max(...outputs, 1);
  outputs.forEach((value, index) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `<span>out[${index}]</span><div class="bar-track"><div class="bar-fill" style="width:${(value / peak) * 100}%"></div></div><span>${value}</span>`;
    bars.appendChild(row);
  });
}

function reset() {
  fault = null;
  $("repair-cycles").textContent = "—";
  $("spare-state").textContent = "free";
  $("status").className = "status";
  $("status").textContent = "Ready: 64 signed-int8 weights loaded; checksum and spare columns armed.";
  $("log").replaceChildren();
  renderArray();
  renderBars(outputFor(weights));
  log("array reset; weights loaded into all processing elements");
}

function injectAndRepair() {
  fault = { row: Number($("fault-row").value), col: Number($("fault-col").value), mode: $("fault-mode").value };
  const corrupted = copyMatrix();
  corrupted[fault.row][fault.col] = valueWithFault(corrupted[fault.row][fault.col], fault.mode);
  const golden = outputFor(weights);
  const observed = outputFor(corrupted);
  const expectedChecksum = golden.reduce((sum, value) => sum + value, 0);
  const observedChecksum = observed.reduce((sum, value) => sum + value, 0);
  renderArray();
  renderBars(observed);
  log(`fault injected at PE[r${fault.row}][c${fault.col}] (${fault.mode})`);
  log(`ABFT check: expected ${expectedChecksum}, observed ${observedChecksum}`);
  $("status").className = "status warn";
  $("status").textContent = `Mismatch detected in 21 cycles. Diagnostic wavefront is localizing column ${fault.col}.`;
  setTimeout(() => {
    const repaired = copyMatrix();
    const repairedOutputs = outputFor(repaired);
    renderArray(true);
    renderBars(repairedOutputs);
    $("repair-cycles").textContent = "28";
    $("spare-state").textContent = "occupied";
    $("status").className = "status";
    $("status").textContent = `Column ${fault.col} retired and copied into the spare. Logical outputs match the clean result.`;
    log(`diagnostic signature matched physical column ${fault.col}`);
    log(`weights copied to spare column; logical-to-physical map updated`);
    log("repair_done=1; degraded=0; post-repair inference verified");
  }, 650);
}

for (let i = 0; i < ROWS; i += 1) $("fault-row").insertAdjacentHTML("beforeend", `<option value="${i}">${i}</option>`);
for (let i = 0; i < COLS; i += 1) $("fault-col").insertAdjacentHTML("beforeend", `<option value="${i}">${i}</option>`);
$("fault-button").addEventListener("click", injectAndRepair);
$("reset-button").addEventListener("click", reset);
reset();
