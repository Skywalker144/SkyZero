const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const statusTextEl = document.getElementById("status-text");
const statusProgressEl = document.getElementById("status-progress");
const statusEtaEl = document.getElementById("status-eta");
const statusFootEl = document.getElementById("status-foot");
const statusProgressBarEl = document.getElementById("status-progress-bar");
const undoBtn = document.getElementById("undo-btn");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingTextEl = document.getElementById("loading-text");
const loadingProgressBarEl = document.getElementById("loading-progress-bar");
const loadingProgressTextEl = document.getElementById("loading-progress-text");

const boardSize = 15;
let cellSize = 0;
let margin = 0;

let game = new Gomoku(boardSize, 2, true);
let state = game.getInitialState();
let toPlay = 1; // 1 for Black, -1 for White
let playerColor = 1; // 1 for Black, -1 for White
let history = [];
let aiRunning = false;
let lastMove = null; // 记录上一手棋的位置 {r, c}
let numSimulations = 200;
const undoLimit = 3;
let undoCount = 0;

// Heatmap data
let lastResults = {
    policy: null
};

const worker = new Worker("worker.js");
const chartCanvas = document.getElementById("win-prob-chart");
const chartCtx = chartCanvas.getContext("2d");
let winProbHistory = []; // Start empty
let showHeatmap = true;
let showForbidden = true;

let searchStartTime = 0;
let lastSearchDuration = null;
let lastMoveFromAI = false;

const simSlider = document.getElementById("sim-slider");
const simValueEl = document.getElementById("sim-value");

function updateSimulations(value) {
    const parsed = parseInt(value, 10);
    numSimulations = Number.isFinite(parsed) ? parsed : numSimulations;
    if (simValueEl) simValueEl.textContent = numSimulations;
}

if (simSlider) {
    updateSimulations(simSlider.value);
    simSlider.addEventListener("input", () => {
        updateSimulations(simSlider.value);
    });
}

function updateCanvasSize() {
    const dpr = window.devicePixelRatio || 1;
    // Logical size: base it on container width but cap at 960
    const containerWidth = canvas.parentElement.clientWidth;
    const logicalSize = Math.min(960, containerWidth);
    
    // Set physical size
    canvas.width = logicalSize * dpr;
    canvas.height = logicalSize * dpr;
    
    // Set display size
    canvas.style.width = logicalSize + "px";
    canvas.style.height = logicalSize + "px";
    
    // Scale context for all subsequent drawing
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    
    // Update board constants based on logical size
    const marginRatio = 0.7;
    cellSize = logicalSize / (boardSize - 1 + 2 * marginRatio);
    margin = cellSize * marginRatio;
    
    drawBoard();
}

function updateChartSize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = chartCanvas.getBoundingClientRect();
    chartCanvas.width = rect.width * dpr;
    chartCanvas.height = rect.height * dpr;
    
    // Also scale chart context
    chartCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    
    drawWinProbChart();
}

window.addEventListener("resize", () => {
    updateCanvasSize();
    updateChartSize();
});

// Initial size setup
setTimeout(() => {
    updateCanvasSize();
    updateChartSize();
    updateSlider();
}, 100);

worker.postMessage({ type: "init" });

worker.onmessage = function(e) {
    const data = e.data;
    if (data.type === "ready") {
        loadingOverlay.style.display = "none";
        resetGame();
    } else if (data.type === "error") {
        loadingOverlay.innerHTML = `<p style="color:red">Error: ${data.message}</p>`;
        console.error("Worker Error:", data.message);
    } else if (data.type === "model-progress") {
        updateLoadingProgress(data);
    } else if (data.type === "progress") {
        if (data.searchId === searchId) {
            updateSearchProgress(data.progress);
        }
    } else if (data.type === "result") {
        handleAIResult(data);
    }
};

function updateLoadingProgress({ percent, loaded, total }) {
    if (!loadingProgressBarEl || !loadingProgressTextEl) return;
    if (Number.isFinite(percent)) {
        const bounded = Math.min(100, Math.max(0, percent));
        loadingProgressBarEl.style.width = `${bounded}%`;
        loadingProgressTextEl.textContent = `${Math.round(bounded)}%`;
        if (loadingTextEl) loadingTextEl.textContent = "权重下载中";
        return;
    }

    if (Number.isFinite(loaded) && Number.isFinite(total) && total > 0) {
        const computed = (loaded / total) * 100;
        const bounded = Math.min(100, Math.max(0, computed));
        loadingProgressBarEl.style.width = `${bounded}%`;
        loadingProgressTextEl.textContent = `${Math.round(bounded)}%`;
        if (loadingTextEl) loadingTextEl.textContent = "权重下载中";
        return;
    }

    loadingProgressBarEl.style.width = "0%";
    loadingProgressTextEl.textContent = "--%";
    if (loadingTextEl) loadingTextEl.textContent = "权重下载中";
}

function setStatusText(text) {
    if (statusTextEl) {
        statusTextEl.textContent = text;
    } else if (statusEl) {
        statusEl.innerText = text;
    }
}

function updateUndoButton() {
    if (!undoBtn) return;
    const remaining = Math.max(0, undoLimit - undoCount);
    undoBtn.disabled = remaining === 0;
    if (remaining === undoLimit) {
        undoBtn.textContent = "悔棋";
    } else {
        undoBtn.textContent = `悔棋 (剩余${remaining})`;
    }
}

function updateStatusFoot() {
    if (!statusFootEl) return;
    if (lastSearchDuration === null) {
        statusFootEl.textContent = "";
        statusFootEl.style.opacity = "0";
        return;
    }
    statusFootEl.textContent = `思考耗时 ${lastSearchDuration.toFixed(1)} 秒`;
    statusFootEl.style.opacity = "1";
}

function clearStatusFoot() {
    lastSearchDuration = null;
    updateStatusFoot();
}

function setStatusProgress(progressPercent, etaSeconds) {
    if (!statusProgressEl || !statusEtaEl) return;
    if (!Number.isFinite(progressPercent)) {
        statusProgressEl.textContent = "--";
        statusEtaEl.textContent = "--";
        if (statusProgressBarEl) statusProgressBarEl.style.width = "0%";
        return;
    }

    const boundedProgress = Math.min(100, Math.max(0, progressPercent));
    statusProgressEl.textContent = `${Math.round(boundedProgress)}%`;
    if (statusProgressBarEl) statusProgressBarEl.style.width = `${boundedProgress}%`;
    if (Number.isFinite(etaSeconds)) {
        const boundedEta = Math.max(0, etaSeconds);
        statusEtaEl.textContent = `${boundedEta.toFixed(1)} 秒`;
    } else {
        statusEtaEl.textContent = "--";
    }
}

function updateSearchProgress(progressPercent) {
    if (!Number.isFinite(progressPercent)) return;
    const now = performance.now();
    if (!searchStartTime) searchStartTime = now;
    const elapsed = (now - searchStartTime) / 1000;
    const ratio = Math.min(1, Math.max(0, progressPercent / 100));
    const eta = ratio > 0 ? (elapsed * (1 - ratio)) / ratio : null;
    setStatusProgress(progressPercent, eta);
}

function startSearchStatus() {
    searchStartTime = performance.now();
    clearStatusFoot();
    setStatusText("SkyZero 思考中...");
    setStatusProgress(0, null);
}

function finishSearchStatus() {
    if (!searchStartTime) return;
    const elapsed = (performance.now() - searchStartTime) / 1000;
    lastSearchDuration = elapsed;
    setStatusProgress(100, 0);
    updateStatusFoot();
    searchStartTime = 0;
}

function setIdleStatus(text, clearFoot = false, clearProgress = false) {
    setStatusText(text);
    if (clearProgress) setStatusProgress(null, null);
    if (clearFoot) clearStatusFoot();
}

function drawBoard() {
    const dpr = window.devicePixelRatio || 1;
    const logicalSize = canvas.width / dpr;
    ctx.clearRect(0, 0, logicalSize, logicalSize);
    
    // Draw board background - warm wood tone for better contrast
    ctx.fillStyle = "#e8d4b8";
    ctx.fillRect(0, 0, logicalSize, logicalSize);
    
    // Draw grid lines
    ctx.strokeStyle = "#5a4a3a";
    ctx.lineWidth = 1;
    for (let i = 0; i < boardSize; i++) {
        // Vertical
        ctx.beginPath();
        ctx.moveTo(margin + i * cellSize, margin);
        ctx.lineTo(margin + i * cellSize, margin + (boardSize - 1) * cellSize);
        ctx.stroke();
        // Horizontal
        ctx.beginPath();
        ctx.moveTo(margin, margin + i * cellSize);
        ctx.lineTo(margin + (boardSize - 1) * cellSize, margin + i * cellSize);
        ctx.stroke();
    }

    // Draw outer boundary with thicker line
    ctx.lineWidth = 2.5;
    ctx.strokeRect(margin, margin, (boardSize - 1) * cellSize, (boardSize - 1) * cellSize);
    
    // Draw star points (hoshi)
    drawStarPoints();
    
    // Draw Heatmap
    drawHeatmap();
    
    // Draw Forbidden Points
    drawForbiddenPoints();
    
    // Draw stones
    const currentBoard = state[state.length - 1];
    for (let i = 0; i < currentBoard.length; i++) {
        if (currentBoard[i] !== 0) {
            const r = Math.floor(i / boardSize);
            const c = i % boardSize;
            drawStone(r, c, currentBoard[i]);
        }
    }
    
    // Draw last move marker
    if (lastMove) {
        drawLastMoveMarker(lastMove.r, lastMove.c);
    }
}

function drawStone(r, c, color) {
    const x = margin + c * cellSize;
    const y = margin + r * cellSize;
    const radius = cellSize * 0.44;
    
    if (color === 1) {
        // Black stone - 3D effect with highlight
        // Drop shadow
        ctx.beginPath();
        ctx.arc(x + 1, y + 1, radius, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0, 0, 0, 0.15)";
        ctx.fill();
        
        // Main body
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        const gradient = ctx.createRadialGradient(x - radius*0.3, y - radius*0.3, 0, x, y, radius);
        gradient.addColorStop(0, "#3a3a3a");
        gradient.addColorStop(0.5, "#2a2a2a");
        gradient.addColorStop(1, "#0a0a0a");
        ctx.fillStyle = gradient;
        ctx.fill();
        
        // Top highlight
        ctx.beginPath();
        ctx.arc(x - radius*0.25, y - radius*0.25, radius*0.35, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255, 255, 255, 0.08)";
        ctx.fill();
    } else {
        // White stone - enhanced visibility with proper contrast
        // Drop shadow
        ctx.beginPath();
        ctx.arc(x + 1, y + 1, radius, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0, 0, 0, 0.2)";
        ctx.fill();
        
        // Main body
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        const gradient = ctx.createRadialGradient(x - radius*0.2, y - radius*0.2, 0, x, y, radius);
        gradient.addColorStop(0, "#f8f8f8");
        gradient.addColorStop(0.6, "#f5f5f5");
        gradient.addColorStop(1, "#e5e5e5");
        ctx.fillStyle = gradient;
        ctx.fill();
        
        // Subtle edge definition
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(0, 0, 0, 0.08)";
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Inner highlight
        ctx.beginPath();
        ctx.arc(x - radius*0.25, y - radius*0.25, radius*0.3, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
        ctx.fill();
    }
}

function drawStarPoints() {
    const starPoints = [
        [3, 3], [3, 11], [7, 7], [11, 3], [11, 11]
    ];
    
    ctx.fillStyle = "#5a4a3a";
    for (const [r, c] of starPoints) {
        const x = margin + c * cellSize;
        const y = margin + r * cellSize;
        ctx.beginPath();
        ctx.arc(x, y, cellSize * 0.12, 0, Math.PI * 2);
        ctx.fill();
    }
}

function drawLastMoveMarker(r, c) {
    const x = margin + c * cellSize;
    const y = margin + r * cellSize;
    const size = cellSize * 0.25;
    
    ctx.fillStyle = "#e53935";
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + size, y);
    ctx.lineTo(x, y + size);
    ctx.closePath();
    ctx.fill();
}

function drawHeatmap() {
    if (!showHeatmap) return;
    let data = null;
    let colorScale = (v) => `rgba(255, 255, 255, ${v * 0.15})`;

    if (lastResults.policy) {
        data = lastResults.policy;
        const max = Math.max(...data);
        data = Array.from(data).map(v => v / (max || 1));
        colorScale = (v) => `rgba(0, 112, 243, ${v * 0.4})`;
    }

    if (!data) return;

    for (let i = 0; i < data.length; i++) {
        const r = Math.floor(i / boardSize);
        const c = i % boardSize;
        ctx.fillStyle = colorScale(data[i]);
        ctx.fillRect(margin + c * cellSize - cellSize / 2, margin + r * cellSize - cellSize / 2, cellSize, cellSize);
    }
}

function drawForbiddenPoints() {
    if (!showForbidden) return;
    
    const currentBoard = state[state.length - 1];
    game.fpf.clear();
    for (let i = 0; i < currentBoard.length; i++) {
        if (currentBoard[i] !== 0) {
            const r = Math.floor(i / boardSize);
            const c = i % boardSize;
            game.fpf.setStone(r, c, currentBoard[i] === 1 ? C_BLACK : C_WHITE);
        }
    }

    ctx.strokeStyle = "#e53935";
    ctx.lineWidth = 2;
    const size = cellSize * 0.2;

    for (let i = 0; i < currentBoard.length; i++) {
        if (currentBoard[i] === 0) {
            const r = Math.floor(i / boardSize);
            const c = i % boardSize;
            if (game.fpf.isForbidden(r, c)) {
                const x = margin + c * cellSize;
                const y = margin + r * cellSize;
                
                ctx.beginPath();
                ctx.moveTo(x - size, y - size);
                ctx.lineTo(x + size, y + size);
                ctx.stroke();
                
                ctx.beginPath();
                ctx.moveTo(x + size, y - size);
                ctx.lineTo(x - size, y + size);
                ctx.stroke();
            }
        }
    }
}

function drawWinProbChart() {
    const dpr = window.devicePixelRatio || 1;
    const w = chartCanvas.width / dpr;
    const h = chartCanvas.height / dpr;
    chartCtx.clearRect(0, 0, w, h);

    if (winProbHistory.length < 1) return;

    const curveLineWidth = 3;
    const padding = Math.max(2, curveLineWidth / 2 + 1);
    const innerW = Math.max(0, w - padding * 2);
    const innerH = Math.max(0, h - padding * 2);

    // Create vertical gradient for the curve: red above 50%, green below 50%
    const curveGradient = chartCtx.createLinearGradient(0, padding, 0, padding + innerH);
    curveGradient.addColorStop(0, "#ff4d4f");    // Red (Win > 50%)
    curveGradient.addColorStop(0.48, "#ff4d4f"); 
    curveGradient.addColorStop(0.5, "#d9d9d9");  // Neutral middle
    curveGradient.addColorStop(0.52, "#52c41a"); // Green (Win < 50%)
    curveGradient.addColorStop(1, "#52c41a");

    // Draw grid line for 50%
    chartCtx.beginPath();
    chartCtx.strokeStyle = "rgba(0, 0, 0, 0.05)";
    chartCtx.lineWidth = 1;
    chartCtx.setLineDash([5, 5]);
    const midY = padding + innerH / 2;
    chartCtx.moveTo(padding, midY);
    chartCtx.lineTo(padding + innerW, midY);
    chartCtx.stroke();
    chartCtx.setLineDash([]);

    // Draw win prob curve
    chartCtx.beginPath();
    chartCtx.strokeStyle = curveGradient;
    chartCtx.lineWidth = curveLineWidth;
    chartCtx.lineJoin = "round";
    chartCtx.lineCap = "round";

    if (winProbHistory.length === 1) {
        const y = padding + innerH - (winProbHistory[0] * innerH);
        chartCtx.moveTo(padding, y);
        chartCtx.lineTo(padding + innerW, y);
    } else {
        for (let i = 0; i < winProbHistory.length; i++) {
            const x = padding + (i / (winProbHistory.length - 1)) * innerW;
            const y = padding + innerH - (winProbHistory[i] * innerH);
            if (i === 0) chartCtx.moveTo(x, y);
            else chartCtx.lineTo(x, y);
        }
    }
    chartCtx.stroke();
}

function renderResults(data) {
    lastResults = data || { policy: null };
    
    let prob = 0.5;
    const aiColor = -playerColor;

    if (data && data.policy) {
        const v = data.rootValue;
        const rootToPlay = data.rootToPlay;
        prob = (rootToPlay === aiColor) ? (v + 1) / 2 : 1 - (v + 1) / 2;
        document.getElementById("mcts-value").innerText = (prob * 100).toFixed(1) + "%";
        
        if (winProbHistory.length === history.length + 1) {
            winProbHistory[winProbHistory.length - 1] = prob;
        } else {
            winProbHistory.push(prob);
        }
    } else {
        const winner = game.getWinner(state);
        if (winner !== null) {
            prob = (winner === aiColor) ? 1.0 : (winner === 0 ? 0.5 : 0.0);
            document.getElementById("mcts-value").innerText = (prob * 100).toFixed(1) + "%";
            
            if (winProbHistory.length === history.length + 1) {
                winProbHistory[winProbHistory.length - 1] = prob;
            } else {
                winProbHistory.push(prob);
            }
        } else {
            document.getElementById("mcts-value").innerText = "50.0%";
            // Don"t push 50% to history if it"s just the default
        }
    }
    
    drawWinProbChart();
}

let searchId = 0;

function handleAIResult(data) {
    if (data.searchId !== searchId) return;
    aiRunning = false;
    finishSearchStatus();
    renderResults(data);
    
    // Best move
    let bestAction = 0;
    let maxN = -1;
    for (let i = 0; i < data.policy.length; i++) {
        if (data.policy[i] > maxN) {
            maxN = data.policy[i];
            bestAction = i;
        }
    }

    lastMoveFromAI = true;
    makeMove(bestAction);
    lastMoveFromAI = false;
    drawBoard();
}

function makeMove(action) {
    history.push({ 
        state: state.map(l => new Int8Array(l)), 
        toPlay, 
        lastResults: lastResults,
        lastMove: lastMove ? { ...lastMove } : null
    });
    state = game.getNextState(state, action, toPlay);
    
    // Record last move
    const r = Math.floor(action / boardSize);
    const c = action % boardSize;
    lastMove = { r, c };
    
    // Notify worker for tree reuse
    worker.postMessage({ 
        type: "move", 
        action: action, 
        nextState: state, 
        nextToPlay: -toPlay 
    });

    toPlay = -toPlay;

    const winner = game.getWinner(state);
    if (winner !== null) {
        setIdleStatus(
            winner === 1 ? "分析完成：黑胜" : (winner === -1 ? "分析完成：白胜" : "分析完成：平局"),
            !lastMoveFromAI,
            !lastMoveFromAI
        );
        aiRunning = true; // Block moves
        renderResults(null); // Update win prob to final state
    } else {
        if (toPlay === playerColor) {
            setIdleStatus(toPlay === 1 ? "轮到黑棋" : "轮到白棋");
        } else {
            startSearchStatus();
            aiRunning = true;
            searchId++;
            worker.postMessage({ type: "search", simulations: numSimulations, state, toPlay, searchId });
        }
    }
}

canvas.onclick = function(e) {
    if (aiRunning || toPlay !== playerColor) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const c = Math.round((x - margin) / cellSize);
    const r = Math.round((y - margin) / cellSize);

    if (r >= 0 && r < boardSize && c >= 0 && c < boardSize) {
        const action = r * boardSize + c;
        const legalMask = game.getLegalActions(state, toPlay, false);
        if (legalMask[action]) {
            makeMove(action);
            drawBoard();
        }
    }
};

function resetGame() {
    state = game.getInitialState();
    toPlay = 1;
    history = [];
    winProbHistory = []; // Clear history
    aiRunning = false;
    lastMove = null;
    undoCount = 0;
    updateUndoButton();
    searchId++;
    renderResults(null);
    worker.postMessage({ type: "reset" });
    
    if (toPlay === playerColor) {
        setIdleStatus("轮到黑棋", true, true);
    } else {
        startSearchStatus();
        aiRunning = true;
        worker.postMessage({ type: "search", simulations: numSimulations, state, toPlay, searchId });
    }
    
    drawBoard();
}

function undo() {
    const isGameOver = game.getWinner(state) !== null;
    if (undoCount >= undoLimit) {
        setIdleStatus("本局悔棋次数已用完", true, true);
        return;
    }
    
    // Allow undo to interrupt AI search and perform undo
    if (aiRunning && !isGameOver) {
        aiRunning = false;
        searchId++;
        worker.postMessage({ type: "reset" });
        
        if (history.length > 0) {
            const prev = history.pop();
            winProbHistory.pop();
            state = prev.state;
            toPlay = prev.toPlay;
            lastMove = prev.lastMove;
            renderResults(prev.lastResults);
            drawBoard();
            undoCount++;
            updateUndoButton();
        }
        setIdleStatus(toPlay === 1 ? "轮到黑棋" : "轮到白棋", true, true);
        return;
    }
    
    if (history.length === 0) return;

    let prev;
    if (isGameOver) {
        if (toPlay === playerColor && history.length >= 2) {
            // Game ended on AI move, undo 2 moves to return to human turn
            history.pop();
            prev = history.pop();
            winProbHistory.pop();
            winProbHistory.pop();
        } else {
            // Game ended on human move (or not enough history), undo 1 move
            prev = history.pop();
            winProbHistory.pop();
        }
    } else if (toPlay !== playerColor) {
        // AI"s turn, undo 1 move
        prev = history.pop();
        winProbHistory.pop();
    } else if (history.length >= 2) {
        // Human"s turn, undo 2 moves (AI + Human)
        history.pop();
        prev = history.pop();
        winProbHistory.pop();
        winProbHistory.pop();
    } else {
        return;
    }

    state = prev.state;
    toPlay = prev.toPlay;
    lastMove = prev.lastMove;
    searchId++;
    renderResults(prev.lastResults);
    aiRunning = false;
    
    if (toPlay === playerColor) {
        setIdleStatus(toPlay === 1 ? "轮到黑棋" : "轮到白棋", true, true);
    } else {
        startSearchStatus();
        // Note: we don"t auto-trigger AI move on undo to human turn
    }
    
    worker.postMessage({ type: "reset" });
    drawBoard();
    undoCount++;
    updateUndoButton();
}

function updateSlider() {
    const slider = document.getElementById("player-slider");
    if (!slider) return;
    if (playerColor === 1) {
        slider.style.transform = "translateX(0)";
    } else {
        slider.style.transform = "translateX(100%)";
    }
}

document.getElementById("pick-black").onclick = () => {
    if (playerColor === 1) return;
    playerColor = 1;
    document.getElementById("pick-black").classList.add("active");
    document.getElementById("pick-white").classList.remove("active");
    updateSlider();
    resetGame();
};

document.getElementById("pick-white").onclick = () => {
    if (playerColor === -1) return;
    playerColor = -1;
    document.getElementById("pick-white").classList.add("active");
    document.getElementById("pick-black").classList.remove("active");
    updateSlider();
    resetGame();
};

document.getElementById("reset-btn").onclick = resetGame;
document.getElementById("undo-btn").onclick = undo;

document.getElementById("toggle-mcts").onclick = () => {
    showHeatmap = !showHeatmap;
    document.getElementById("toggle-mcts").classList.toggle("active", showHeatmap);
    drawBoard();
};

document.getElementById("toggle-forbidden").onclick = () => {
    showForbidden = !showForbidden;
    document.getElementById("toggle-forbidden").classList.toggle("active", showForbidden);
    drawBoard();
};

document.getElementById("toggle-mcts").classList.toggle("active", showHeatmap);
document.getElementById("toggle-forbidden").classList.toggle("active", showForbidden);

drawBoard();
