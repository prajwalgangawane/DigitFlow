const TRAINING_DATA = [
    { input: [1,1,1,1,1,1,0], target: [1,0,0,0,0,0,0,0,0,0], label: 0 },
    { input: [0,1,1,0,0,0,0], target: [0,1,0,0,0,0,0,0,0,0], label: 1 },
    { input: [1,1,0,1,1,0,1], target: [0,0,1,0,0,0,0,0,0,0], label: 2 },
    { input: [1,1,1,1,0,0,1], target: [0,0,0,1,0,0,0,0,0,0], label: 3 },
    { input: [0,1,1,0,0,1,1], target: [0,0,0,0,1,0,0,0,0,0], label: 4 },
    { input: [1,0,1,1,0,1,1], target: [0,0,0,0,0,1,0,0,0,0], label: 5 },
    { input: [1,0,1,1,1,1,1], target: [0,0,0,0,0,0,1,0,0,0], label: 6 },
    { input: [1,1,1,0,0,0,0], target: [0,0,0,0,0,0,0,1,0,0], label: 7 },
    { input: [1,1,1,1,1,1,1], target: [0,0,0,0,0,0,0,0,1,0], label: 8 },
    { input: [1,1,1,1,0,1,1], target: [0,0,0,0,0,0,0,0,0,1], label: 9 },
];

const SEG_NAMES = ["a","b","c","d","e","f","g"];

class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize, lr = 0.5, momentum = 0.0, dropout = 0.0) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.lr = lr;
        this.momentum = momentum;
        this.dropout = dropout;
        this.initWeights();
    }

    initWeights() {
        const scale1 = Math.sqrt(2.0 / this.inputSize);
        this.w1 = this.randomMatrix(this.inputSize, this.hiddenSize, scale1);
        this.b1 = new Array(this.hiddenSize).fill(0);
        const scale2 = Math.sqrt(2.0 / this.hiddenSize);
        this.w2 = this.randomMatrix(this.hiddenSize, this.outputSize, scale2);
        this.b2 = new Array(this.outputSize).fill(0);
        this.vw1 = this.zeroMatrix(this.inputSize, this.hiddenSize);
        this.vb1 = new Array(this.hiddenSize).fill(0);
        this.vw2 = this.zeroMatrix(this.hiddenSize, this.outputSize);
        this.vb2 = new Array(this.outputSize).fill(0);
        this.lastHidden = null;
        this.lastOutput = null;
        this.lastHiddenRaw = null;
        this.lastOutputRaw = null;
        this.lastInput = null;
        this.lastGradients = null;
    }

    snapshot() {
        return {
            w1: this.w1.map(r => [...r]),
            w2: this.w2.map(r => [...r]),
            b1: [...this.b1],
            b2: [...this.b2],
            vw1: this.vw1.map(r => [...r]),
            vw2: this.vw2.map(r => [...r]),
            vb1: [...this.vb1],
            vb2: [...this.vb2],
        };
    }

    restore(snap) {
        this.w1 = snap.w1.map(r => [...r]);
        this.w2 = snap.w2.map(r => [...r]);
        this.b1 = [...snap.b1];
        this.b2 = [...snap.b2];
        this.vw1 = snap.vw1.map(r => [...r]);
        this.vw2 = snap.vw2.map(r => [...r]);
        this.vb1 = [...snap.vb1];
        this.vb2 = [...snap.vb2];
    }

    randomMatrix(rows, cols, scale) {
        const m = [];
        for (let i = 0; i < rows; i++) {
            const row = [];
            for (let j = 0; j < cols; j++) {
                row.push((Math.random() * 2 - 1) * scale);
            }
            m.push(row);
        }
        return m;
    }

    zeroMatrix(rows, cols) {
        return Array.from({ length: rows }, () => new Array(cols).fill(0));
    }

    sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))); }
    sigmoidDeriv(y) { return y * (1 - y); }

    softmax(arr) {
        const max = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sum);
    }

    forward(input) {
        this.lastInput = [...input];
        const hiddenRaw = [];
        const hidden = [];
        for (let j = 0; j < this.hiddenSize; j++) {
            let sum = this.b1[j];
            for (let i = 0; i < this.inputSize; i++) {
                sum += input[i] * this.w1[i][j];
            }
            hiddenRaw.push(sum);
            hidden.push(this.sigmoid(sum));
        }
        const outputRaw = [];
        for (let j = 0; j < this.outputSize; j++) {
            let sum = this.b2[j];
            for (let i = 0; i < this.hiddenSize; i++) {
                sum += hidden[i] * this.w2[i][j];
            }
            outputRaw.push(sum);
        }
        const output = this.softmax(outputRaw);
        this.lastHidden = hidden;
        this.lastOutput = output;
        this.lastHiddenRaw = hiddenRaw;
        this.lastOutputRaw = outputRaw;
        return { input, hidden, output, hiddenRaw, outputRaw };
    }

    backward(target, dropoutMask = null) {
        const input = this.lastInput;
        const hidden = this.lastHidden;
        const output = this.lastOutput;
        const outDelta = output.map((o, i) => o - target[i]);
        const dW2 = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            dW2.push(outDelta.map(d => hidden[i] * d));
        }
        const dB2 = outDelta.map(d => d);
        const hiddenDelta = [];
        for (let i = 0; i < this.hiddenSize; i++) {
            let sum = 0;
            for (let j = 0; j < this.outputSize; j++) {
                sum += outDelta[j] * this.w2[i][j];
            }
            const deriv = this.sigmoidDeriv(hidden[i]);
            const mask = (dropoutMask && dropoutMask[i]) ? 0 : 1;
            hiddenDelta.push(sum * deriv * mask);
        }
        const dW1 = [];
        for (let i = 0; i < this.inputSize; i++) {
            dW1.push(hiddenDelta.map(d => input[i] * d));
        }
        const dB1 = hiddenDelta.map(d => d);
        this.lastGradients = { dW1, dB1, dW2, dB2, hiddenDelta, outDelta };
        for (let i = 0; i < this.inputSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                this.vw1[i][j] = this.momentum * this.vw1[i][j] - this.lr * dW1[i][j];
                this.w1[i][j] += this.vw1[i][j];
            }
        }
        for (let j = 0; j < this.hiddenSize; j++) {
            this.vb1[j] = this.momentum * this.vb1[j] - this.lr * dB1[j];
            this.b1[j] += this.vb1[j];
        }
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.outputSize; j++) {
                this.vw2[i][j] = this.momentum * this.vw2[i][j] - this.lr * dW2[i][j];
                this.w2[i][j] += this.vw2[i][j];
            }
        }
        for (let j = 0; j < this.outputSize; j++) {
            this.vb2[j] = this.momentum * this.vb2[j] - this.lr * dB2[j];
            this.b2[j] += this.vb2[j];
        }
        const loss = target.reduce((s, t, i) => s - t * Math.log(Math.max(output[i], 1e-10)), 0);
        return loss;
    }

    trainSample() {
        const sample = TRAINING_DATA[Math.floor(Math.random() * TRAINING_DATA.length)];
        let dropoutMask = null;
        if (this.dropout > 0) {
            dropoutMask = Array.from({ length: this.hiddenSize }, () => Math.random() > this.dropout);
        }
        this.forward(sample.input);
        const pred = this.lastOutput;
        const predIdx = pred.indexOf(Math.max(...pred));
        const targetIdx = sample.target.indexOf(1);
        const correct = predIdx === targetIdx;
        const loss = this.backward(sample.target, dropoutMask);
        return { loss, correct, sample: TRAINING_DATA.indexOf(sample) };
    }

    evaluateAll() {
        const results = TRAINING_DATA.map((s, i) => {
            this.forward(s.input);
            const output = [...this.lastOutput];
            const loss = s.target.reduce((sum, t, j) => sum - t * Math.log(Math.max(output[j], 1e-10)), 0);
            const predIdx = output.indexOf(Math.max(...output));
            return { digit: i, loss, output, predIdx, correct: predIdx === i, target: s.target };
        });
        const avgLoss = results.reduce((s, r) => s + r.loss, 0) / results.length;
        const avgAcc = results.filter(r => r.correct).length / results.length;
        return { results, avgLoss, avgAcc };
    }

    predict(input) {
        return this.forward(input);
    }
}

function SevenSegmentDisplay({ segments, toggleSegment }) {
    const segStyle = (key, on) => {
        const base = {
            position: "absolute",
            background: on ? "#ff2d2d" : "#300",
            cursor: "pointer",
            borderRadius: "3px",
            transition: "background 0.15s, box-shadow 0.15s",
            boxShadow: on ? "0 0 10px rgba(255,45,45,0.6)" : "none",
        };
        const pos = {
            a: { top: 0, left: 22, width: 56, height: 10 },
            b: { top: 12, right: 0, width: 10, height: 68 },
            c: { bottom: 12, right: 0, width: 10, height: 68 },
            d: { bottom: 0, left: 22, width: 56, height: 10 },
            e: { bottom: 12, left: 0, width: 10, height: 68 },
            f: { top: 12, left: 0, width: 10, height: 68 },
            g: { top: 85, left: 22, width: 56, height: 10 },
        };
        return { ...base, ...pos[key] };
    };

    return (
        <div style={{ position: "relative", width: 100, height: 180, margin: "16px auto" }}>
            {SEG_NAMES.map(key => (
                <div key={key} onClick={() => toggleSegment(key)} style={segStyle(key, segments[key])} />
            ))}
        </div>
    );
}

function NetworkVisualization({ nn, segments, lastForward, inspectTarget, setInspectTarget, animationProgress, weightVersion }) {
    const containerRef = React.useRef(null);
    const [dims, setDims] = React.useState({ width: 600, height: 300 });

    React.useEffect(() => {
        const update = () => {
            if (containerRef.current) {
                const r = containerRef.current.getBoundingClientRect();
                setDims({ width: Math.max(300, Math.round(r.width)), height: Math.max(200, Math.round(r.height)) });
            }
        };
        update();
        window.addEventListener("resize", update);
        return () => window.removeEventListener("resize", update);
    }, []);

    const inputVals = SEG_NAMES.map(k => segments[k] ? 1 : 0);
    const layers = [
        { name: "Input", count: 7, activations: inputVals, labels: SEG_NAMES },
        { name: "Hidden", count: nn.hiddenSize, activations: lastForward?.hidden || new Array(nn.hiddenSize).fill(0), labels: null },
        { name: "Output", count: 10, activations: lastForward?.output || new Array(10).fill(0.1), labels: [0,1,2,3,4,5,6,7,8,9] },
    ];

    const marginX = 50;
    const marginY = 25;
    const layerX = layers.map((_, i) => marginX + (i * (dims.width - marginX * 2)) / Math.max(layers.length - 1, 1));
    const neuronPositions = layers.map((layer, li) => {
        const positions = [];
        const spacing = Math.min(36, (dims.height - marginY * 2) / layer.count);
        const totalH = (layer.count - 1) * spacing;
        const startY = (dims.height - totalH) / 2;
        for (let i = 0; i < layer.count; i++) {
            positions.push({ x: layerX[li], y: startY + i * spacing, activation: layer.activations[i], label: layer.labels?.[i] ?? null });
        }
        return positions;
    });

    const connections = [];
    for (let l = 0; l < layers.length - 1; l++) {
        const matrix = l === 0 ? nn.w1 : nn.w2;
        for (let i = 0; i < neuronPositions[l].length; i++) {
            for (let j = 0; j < neuronPositions[l + 1].length; j++) {
                const w = matrix[i]?.[j] ?? 0;
                const contribution = neuronPositions[l][i].activation * w;
                connections.push({
                    x1: neuronPositions[l][i].x, y1: neuronPositions[l][i].y,
                    x2: neuronPositions[l + 1][j].x, y2: neuronPositions[l + 1][j].y,
                    weight: w, contribution, fromLayer: l, fromIdx: i, toLayer: l + 1, toIdx: j,
                });
            }
        }
    }

    const maxAbsW = Math.max(1, ...connections.map(c => Math.abs(c.weight)));
    const progress = animationProgress > 0 ? Math.min(animationProgress, 1) : 0;
    const particleX = 60 + progress * (dims.width - 120);

    return (
        <div style={{ width: "100%", height: "100%" }} ref={containerRef}>
            <svg width={dims.width} height={dims.height} style={{ display: "block" }}>
                <defs>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                        <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
                    </filter>
                </defs>
                {connections.map((c, i) => {
                    const opacity = Math.abs(c.weight) / maxAbsW * 0.4 + 0.05;
                    const color = c.weight >= 0 ? `rgba(255,100,80,${opacity})` : `rgba(80,140,255,${opacity})`;
                    return (
                        <line
                            key={`c-${i}`}
                            x1={c.x1} y1={c.y1} x2={c.x2} y2={c.y2}
                            stroke={color} strokeWidth={Math.abs(c.weight) / maxAbsW * 2 + 0.5}
                            onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); setInspectTarget({ type: "connection", weight: c.weight, from: `L${c.fromLayer}[${c.fromIdx}]`, to: `L${c.toLayer}[${c.toIdx}]`, contribution: c.contribution, px: r.left + (r.width/2), py: r.top }); }}
                            style={{ cursor: "pointer" }}
                        />
                    );
                })}
                {animationProgress > 0 && animationProgress <= 1 && (
                    <>
                        <line x1={particleX} y1={0} x2={particleX} y2={dims.height} stroke="#58a6ff" strokeWidth="1" opacity="0.3" strokeDasharray="4,4" />
                        {neuronPositions.map((layer, li) =>
                            layer.map((n, ni) => {
                                const layerProgress = (li + 0.5) / layers.length;
                                if (Math.abs(progress - layerProgress) < 0.15) {
                                    return (
                                        <circle key={`p-${li}-${ni}`} cx={n.x} cy={n.y} r={6} fill="#58a6ff" opacity="0.8" filter="url(#glow)">
                                            <animate attributeName="opacity" values="0.8;0.3;0.8" dur="0.5s" repeatCount="indefinite" />
                                        </circle>
                                    );
                                }
                                return null;
                            })
                        )}
                    </>
                )}
                {neuronPositions.map((layer, li) =>
                    layer.map((n, ni) => {
                        const act = Math.max(0, Math.min(1, n.activation));
                        const r = li === 0 ? 14 : (li === 1 ? 16 : 12);
                        const fill = li === 0
                            ? (act > 0.5 ? "#ff2d2d" : "#30363d")
                            : `rgb(${Math.floor(act * 80)}, ${Math.floor(100 + act * 155)}, ${Math.floor(80 + act * 56)})`;
                        return (
                            <g key={`n-${li}-${ni}`}>
                                <circle cx={n.x} cy={n.y} r={r} fill={fill} stroke="#444c56" strokeWidth="1.5"
                                    onClick={(e) => { const rect = e.currentTarget.closest('svg').getBoundingClientRect(); setInspectTarget({
                                        type: "neuron", layer: li, idx: ni, activation: act,
                                        raw: li === 1 ? (nn.lastHiddenRaw?.[ni] ?? 0) : (nn.lastOutputRaw?.[ni] ?? 0),
                                        bias: li === 1 ? nn.b1[ni] : nn.b2[ni],
                                        px: rect.left + n.x, py: rect.top + n.y,
                                    })}}
                                    style={{ cursor: "pointer" }}
                                />
                                <text x={n.x} y={n.y + 4} textAnchor="middle" fill="#fff" fontSize="10" fontWeight="bold" pointerEvents="none">
                                    {n.label ?? ""}
                                </text>
                                {li === 0 && (
                                    <text x={n.x} y={n.y + r + 12} textAnchor="middle" fill="#8b949e" fontSize="9" pointerEvents="none">
                                        {SEG_NAMES[ni]}
                                    </text>
                                )}
                                {li === 2 && (
                                    <text x={n.x} y={n.y + r + 12} textAnchor="middle" fill={act === Math.max(...layer.map(l2 => l2.activation)) ? "#2ea043" : "#8b949e"} fontSize="9" fontWeight={act === Math.max(...layer.map(l2 => l2.activation)) ? "bold" : "normal"} pointerEvents="none">
                                        {(act * 100).toFixed(0)}%
                                    </text>
                                )}
                            </g>
                        );
                    })
                )}
                {layers.map((l, i) => (
                    <text key={`lname-${i}`} x={layerX[i]} y={18} textAnchor="middle" fill="#8b949e" fontSize="11" fontWeight="bold">
                        {l.name}
                    </text>
                ))}
            </svg>
        </div>
    );
}

function LossCurve({ history, height = 120, showAcc = true }) {
    const [width, setWidth] = React.useState(500);
    const svgRef = React.useRef(null);

    React.useEffect(() => {
        const update = () => {
            if (svgRef.current?.parentElement) {
                setWidth(svgRef.current.parentElement.getBoundingClientRect().width);
            }
        };
        update();
        window.addEventListener("resize", update);
        return () => window.removeEventListener("resize", update);
    }, []);

    const pad = { top: 10, right: 10, bottom: 20, left: 40 };
    const w = width - pad.left - pad.right;
    const chartH = height - pad.top - pad.bottom;

    if (history.length < 2) return <div ref={svgRef} style={{ width: "100%", height }} />;

    const maxLoss = Math.max(0.01, ...history.map(e => e.loss));
    const lossPoints = history.map((e, i) => ({
        x: pad.left + (i / (history.length - 1)) * w,
        y: pad.top + chartH * (1 - e.loss / maxLoss),
    }));
    const accPoints = history.map((e, i) => ({
        x: pad.left + (i / (history.length - 1)) * w,
        y: pad.top + chartH * (1 - e.accuracy),
    }));
    const lossPath = lossPoints.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");
    const accPath = accPoints.map((p, i) => `${i === 0 ? "M" : "L"}${p.x},${p.y}`).join(" ");
    const lastLoss = lossPoints[lossPoints.length - 1];
    const lastAcc = accPoints[accPoints.length - 1];

    return (
        <div ref={svgRef} style={{ width: "100%", height }}>
            <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
                {[0, 0.25, 0.5, 0.75, 1].map(f => (
                    <line key={f} x1={pad.left} y1={pad.top + chartH * (1 - f)} x2={pad.left + w} y2={pad.top + chartH * (1 - f)} stroke="#21262d" />
                ))}
                <path d={lossPath} fill="none" stroke="#f85149" strokeWidth="2" />
                {showAcc && <path d={accPath} fill="none" stroke="#2ea043" strokeWidth="2" />}
                <circle cx={lastLoss.x} cy={lastLoss.y} r={3} fill="#f85149" />
                <text x={lastLoss.x - 5} y={lastLoss.y - 6} textAnchor="end" fill="#f85149" fontSize="9" fontWeight="bold">
                    {history[history.length-1].loss.toFixed(3)}
                </text>
                {showAcc && (
                    <>
                        <circle cx={lastAcc.x} cy={lastAcc.y} r={3} fill="#2ea043" />
                        <text x={pad.left + w - 3} y={lastLoss.y - 5} textAnchor="end" fill="#f85149" fontSize="8">Loss</text>
                        <text x={pad.left + w - 3} y={lastAcc.y + 10} textAnchor="end" fill="#2ea043" fontSize="8">Acc</text>
                    </>
                )}
                <text x={pad.left} y={height - 3} fill="#8b949e" fontSize="9">Epoch</text>
            </svg>
        </div>
    );
}

function InfoTooltip({ text, title }) {
    const [show, setShow] = React.useState(false);
    const ref = React.useRef(null);
    const [pos, setPos] = React.useState({ top: 0, left: 0 });

    const handleShow = () => {
        if (ref.current) {
            const r = ref.current.getBoundingClientRect();
            setPos({ top: r.bottom + 6, left: Math.min(r.left, window.innerWidth - 310) });
        }
        setShow(true);
    };

    return (
        <>
            <span ref={ref} className="info-trigger" onMouseEnter={handleShow} onMouseLeave={() => setShow(false)} onClick={() => setShow(!show)} tabIndex={0}>?</span>
            {show && (
                <>
                    <div style={{ position: "fixed", inset: 0, zIndex: 199 }} onMouseDown={() => setShow(false)} />
                    <div className="info-popup" style={{ top: pos.top, left: pos.left }}>
                        <div className="info-title">{title}</div>
                        <div className="info-body">{text}</div>
                    </div>
                </>
            )}
        </>
    );
}

function WeightMatrixView({ w1, w2, prevW1, prevW2, b1, b2, setWeight, setBias, label, editingCell, setEditingCell, editValue, setEditValue }) {
    const handleCellClick = (matrix, row, col, value) => {
        setEditingCell({ matrix, row, col });
        setEditValue(value.toFixed(4));
    };
    const handleEditSubmit = () => {
        if (!editingCell) return;
        const num = parseFloat(editValue);
        if (isNaN(num)) { setEditingCell(null); return; }
        if (editingCell.matrix === "w1") {
            setWeight("w1", editingCell.row, editingCell.col, num);
        } else if (editingCell.matrix === "w2") {
            setWeight("w2", editingCell.row, editingCell.col, num);
        } else if (editingCell.matrix === "b1") {
            setBias("b1", editingCell.row, num);
        } else if (editingCell.matrix === "b2") {
            setBias("b2", editingCell.row, num);
        }
        setEditingCell(null);
    };

    const segLabels = ["a","b","c","d","e","f","g"];
    const digitLabels = [0,1,2,3,4,5,6,7,8,9];

    return (
            <div className="matrix-container">
            <div>
                <div style={{ fontSize: "11px", color: "#8b949e", marginBottom: "4px" }}>{label}</div>
                <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
                <div>
                    <div style={{ fontSize: "10px", color: "#58a6ff", marginBottom: "2px" }}>Input→Hidden (w1)</div>
                    <table className="matrix-table">
                        <thead>
                            <tr>
                                <td style={{ color: "#58a6ff", fontWeight: "bold" }}>In</td>
                                {w1[0]?.map((_, j) => <td key={j} style={{ color: "#58a6ff", fontWeight: "bold", fontSize: "8px" }}>h{j}</td>)}
                            </tr>
                        </thead>
                        <tbody>
                            {w1.map((row, i) => (
                                <tr key={i}>
                                    <td style={{ color: "#8b949e", fontSize: "9px", fontWeight: "bold" }}>{segLabels[i]}</td>
                                    {row.map((v, j) => {
                                        const isEditing = editingCell?.matrix === "w1" && editingCell?.row === i && editingCell?.col === j;
                                        const prev = prevW1?.[i]?.[j];
                                        const delta = prev !== undefined ? v - prev : 0;
                                        const color = delta > 0.001 ? "#2ea043" : delta < -0.001 ? "#f85149" : "#8b949e";
                                        return (
                                            <td key={j}>
                                                {isEditing ? (
                                                    <input
                                                        type="number"
                                                        step="0.1"
                                                        value={editValue}
                                                        onChange={e => setEditValue(e.target.value)}
                                                        onBlur={handleEditSubmit}
                                                        onKeyDown={e => { if (e.key === "Enter") handleEditSubmit(); if (e.key === "Escape") setEditingCell(null); }}
                                                        style={{
                                                            width: "52px",
                                                            background: "#0d1117",
                                                            color: "#e6edf3",
                                                            border: "1px solid #58a6ff",
                                                            borderRadius: "2px",
                                                            padding: "1px 2px",
                                                            fontSize: "10px",
                                                            textAlign: "center",
                                                        }}
                                                        autoFocus
                                                    />
                                                ) : (
                                                    <span
                                                        onClick={() => handleCellClick("w1", i, j, v)}
                                                        style={{ color, cursor: "pointer" }}
                                                        title={`Click to edit w1[${i}][${j}]`}
                                                    >
                                                        {v.toFixed(2)}
                                                    </span>
                                                )}
                                            </td>
                                        );
                                    })}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
                <div>
                    <div style={{ fontSize: "10px", color: "#d29922", marginBottom: "2px" }}>Hidden→Output (w2)</div>
                    <table className="matrix-table">
                        <thead>
                            <tr>
                                <td style={{ color: "#d29922", fontWeight: "bold" }}>Hid</td>
                                {w2[0]?.map((_, j) => <td key={j} style={{ color: "#d29922", fontWeight: "bold", fontSize: "8px" }}>{digitLabels[j]}</td>)}
                            </tr>
                        </thead>
                        <tbody>
                            {w2.map((row, i) => (
                                <tr key={i}>
                                    <td style={{ color: "#8b949e", fontSize: "9px", fontWeight: "bold" }}>h{i}</td>
                                    {row.map((v, j) => {
                                        const isEditing = editingCell?.matrix === "w2" && editingCell?.row === i && editingCell?.col === j;
                                        const prev = prevW2?.[i]?.[j];
                                        const delta = prev !== undefined ? v - prev : 0;
                                        const color = delta > 0.001 ? "#2ea043" : delta < -0.001 ? "#f85149" : "#8b949e";
                                        return (
                                            <td key={j}>
                                                {isEditing ? (
                                                    <input
                                                        type="number"
                                                        step="0.1"
                                                        value={editValue}
                                                        onChange={e => setEditValue(e.target.value)}
                                                        onBlur={handleEditSubmit}
                                                        onKeyDown={e => { if (e.key === "Enter") handleEditSubmit(); if (e.key === "Escape") setEditingCell(null); }}
                                                        style={{
                                                            width: "52px",
                                                            background: "#0d1117",
                                                            color: "#e6edf3",
                                                            border: "1px solid #58a6ff",
                                                            borderRadius: "2px",
                                                            padding: "1px 2px",
                                                            fontSize: "10px",
                                                            textAlign: "center",
                                                        }}
                                                        autoFocus
                                                    />
                                                ) : (
                                                    <span
                                                        onClick={() => handleCellClick("w2", i, j, v)}
                                                        style={{ color, cursor: "pointer" }}
                                                        title={`Click to edit w2[${i}][${j}]`}
                                                    >
                                                        {v.toFixed(2)}
                                                    </span>
                                                )}
                                            </td>
                                        );
                                    })}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
            <div style={{ display: "flex", gap: "16px", marginTop: "12px" }}>
                <div>
                    <div style={{ fontSize: "10px", color: "#58a6ff", marginBottom: "4px" }}>Hidden Biases (b1)</div>
                    <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                        {b1.map((v, i) => {
                            const isEditing = editingCell?.matrix === "b1" && editingCell?.row === i;
                            return (
                                <div key={i} style={{ display: "flex", alignItems: "center", gap: "2px" }}>
                                    <span style={{ fontSize: "9px", color: "#8b949e" }}>h{i}:</span>
                                    {isEditing ? (
                                        <input
                                            type="number"
                                            step="0.1"
                                            value={editValue}
                                            onChange={e => setEditValue(e.target.value)}
                                            onBlur={handleEditSubmit}
                                            onKeyDown={e => { if (e.key === "Enter") handleEditSubmit(); if (e.key === "Escape") setEditingCell(null); }}
                                            style={{
                                                width: "52px",
                                                background: "#0d1117",
                                                color: "#e6edf3",
                                                border: "1px solid #58a6ff",
                                                borderRadius: "2px",
                                                padding: "1px 2px",
                                                fontSize: "10px",
                                                textAlign: "center",
                                            }}
                                            autoFocus
                                        />
                                    ) : (
                                        <span onClick={() => handleCellClick("b1", i, 0, v)} style={{ fontSize: "10px", color: "#e6edf3", cursor: "pointer" }} title="Click to edit">
                                            {v.toFixed(3)}
                                        </span>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                </div>
                <div>
                    <div style={{ fontSize: "10px", color: "#d29922", marginBottom: "4px" }}>Output Biases (b2)</div>
                    <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                        {b2.map((v, i) => {
                            const isEditing = editingCell?.matrix === "b2" && editingCell?.row === i;
                            return (
                                <div key={i} style={{ display: "flex", alignItems: "center", gap: "2px" }}>
                                    <span style={{ fontSize: "9px", color: "#8b949e" }}>{digitLabels[i]}:</span>
                                    {isEditing ? (
                                        <input
                                            type="number"
                                            step="0.1"
                                            value={editValue}
                                            onChange={e => setEditValue(e.target.value)}
                                            onBlur={handleEditSubmit}
                                            onKeyDown={e => { if (e.key === "Enter") handleEditSubmit(); if (e.key === "Escape") setEditingCell(null); }}
                                            style={{
                                                width: "52px",
                                                background: "#0d1117",
                                                color: "#e6edf3",
                                                border: "1px solid #58a6ff",
                                                borderRadius: "2px",
                                                padding: "1px 2px",
                                                fontSize: "10px",
                                                textAlign: "center",
                                            }}
                                            autoFocus
                                        />
                                    ) : (
                                        <span onClick={() => handleCellClick("b2", i, 0, v)} style={{ fontSize: "10px", color: "#e6edf3", cursor: "pointer" }} title="Click to edit">
                                            {v.toFixed(3)}
                                        </span>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>
            </div>
        </div>
    );
}

function App() {
    const [hiddenSize, setHiddenSize] = React.useState(8);
    const [learningRate, setLearningRate] = React.useState(0.5);
    const [batchSize, setBatchSize] = React.useState(1);
    const [momentum, setMomentum] = React.useState(0.0);
    const [dropout, setDropout] = React.useState(0.0);

    const [nn, setNn] = React.useState(() => new NeuralNetwork(7, hiddenSize, 10, learningRate, momentum, dropout));
    const [history, setHistory] = React.useState([]);
    const [epoch, setEpoch] = React.useState(0);
    const [autoTraining, setAutoTraining] = React.useState(false);
    const [activeTab, setActiveTab] = React.useState("prediction");
    const [inspectTarget, setInspectTarget] = React.useState(null);
    const [animationProgress, setAnimationProgress] = React.useState(0);
    const [lastForward, setLastForward] = React.useState(null);
    const [prevWeights, setPrevWeights] = React.useState({ w1: null, w2: null });
    const [undoStack, setUndoStack] = React.useState([]);
    const [weightVersion, setWeightVersion] = React.useState(0);
    const [editingCell, setEditingCell] = React.useState(null);
    const [editValue, setEditValue] = React.useState("");

    const [manualInput, setManualInput] = React.useState([0,0,0,0,0,0,0]);
    const [manualTarget, setManualTarget] = React.useState(0);
    const [manualResult, setManualResult] = React.useState(null);
    const [savedEpochs, setSavedEpochs] = React.useState(() => {
        try {
            return JSON.parse(localStorage.getItem('nn-epochs') || '[]');
        } catch { return []; }
    });
    const [showEpochList, setShowEpochList] = React.useState(false);

    const [segments, setSegments] = React.useState({
        a: false, b: false, c: false, d: false, e: false, f: false, g: false
    });

    const nnRef = React.useRef(nn);
    const batchSizeRef = React.useRef(batchSize);
    const undoStackRef = React.useRef([]);
    React.useEffect(() => { nnRef.current = nn; }, [nn]);
    React.useEffect(() => { batchSizeRef.current = batchSize; }, [batchSize]);
    React.useEffect(() => { undoStackRef.current = undoStack; }, [undoStack]);

    React.useEffect(() => {
        const newNn = new NeuralNetwork(7, hiddenSize, 10, learningRate, momentum, dropout);
        setNn(newNn);
        setHistory([]);
        setEpoch(0);
        setLastForward(null);
        setPrevWeights({ w1: null, w2: null });
        setUndoStack([]);
        setManualResult(null);
        setSavedEpochs([]);
        localStorage.removeItem('nn-epochs');
    }, [hiddenSize]);

    const doTrainStepRef = React.useRef(null);
    const doUndoRef = React.useRef(null);

    React.useEffect(() => {
        nn.lr = learningRate;
        nn.momentum = momentum;
        nn.dropout = dropout;
    }, [nn, learningRate, momentum, dropout]);

    React.useEffect(() => {
        const handler = (e) => {
            if (e.target.tagName === "INPUT") return;
            if (e.key === " " || e.key === "Spacebar") {
                e.preventDefault();
                setAutoTraining(a => !a);
            } else if (e.key === "ArrowRight" && e.shiftKey) {
                e.preventDefault();
                doTrainStepRef.current?.();
            } else if (e.key === "ArrowLeft" && e.shiftKey) {
                e.preventDefault();
                doUndoRef.current?.();
            }
        };
        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
    }, []);

    const epochRef = React.useRef(epoch);
    React.useEffect(() => { epochRef.current = epoch; }, [epoch]);

    React.useEffect(() => {
        if (!autoTraining) return;
        const interval = setInterval(() => {
            const currentNn = nnRef.current;
            const currentBatch = batchSizeRef.current;
            for (let i = 0; i < currentBatch; i++) {
                const snap = currentNn.snapshot();
                const result = currentNn.trainSample();
                setEpoch(e => {
                    const ne = e + 1;
                    epochRef.current = ne;
                    setHistory(h => [...h, { epoch: ne, loss: result.loss, accuracy: result.correct ? 1 : 0 }]);
                    return ne;
                });
                setPrevWeights({ w1: snap.w1, w2: snap.w2 });
                const newUndoStack = [...undoStackRef.current, snap];
                setUndoStack(newUndoStack);
                undoStackRef.current = newUndoStack;
                saveEpochToStorage(epochRef.current, snap, result.loss, result.correct ? 1 : 0);
            }
        }, 100);
        return () => clearInterval(interval);
    }, [autoTraining]);

    const doTrainStep = () => {
        const snap = nn.snapshot();
        const result = nn.trainSample();
        const newEpoch = epoch + 1;
        setEpoch(newEpoch);
        setHistory(prev => [...prev, { epoch: newEpoch, loss: result.loss, accuracy: result.correct ? 1 : 0 }]);
        setPrevWeights({ w1: snap.w1, w2: snap.w2 });
        setUndoStack(prev => [...prev, snap]);
        saveEpochToStorage(newEpoch, snap, result.loss, result.correct ? 1 : 0);
    };

    const doUndo = () => {
        if (undoStack.length === 0) return;
        const snap = undoStack[undoStack.length - 1];
        nn.restore(snap);
        setUndoStack(prev => prev.slice(0, -1));
        setEpoch(prev => prev - 1);
        setHistory(prev => prev.slice(0, -1));
        setPrevWeights({ w1: null, w2: null });
        setLastForward(null);
        setWeightVersion(prev => prev + 1);
    };

    React.useEffect(() => {
        doTrainStepRef.current = doTrainStep;
        doUndoRef.current = doUndo;
    });

    const pushUndoSnap = () => {
        const snap = nn.snapshot();
        setUndoStack(prev => [...prev, snap]);
        return snap;
    };

    const saveEpochToStorage = (epochNum, snap, loss, accuracy) => {
        setSavedEpochs(prev => {
            const entry = { epoch: epochNum, snap, loss, accuracy, timestamp: Date.now() };
            const next = [...prev, entry];
            if (next.length > 500) next.splice(0, next.length - 500);
            try { localStorage.setItem('nn-epochs', JSON.stringify(next)); } catch {}
            return next;
        });
    };

    const loadEpoch = (entry) => {
        nn.restore(entry.snap);
        setEpoch(entry.epoch);
        setHistory(h => h.slice(0, entry.epoch));
        setPrevWeights({ w1: null, w2: null });
        setLastForward(null);
        setManualResult(null);
        setUndoStack([]);
        setWeightVersion(prev => prev + 1);
        setShowEpochList(false);
    };

    const clearSavedEpochs = () => {
        localStorage.removeItem('nn-epochs');
        setSavedEpochs([]);
    };

    const exportWeightsCSV = () => {
        const rows = [];
        rows.push("# w1: Input→Hidden");
        rows.push("from,to,weight");
        for (let i = 0; i < nn.w1.length; i++)
            for (let j = 0; j < nn.w1[i].length; j++)
                rows.push(`w1,${i},${j},${nn.w1[i][j].toFixed(6)}`);
        rows.push("# w2: Hidden→Output");
        rows.push("from,to,weight");
        for (let i = 0; i < nn.w2.length; i++)
            for (let j = 0; j < nn.w2[i].length; j++)
                rows.push(`w2,${i},${j},${nn.w2[i][j].toFixed(6)}`);
        rows.push("# b1: Hidden Biases");
        rows.push("index,bias");
        for (let i = 0; i < nn.b1.length; i++)
            rows.push(`b1,${i},${nn.b1[i].toFixed(6)}`);
        rows.push("# b2: Output Biases");
        rows.push("index,bias");
        for (let i = 0; i < nn.b2.length; i++)
            rows.push(`b2,${i},${nn.b2[i].toFixed(6)}`);
        const blob = new Blob([rows.join("\n")], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `nn-weights-epoch${epoch}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    };

    const importWeightsCSV = (file) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const lines = e.target.result.split("\n").filter(l => l.trim() && !l.startsWith("#"));
                pushUndoSnap();
                for (const line of lines) {
                    const parts = line.split(",");
                    if (parts[0] === "w1") nn.w1[parseInt(parts[1])][parseInt(parts[2])] = parseFloat(parts[3]);
                    else if (parts[0] === "w2") nn.w2[parseInt(parts[1])][parseInt(parts[2])] = parseFloat(parts[3]);
                    else if (parts[0] === "b1") nn.b1[parseInt(parts[1])] = parseFloat(parts[2]);
                    else if (parts[0] === "b2") nn.b2[parseInt(parts[1])] = parseFloat(parts[2]);
                }
                setWeightVersion(prev => prev + 1);
                setLastForward(null);
                setEditingCell(null);
            } catch (err) {
                alert("Failed to import CSV: " + err.message);
            }
        };
        reader.readAsText(file);
    };

    const setWeight = (matrix, row, col, value) => {
        pushUndoSnap();
        if (matrix === "w1") {
            nn.w1[row][col] = value;
        } else {
            nn.w2[row][col] = value;
        }
        setWeightVersion(prev => prev + 1);
    };

    const setBias = (layer, idx, value) => {
        pushUndoSnap();
        if (layer === "b1") {
            nn.b1[idx] = value;
        } else {
            nn.b2[idx] = value;
        }
        setWeightVersion(prev => prev + 1);
    };

    const skipTraining = () => {
        let e = epoch;
        const newHistory = [...history];
        const saved = [...savedEpochs];
        for (let i = 0; i < 2000; i++) {
            const snap = nn.snapshot();
            const result = nn.trainSample();
            e++;
            newHistory.push({ epoch: e, loss: result.loss, accuracy: result.correct ? 1 : 0 });
            saved.push({ epoch: e, snap, loss: result.loss, accuracy: result.correct ? 1 : 0, timestamp: Date.now() });
        }
        setEpoch(e);
        setHistory(newHistory);
        const trimmed = saved.length > 500 ? saved.slice(-500) : saved;
        setSavedEpochs(trimmed);
        try { localStorage.setItem('nn-epochs', JSON.stringify(trimmed)); } catch {}
    };

    const runPrediction = () => {
        const input = SEG_NAMES.map(k => segments[k] ? 1 : 0);
        setAnimationProgress(0.01);
        let progress = 0;
        const interval = setInterval(() => {
            progress += 0.03;
            setAnimationProgress(progress);
            if (progress >= 1) {
                clearInterval(interval);
                const result = nn.predict(input);
                setLastForward(result);
                setAnimationProgress(0);
            }
        }, 30);
    };

    const clearAll = () => {
        setSegments({ a: false, b: false, c: false, d: false, e: false, f: false, g: false });
        setLastForward(null);
        setAnimationProgress(0);
    };

    const toggleSegment = (key) => {
        setSegments(prev => ({ ...prev, [key]: !prev[key] }));
    };

    const doManualForward = () => {
        const target = new Array(10).fill(0);
        target[manualTarget] = 1;
        const snap = nn.snapshot();
        nn.forward(manualInput);
        const output = [...nn.lastOutput];
        const hidden = [...nn.lastHidden];
        const hiddenRaw = [...nn.lastHiddenRaw];
        const outputRaw = [...nn.lastOutputRaw];
        const loss = target.reduce((s, t, i) => s - t * Math.log(Math.max(output[i], 1e-10)), 0);
        const predIdx = output.indexOf(Math.max(...output));
        setLastForward({ input: manualInput, hidden, output, hiddenRaw, outputRaw });
        setManualResult({ snap, input: [...manualInput], target, output, hidden, hiddenRaw, outputRaw, loss, predIdx, correct: predIdx === manualTarget });
        setPrevWeights({ w1: null, w2: null });
        setWeightVersion(prev => prev + 1);
    };

    const doManualBackward = () => {
        if (!manualResult) return;
        nn.restore(manualResult.snap);
        nn.forward(manualResult.input);
        nn.backward(manualResult.target);
        const newEpoch = epoch + 1;
        setEpoch(newEpoch);
        const loss = manualResult.loss;
        setHistory(prev => [...prev, { epoch: newEpoch, loss, accuracy: manualResult.correct ? 1 : 0 }]);
        setPrevWeights({ w1: manualResult.snap.w1, w2: manualResult.snap.w2 });
        setUndoStack(prev => [...prev, manualResult.snap]);
        setLastForward({
            input: manualResult.input,
            hidden: [...nn.lastHidden],
            output: [...nn.lastOutput],
            hiddenRaw: [...nn.lastHiddenRaw],
            outputRaw: [...nn.lastOutputRaw],
        });
        const newSnap = nn.snapshot();
        saveEpochToStorage(newEpoch, newSnap, loss, manualResult.correct ? 1 : 0);
        setManualResult({
            ...manualResult,
            loss: nn.lastOutput.reduce((s, t, i) => s - manualResult.target[i] * Math.log(Math.max(nn.lastOutput[i], 1e-10)), 0),
            output: [...nn.lastOutput],
            gradHidden: [...nn.lastGradients.hiddenDelta],
            gradOutput: [...nn.lastGradients.outDelta],
        });
        setWeightVersion(prev => prev + 1);
    };

    const setInputSeg = (idx) => {
        setManualInput(prev => {
            const next = [...prev];
            next[idx] = next[idx] ? 0 : 1;
            return next;
        });
    };

    const setManualInputFromSegments = () => {
        setManualInput(SEG_NAMES.map(k => segments[k] ? 1 : 0));
    };

    const evalData = epoch > 0 ? nn.evaluateAll() : null;
    const currentOutput = lastForward?.output || (epoch > 0 ? nn.lastOutput : null);
    const prediction = currentOutput ? currentOutput.indexOf(Math.max(...currentOutput)) : "-";

    const tabs = [
        { id: "prediction", label: "Prediction" },
        { id: "train", label: "Train" },
        { id: "evaluate", label: "Evaluate" },
        { id: "loss", label: "Expected vs Actual Loss" },
    ];

    return (
        <>
            <div className="top-bar">
                <div className="param-group">
                    <label>LR:<InfoTooltip title="Learning Rate" text={<><p>Controls how big each weight update is after seeing an error.</p><p>• Too high → training overshoots, loss oscillates</p><p>• Too low → training crawls, gets stuck</p><p>• Typical range: <code>0.01 – 0.5</code></p></>} /></label>
                    <input type="range" min="0.001" max="1" step="0.001" value={learningRate} onChange={e => setLearningRate(parseFloat(e.target.value))} />
                    <span className="param-val">{learningRate.toFixed(3)}</span>
                </div>
                <div className="param-group">
                    <label>Hidden:<InfoTooltip title="Hidden Neurons" text={<><p>Number of neurons in the hidden layer.</p><p>• More neurons = more capacity to learn complex patterns</p><p>• Too few → underfitting, can't learn all digits</p><p>• Too many → overfitting, memorizes training data</p><p>• Default <code>8</code> works well for 7-segment digits</p><p>• Changing this resets all weights and clears saved epochs</p></>} /></label>
                    <input type="number" min="4" max="32" value={hiddenSize} onChange={e => setHiddenSize(Math.max(4, Math.min(32, parseInt(e.target.value) || 8)))} />
                </div>
                <div className="param-group">
                    <label>Batch:<InfoTooltip title="Batch Size" text={<><p>Number of training samples processed per auto-train tick.</p><p>• <code>1</code> = pure stochastic gradient descent (noisy but fast)</p><p>• Higher = more stable updates, faster epochs</p><p>• Max <code>10</code> since there are only 10 training samples total</p></>} /></label>
                    <input type="number" min="1" max="10" value={batchSize} onChange={e => setBatchSize(Math.max(1, Math.min(10, parseInt(e.target.value) || 1)))} />
                </div>
                <div className="param-group">
                    <label>Momentum:<InfoTooltip title="Momentum" text={<><p>Adds inertia to weight updates, carrying forward the direction of previous steps.</p><p>• Helps escape shallow local minima and speeds up convergence</p><p>• <code>0</code> = no momentum, pure gradient descent</p><p>• <code>0.9</code> = strong momentum (common in practice)</p><p>• Too high → overshooting, oscillation around the minimum</p></>} /></label>
                    <input type="range" min="0" max="0.99" step="0.01" value={momentum} onChange={e => setMomentum(parseFloat(e.target.value))} />
                    <span className="param-val">{momentum.toFixed(2)}</span>
                </div>
                <div className="param-group">
                    <label>Dropout:<InfoTooltip title="Dropout" text={<><p>Randomly disables a fraction of hidden neurons during training to prevent overfitting.</p><p>• <code>0</code> = no dropout (all neurons active)</p><p>• <code>0.2</code> = 20% of hidden neurons randomly silenced each step</p><p>• Forces the network to learn redundant, robust features</p><p>• Only useful with larger hidden layers (&gt;12 neurons)</p></>} /></label>
                    <input type="range" min="0" max="0.5" step="0.01" value={dropout} onChange={e => setDropout(parseFloat(e.target.value))} />
                    <span className="param-val">{dropout.toFixed(2)}</span>
                </div>
            </div>

            <div className="main-layout">
                <div className="panel">
                    <h2>7-Segment Input</h2>
                    <SevenSegmentDisplay segments={segments} toggleSegment={toggleSegment} />
                    <div style={{ textAlign: "center", marginTop: "16px", display: "flex", gap: "8px", justifyContent: "center" }}>
                        <button className="primary" onClick={runPrediction} disabled={animationProgress > 0} title="Run the current 7-segment pattern through the network and show the predicted digit">Predict</button>
                        <button className="danger" onClick={clearAll} title="Turn off all segments and clear the prediction">Clear</button>
                    </div>
                    <div className="prediction-big">{animationProgress > 0 ? "..." : prediction}</div>
                </div>

                <div className="center-area">
                    <div className="panel network-panel">
                        <h2>Neural Network <span style={{ fontSize: "11px", color: "#8b949e", fontWeight: "normal", float: "right" }}>7 → {nn.hiddenSize} → 10 | {7 * nn.hiddenSize + nn.hiddenSize * 10} weights</span></h2>
                        <NetworkVisualization
                            nn={nn}
                            segments={segments}
                            lastForward={lastForward}
                            inspectTarget={inspectTarget}
                            setInspectTarget={setInspectTarget}
                            animationProgress={animationProgress}
                            weightVersion={weightVersion}
                        />
                    </div>

                    <div className="tabs-bar">
                        {tabs.map(t => (
                            <button key={t.id} className={`graph-tab ${activeTab === t.id ? "active" : ""}`}
                                onClick={() => setActiveTab(t.id)}>
                                {t.label}
                            </button>
                        ))}
                    </div>

                    <div className="panel tab-content">
                        {activeTab === "prediction" && (
                            <>
                                <h2>Predictions — Digits 0-9</h2>
                                {currentOutput ? (
                                    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                                        {currentOutput.map((val, i) => {
                                            const pct = val * 100;
                                            const isWinner = val === Math.max(...currentOutput);
                                            return (
                                                <div key={i} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                                    <span style={{ width: "20px", textAlign: "center", fontWeight: isWinner ? "bold" : "normal", color: isWinner ? "#2ea043" : "#8b949e", fontSize: "14px" }}>{i}</span>
                                                    <div style={{ flex: 1, height: "18px", background: "#0d1117", borderRadius: "4px", overflow: "hidden" }}>
                                                        <div style={{ height: "100%", width: `${pct}%`, background: isWinner ? "linear-gradient(90deg,#238636,#2ea043)" : "#30363d", borderRadius: "4px", transition: "width 0.3s, background 0.3s" }} />
                                                    </div>
                                                    <span style={{ fontSize: "12px", minWidth: "55px", textAlign: "right", color: isWinner ? "#2ea043" : "#8b949e" }}>
                                                        {pct.toFixed(1)}% {isWinner ? "★" : ""}
                                                    </span>
                                                </div>
                                            );
                                        })}
                                    </div>
                                ) : (
                                    <div style={{ textAlign: "center", color: "#8b949e", padding: "20px" }}>
                                        Train the model or click Predict to see outputs
                                    </div>
                                )}
                                <h3 style={{ marginTop: "16px" }}>Loss Curve</h3>
                                <LossCurve history={history} height={100} showAcc={true} />
                            </>
                        )}

                        {activeTab === "train" && (
                            <>
                                <div style={{ fontSize: "10px", color: "#484f58", textAlign: "right", marginBottom: "4px" }}>Keyboard: <code style={{ color: "#8b949e" }}>Space</code> = Auto Train · <code style={{ color: "#8b949e" }}>Shift+←</code> = Undo · <code style={{ color: "#8b949e" }}>Shift+→</code> = Step</div>
                                <div className="train-controls">
                                    <button className="danger" onClick={doUndo} disabled={undoStack.length === 0} title="Revert to previous epoch's weights">← Prev Epoch</button>
                                    <button className="primary" onClick={doTrainStep} title="Train on one random sample (forward + backward)">Next Epoch →</button>
                                    <button className="warning" onClick={() => setAutoTraining(!autoTraining)}>{autoTraining ? "Pause" : "Auto Train"}</button>
                                    <button onClick={skipTraining} title="Run 2000 training steps instantly">Skip Training</button>
                                    <button className="reset" onClick={() => { pushUndoSnap(); nn.initWeights(); setPrevWeights({ w1: null, w2: null }); setWeightVersion(prev => prev + 1); setEditingCell(null); }}>Reset Weights</button>
                                    <button onClick={() => setShowEpochList(!showEpochList)} style={{ fontSize: "11px" }}>
                                        {showEpochList ? "Hide" : "Load"} Epoch ({savedEpochs.length})
                                    </button>
                                    <button onClick={exportWeightsCSV} style={{ fontSize: "11px" }} title="Download all weights and biases as CSV">Export CSV</button>
                                    <label style={{ fontSize: "11px", padding: "6px 16px", background: "#21262d", border: "1px solid #30363d", borderRadius: "6px", cursor: "pointer", color: "#e6edf3" }} title="Upload a previously exported CSV to restore weights">
                                        Import CSV
                                        <input type="file" accept=".csv" onChange={e => { if (e.target.files[0]) importWeightsCSV(e.target.files[0]); e.target.value = ""; }} style={{ display: "none" }} />
                                    </label>
                                    <div className="train-status" title="Current training progress. Loss = cross-entropy error (lower is better). Acc = accuracy on the last sample trained.">
                                        Epoch: <span style={{ color: "#58a6ff" }}>{epoch}</span>
                                        {" "} | Loss: <span style={{ color: "#f85149" }}>{history.length > 0 ? history[history.length - 1].loss.toFixed(4) : "—"}</span>
                                        {" "} | Acc: <span style={{ color: "#2ea043" }}>{history.length > 0 ? (history[history.length - 1].accuracy * 100).toFixed(0) + "%" : "—"}</span>
                                    </div>
                                </div>

                                {showEpochList && (
                                    <div style={{ background: "#0d1117", borderRadius: "8px", padding: "12px", marginBottom: "12px" }}>
                                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                                            <h3 style={{ color: "#58a6ff", marginBottom: 0 }}>Saved Epochs</h3>
                                            <button onClick={clearSavedEpochs} style={{ fontSize: "10px", padding: "2px 8px", background: "#da3633", border: "#f85149" }}>Clear All</button>
                                        </div>
                                        <div style={{ maxHeight: "200px", overflowY: "auto" }}>
                                            {savedEpochs.length === 0 ? (
                                                <div style={{ color: "#8b949e", fontSize: "12px", padding: "12px", textAlign: "center" }}>No saved epochs yet. Train the model to create checkpoints.</div>
                                            ) : (
                                                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: "6px" }}>
                                                    {savedEpochs.map((e, i) => (
                                                        <div key={i} onClick={() => loadEpoch(e)}
                                                            style={{
                                                                background: e.epoch === epoch ? "#238636" : "#161b22",
                                                                border: `1px solid ${e.epoch === epoch ? "#2ea043" : "#30363d"}`,
                                                                borderRadius: "6px",
                                                                padding: "6px 8px",
                                                                cursor: "pointer",
                                                                fontSize: "11px",
                                                                transition: "background 0.2s",
                                                            }}>
                                                            <div style={{ fontWeight: "bold", color: "#58a6ff" }}>Epoch {e.epoch}</div>
                                                            <div style={{ color: "#f85149", fontSize: "10px" }}>Loss: {e.loss.toFixed(4)}</div>
                                                            <div style={{ color: "#2ea043", fontSize: "10px" }}>Acc: {(e.accuracy * 100).toFixed(0)}%</div>
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}

                                <div style={{ background: "#0d1117", borderRadius: "8px", padding: "12px", marginBottom: "12px" }}>
                                    <h3 style={{ color: "#58a6ff", marginBottom: "8px" }}>Manual Forward / Backward Pass <InfoTooltip title="How Neural Networks Learn" text={<><p>Training has two alternating steps:</p><p><strong style={{ color: "#58a6ff" }}>1. Forward Pass</strong> — Feed input through the network to get a prediction. Data flows left → right: segments → hidden neurons → output probabilities.</p><p><strong style={{ color: "#d29922" }}>2. Backward Pass</strong> — Compare prediction to the correct answer, compute gradients (how much each weight contributed to the error), then update weights to reduce that error next time.</p><p>These two steps repeated thousands of times is how a network learns. Each pair = <code>1 epoch</code>.</p></>} /></h3>
                                    <div style={{ display: "flex", gap: "16px", alignItems: "flex-start", flexWrap: "wrap" }}>
                                        <div>
                                            <div style={{ fontSize: "11px", color: "#8b949e", marginBottom: "4px" }}>Input [a b c d e f g]</div>
                                            <div style={{ display: "flex", gap: "4px" }}>
                                                {manualInput.map((v, i) => (
                                                    <button key={i} onClick={() => setInputSeg(i)}
                                                        style={{
                                                            width: "32px", height: "32px", padding: 0, fontSize: "14px", fontWeight: "bold",
                                                            background: v ? "#238636" : "#21262d", border: v ? "#2ea043" : "#30363d",
                                                            color: v ? "#fff" : "#8b949e", borderRadius: "4px"
                                                        }}>
                                                        {v}
                                                    </button>
                                                ))}
                                            </div>
                                            <button onClick={setManualInputFromSegments} style={{ marginTop: "4px", fontSize: "10px", padding: "2px 8px" }}>
                                                Use 7-segment input
                                            </button>
                                        </div>
                                        <div>
                                            <div style={{ fontSize: "11px", color: "#8b949e", marginBottom: "4px" }}>Target Digit</div>
                                            <div style={{ display: "flex", gap: "3px" }}>
                                                {Array.from({ length: 10 }).map((_, i) => (
                                                    <button key={i} onClick={() => setManualTarget(i)}
                                                        style={{
                                                            width: "28px", height: "28px", padding: 0, fontSize: "12px", fontWeight: "bold",
                                                            background: manualTarget === i ? "#58a6ff" : "#21262d",
                                                            border: manualTarget === i ? "#58a6ff" : "#30363d",
                                                            color: manualTarget === i ? "#0d1117" : "#8b949e", borderRadius: "4px"
                                                        }}>
                                                        {i}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>
                                        <div style={{ display: "flex", gap: "8px", alignItems: "flex-end" }}>
                                            <button className="primary" onClick={doManualForward}>Forward Pass <InfoTooltip title="Forward Pass" text={<><p>Feeds the current input through every layer of the network:</p><p><code>input × w1 + b1 → sigmoid → hidden</code></p><p><code>hidden × w2 + b2 → softmax → output</code></p><p>Shows the prediction probabilities for all 10 digits and the cross-entropy loss (how far the prediction is from the target). <em>Does not change any weights.</em></p></>} /></button>
                                            <button className="warning" onClick={doManualBackward} disabled={!manualResult}>Backward Pass <InfoTooltip title="Backward Pass (Backpropagation)" text={<><p>Calculates how much each weight contributed to the error and updates them:</p><p>1. Compute <strong>output gradients</strong>: <code>prediction - target</code></p><p>2. Propagate gradients <strong>backward</strong> through w2 to hidden layer</p><p>3. Compute <strong>hidden gradients</strong> via chain rule</p><p>4. Update all weights using gradients + momentum</p><p>After this, the network is slightly better at this input. The epoch counter increments by 1.</p></>} /></button>
                                        </div>
                                    </div>

                                    {manualResult && (
                                        <div style={{ marginTop: "12px", fontSize: "12px" }}>
                                            <div style={{ display: "flex", gap: "16px", marginBottom: "8px", flexWrap: "wrap" }}>
                                                <div>Loss: <span style={{ color: "#f85149", fontWeight: "bold" }}>{manualResult.loss.toFixed(4)}</span></div>
                                                <div>Predicted: <span style={{ color: "#58a6ff", fontWeight: "bold" }}>{manualResult.predIdx}</span>
                                                    {" "}{manualResult.correct ? "✓" : "✗"}
                                                </div>
                                            </div>
                                            <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                                                {manualResult.output.map((v, i) => (
                                                    <div key={i} style={{ textAlign: "center", minWidth: "40px" }}>
                                                        <div style={{ fontSize: "16px", fontWeight: "bold", color: i === manualResult.predIdx ? "#2ea043" : "#8b949e" }}>{i}</div>
                                                        <div style={{ fontSize: "10px", color: "#8b949e" }}>{(v * 100).toFixed(1)}%</div>
                                                        {manualResult.target[i] === 1 && <div style={{ fontSize: "8px", color: "#58a6ff" }}>target</div>}
                                                    </div>
                                                ))}
                                            </div>
                                            {manualResult.gradHidden && (
                                                <>
                                                    <div style={{ marginTop: "8px", fontSize: "11px", color: "#8b949e" }}>Gradients after backward pass:</div>
                                                    <div style={{ display: "flex", gap: "4px", flexWrap: "wrap", marginTop: "4px" }}>
                                                        {manualResult.gradOutput.map((g, i) => (
                                                            <div key={`go-${i}`} style={{ fontSize: "10px", minWidth: "40px", textAlign: "center", color: Math.abs(g) < 0.01 ? "#2ea043" : "#f85149" }}>
                                                                out[{i}]: {g.toFixed(4)}
                                                            </div>
                                                        ))}
                                                    </div>
                                                    <div style={{ display: "flex", gap: "4px", flexWrap: "wrap", marginTop: "4px" }}>
                                                        {manualResult.gradHidden.map((g, i) => (
                                                            <div key={`gh-${i}`} style={{ fontSize: "10px", minWidth: "40px", textAlign: "center", color: Math.abs(g) < 0.01 ? "#2ea043" : "#f85149" }}>
                                                                hid[{i}]: {g.toFixed(4)}
                                                            </div>
                                                        ))}
                                                    </div>
                                                </>
                                            )}
                                        </div>
                                    )}
                                </div>

                                <WeightMatrixView
                                    w1={nn.w1} w2={nn.w2} prevW1={prevWeights.w1} prevW2={prevWeights.w2}
                                    b1={nn.b1} b2={nn.b2}
                                    setWeight={setWeight} setBias={setBias}
                                    label="Click any value to edit — changes saved to undo stack"
                                    editingCell={editingCell} setEditingCell={setEditingCell}
                                    editValue={editValue} setEditValue={setEditValue}
                                />
                                <h3 style={{ marginTop: "12px" }}>Loss Curve</h3>
                                <LossCurve history={history} height={100} showAcc={true} />
                            </>
                        )}

                        {activeTab === "evaluate" && (
                            <>
                                {evalData ? (
                                    <>
                                        <div style={{ display: "flex", gap: "20px", marginBottom: "12px", fontSize: "14px" }}>
                                            <div>Accuracy: <span style={{ color: "#2ea043", fontWeight: "bold" }}>{(evalData.avgAcc * 100).toFixed(0)}%</span></div>
                                            <div>Avg Loss: <span style={{ color: "#f85149", fontWeight: "bold" }}>{evalData.avgLoss.toFixed(4)}</span></div>
                                            <div>Correct: <span style={{ color: "#58a6ff", fontWeight: "bold" }}>{evalData.results.filter(r => r.correct).length}/10</span></div>
                                        </div>
                                        <div className="eval-grid">
                                            {evalData.results.map(r => {
                                                const maxLoss = Math.max(0.01, ...evalData.results.map(x => x.loss));
                                                const lossH = Math.max(4, (r.loss / maxLoss) * 40);
                                                return (
                                                    <div key={r.digit} className="eval-card" style={{
                                                        background: r.correct ? "rgba(46,160,67,0.1)" : "rgba(248,81,73,0.1)",
                                                        border: `1px solid ${r.correct ? "#2ea043" : "#f85149"}`,
                                                    }}>
                                                        <div style={{ fontSize: "24px", fontWeight: "bold", color: r.correct ? "#2ea043" : "#f85149" }}>{r.digit}</div>
                                                        <div style={{ width: "100%", height: "40px", background: "#0d1117", borderRadius: "4px", margin: "6px 0", position: "relative", overflow: "hidden" }}>
                                                            <div style={{ position: "absolute", bottom: 0, left: 0, width: "100%", height: `${lossH}px`, background: r.correct ? "rgba(46,160,67,0.3)" : "rgba(248,81,73,0.3)", borderRadius: "4px" }} />
                                                        </div>
                                                        <div style={{ fontSize: "10px", color: "#8b949e", marginTop: "2px" }}>Predicted: <span style={{ fontWeight: "bold", color: "#58a6ff" }}>{r.predIdx}</span></div>
                                                        <div style={{ fontSize: "10px", color: "#8b949e" }}>Conf: <span style={{ fontWeight: "bold" }}>{(r.output[r.predIdx] * 100).toFixed(1)}%</span></div>
                                                        <div style={{ fontSize: "9px", color: "#8b949e" }}>Loss: {r.loss.toFixed(4)}</div>
                                                        <div style={{ fontSize: "16px", marginTop: "2px" }}>{r.correct ? "✓" : "✗"}</div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </>
                                ) : (
                                    <div style={{ textAlign: "center", color: "#8b949e", padding: "20px" }}>Train the model first to see evaluation</div>
                                )}
                                <h3 style={{ marginTop: "16px" }}>Loss Curve</h3>
                                <LossCurve history={history} height={100} showAcc={true} />
                            </>
                        )}

                        {activeTab === "loss" && (
                            <>
                                {evalData ? (
                                    <>
                                        <div style={{ fontSize: "13px", color: "#8b949e", marginBottom: "12px" }}>
                                            Total Loss: <span style={{ color: "#f85149", fontWeight: "bold" }}>{evalData.avgLoss.toFixed(4)}</span>
                                        </div>
                                        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                                            {evalData.results.map(r => {
                                                const expected = r.target[r.digit];
                                                const actual = r.output[r.digit];
                                                const eH = expected * 80;
                                                const aH = actual * 80;
                                                return (
                                                    <div key={r.digit} style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                                                        <span style={{ width: "20px", textAlign: "center", fontWeight: "bold", fontSize: "14px", color: r.correct ? "#2ea043" : "#f85149" }}>{r.digit}</span>
                                                        <div className="loss-bar-container">
                                                            <div style={{ position: "absolute", bottom: 0, left: 0, width: "50%", height: `${eH}%`, background: "#8b949e", opacity: 0.5, borderRadius: "4px 0 0 4px" }} />
                                                            <div style={{ position: "absolute", bottom: 0, right: 0, width: "50%", height: `${aH}%`, background: r.correct ? "#2ea043" : "#f85149", borderRadius: "0 4px 4px 0" }} />
                                                        </div>
                                                        <span style={{ fontSize: "11px", color: "#8b949e", minWidth: "80px" }}>Actual: {(actual * 100).toFixed(1)}%</span>
                                                        <span style={{ fontSize: "11px", color: r.loss < 0.1 ? "#2ea043" : "#f85149", minWidth: "70px", textAlign: "right" }}>Loss: {r.loss.toFixed(4)}</span>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </>
                                ) : (
                                    <div style={{ textAlign: "center", color: "#8b949e", padding: "20px" }}>Train the model first to see expected vs actual</div>
                                )}
                                <h3 style={{ marginTop: "16px" }}>Loss Curve</h3>
                                <LossCurve history={history} height={100} showAcc={true} />
                            </>
                        )}
                    </div>
                </div>
            </div>

            {inspectTarget && (inspectTarget.type === "neuron" || inspectTarget.type === "connection") && (() => {
                const vw = window.innerWidth;
                const vh = window.innerHeight;
                const pw = 200;
                const ph = inspectTarget.type === "neuron" ? 120 : 90;
                let left = inspectTarget.px + 20;
                let top = inspectTarget.py - 40;
                if (left + pw > vw - 10) left = inspectTarget.px - pw - 20;
                if (top + ph > vh - 10) top = vh - ph - 10;
                if (top < 10) top = 10;
                if (left < 10) left = 10;
                return (
                    <>
                        <div style={{ position: "fixed", inset: 0, zIndex: 99 }} onClick={() => setInspectTarget(null)} />
                        {inspectTarget.type === "neuron" && (
                            <div className="inspect-popup" style={{ top, left }}>
                                <div><span className="label">Layer:</span> <span className="value">{inspectTarget.layer === 1 ? "Hidden" : "Output"}</span></div>
                                <div><span className="label">Neuron:</span> <span className="value">#{inspectTarget.idx}</span></div>
                                <div><span className="label">Bias:</span> <span className="value">{inspectTarget.bias?.toFixed(4)}</span></div>
                                <div><span className="label">Weighted Sum:</span> <span className="value">{inspectTarget.raw?.toFixed(4)}</span></div>
                                <div><span className="label">Activation:</span> <span className="value">{inspectTarget.activation?.toFixed(4)}</span></div>
                            </div>
                        )}
                        {inspectTarget.type === "connection" && (
                            <div className="inspect-popup" style={{ top, left: Math.max(10, left - 20) }}>
                                <div><span className="label">Connection:</span> <span className="value">{inspectTarget.from} → {inspectTarget.to}</span></div>
                                <div><span className="label">Weight:</span> <span className="value">{inspectTarget.weight?.toFixed(4)}</span></div>
                                <div><span className="label">Contribution:</span> <span className="value">{inspectTarget.contribution?.toFixed(4)}</span></div>
                            </div>
                        )}
                    </>
                );
            })()}
        </>
    );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
