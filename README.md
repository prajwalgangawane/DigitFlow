# DigitFlow

An interactive, browser-based neural network trainer & visualizer for 7-segment LED digit classification.

## What It Does

Trains a neural network to recognize digits (0-9) from 7-segment display inputs — all rendered and trained live in your browser with zero dependencies, no build tools, no servers.

## Features

- **Live Training**: Click through epochs one-by-one, auto-train continuously, or skip 2000 steps
- **Visual Network Architecture**: Dynamic SVG rendering of neurons and weighted connections that updates in real-time
- **Forward/Backward Pass Stepping**: Run each step manually, inspect activations, biases, and gradients
- **Direct Weight Editing**: Click any weight or bias cell to modify it; changes are color-coded (green = increased, red = decreased)
- **Click-to-Inspect**: Click any neuron or connection to see its activation, weight, and contribution
- **Undo/Epoch History**: Step back through every training epoch with full weight/bias/momentum snapshots
- **Checkpoint Persistence**: Every epoch is saved to localStorage; restore any checkpoint at any time
- **CSV Export/Import**: Save weights to CSV and reload them later
- **Per-Digit Evaluation**: Accuracy grid with per-digit loss bars showing where the network struggles
- **Configurable Architecture**: Adjust hidden layer size (4-32), learning rate, batch size, momentum, and dropout

## Tech Stack

- React 18 + ReactDOM via CDN
- Babel standalone for JSX transformation
- Zero build tools, zero npm packages — just `index.html` + `app.js`
- Neural network from scratch: sigmoid hidden layer, softmax output, cross-entropy loss, momentum SGD, optional dropout

## Architecture

```
7 Inputs (Segments) → N Hidden Neurons (Sigmoid) → 10 Outputs (Softmax)
```

Default: 7 → 8 → 10 (86 trainable parameters)

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Toggle Auto Train |
| `Shift + →` | Step forward one epoch |
| `Shift + ←` | Undo last epoch |

## Getting Started

Open `index.html` in any modern browser. No installation needed.
