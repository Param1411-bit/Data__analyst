# 🔬 EDA Dashboard — Setup & Run Guide

## Quick Start (3 commands)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start Ollama (in a separate terminal)
ollama serve
ollama pull llama3          # or llama3.2, mistral, qwen2.5, etc.

# 3. Launch the dashboard
streamlit run eda_dashboard.py
```

Then open **localhost** in your browser.

---

## Features

| Tab | What it does |
|-----|-------------|
| 📋 First Look | Preview rows, dtypes, describe(), memory usage |
| 📊 EDA | Distributions, Q-Q plots, scatter plots, skewness table |
| 🩺 Data Quality | Missing values heatmap, duplicates, dtype issues, cardinality |
| 🧹 Cleaning Report | Before/after comparison, download cleaned CSV |
| ⚡ Outliers | IQR + Z-score table, boxplots, Isolation Forest |
| 🔗 Correlation | Heatmap, high-corr pair table, pair plots |
| 🤖 AI Insights | 6 preset prompts + custom — powered by local Ollama |

---

## Ollama Models (recommended)

| Model | Command | Notes |
|-------|---------|-------|
| LLaMA 3 8B | `ollama pull llama3` | Best balance of speed & quality |
| LLaMA 3.2 3B | `ollama pull llama3.2` | Faster, lighter |
| Mistral 7B | `ollama pull mistral` | Great for structured output |
| Qwen 2.5 7B | `ollama pull qwen2.5` | Strong on data/code tasks |
| DeepSeek-R1 7B | `ollama pull deepseek-r1` | Reasoning model |

---

## Sidebar Options

- **Upload**: CSV or Excel (.xlsx / .xls)
- **Ollama host**: Default `http://localhost:11434`
- **Model**: Name of any pulled Ollama model
- **Cleaning options**: Applied live to all tabs
  - Remove duplicates
  - Missing value strategy (drop, mean, median, mode, KNN)
  - Outlier treatment (flag, winsorize, drop)

---

## File Structure

```
.
├── eda_dashboard.py     # Main app
├── requirements.txt     # Dependencies
└── README.md            # This file
```

---

## Troubleshooting

**"Could not connect to Ollama"**
→ Run `ollama serve` in a terminal first

**"Model not found"**
→ Run `ollama pull <model_name>` first

**Slow on large datasets**
→ The AI insight tab sends only a compact text profile, not raw data — it stays fast regardless of dataset size.

**Excel files not loading**
→ Ensure `openpyxl` is installed: `pip install openpyxl`
