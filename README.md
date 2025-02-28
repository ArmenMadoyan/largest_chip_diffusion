# Bass Diffusion Model for AI Chip Market

This repository contains Python code that:
1. Reads AI chip market revenue data (sourced from [Statista](https://www.statista.com/topics/6153/artificial-intelligence-ai-chips)) for the years 2023 to 2025.
2. Estimates the Bass Diffusion Model parameters (`p`, `q`, and `M`) via nonlinear curve fitting.
3. Predicts future adoption (or revenue) and visualizes both cumulative and annual adopters.

## Requirements

- Python 3.9 or higher
- `numpy`
- `matplotlib`
- `scipy`
- `pandas` (if reading from Excel or CSV files)

You can install these with:
```bash
pip install numpy matplotlib scipy pandas
