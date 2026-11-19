# Kathmandu PM2.5 Forecasting

MSc thesis code. Forecasts PM2.5 at +24, +48 and +72 hours and the health-risk class for four stations in Kathmandu Valley (Bhaktapur, Khumaltar, Ratnapark, Shankhapark), using LSTM and GNN-LSTM.

Data is hourly PM2.5 from the Department of Environment, 2024 to 2025. The data is not in this repo.

## Files

- `index.ipynb` - data checks, EDA, train/val/test split, features, LSTM classification and regression, Optuna tuning
- `gnn_lstm.ipynb` - GNN-LSTM, LSTM vs GNN-LSTM over 3 seeds, graph removed ablation
- `results_summary.ipynb` - final tables, RMSE, significance tests, persistence baseline
- `dashboard.py` - Streamlit dashboard

## How to run

1. Put the station CSV files in `AQI_DATA_024_025/024/` and `AQI_DATA_024_025/025/`
2. `pip install -r requirements.txt`
3. Run `index.ipynb`, then `gnn_lstm.ipynb`, then `results_summary.ipynb`
4. `streamlit run dashboard.py`

All outputs go to `processed/`.

The models were trained once and saved. With `RETRAIN = False` (in the config cell) the notebooks load the saved models instead of training again. Set `RETRAIN = True` to train everything from the start. This takes around 6 to 8 hours on CPU.

## Targets

- Regression: hourly PM2.5 at t+24, t+48, t+72
- Classification: EPA class of the 24 hour average PM2.5 at the same times (4 classes, class 4 is very unhealthy and above)

## Main results (test set, mean of 3 seeds)

| Model | MAE (µg/m³) | Macro-F1 |
|---|---:|---:|
| LSTM (per station) | 14.21 | 0.648 |
| GNN-LSTM | 14.77 | 0.664 |
| Shared model without graph | 13.84 | 0.653 |
| Persistence (no model) | 14.45 | 0.688 |

- The graph did not help. The same model without the graph had lower error.
- No model beat persistence by a significant margin on MAE. The models do better at +72 h.
- Persistence has higher macro-F1, but it catches only about 62% of class 4 hours. The models catch almost all of them.
- Almost all class 4 hours come from one pollution episode (March to April 2025), so the class 4 results need care.
