# ForecastOS Deployment

## Streamlit Community Cloud

Use:

- Repository branch: `main`
- Entrypoint: `streamlit_app.py`
- Dependency file: root `requirements.txt`

The default dependency file includes PyTorch so Deep MLP, LSTM, TCN and Transformer are available from the Deep Learning Lab.

## Resource policy

ForecastOS protects interactive deployments in several ways:

- uploaded datasets are capped to the most recent 250,000 rows in the public-demo UI
- Deep Learning Lab is opt-in
- deep sequence training windows are capped
- training sample counts are capped by search mode
- PyTorch thread count is bounded
- validation-tail early stopping is used
- model failures are isolated and do not terminate the whole tournament

For production traffic or larger datasets, split the architecture:

```text
Streamlit UI
    │
    ├── lightweight profiling / visualization
    │
    ▼
Forecast API / worker queue
    │
    ├── classical + ML CPU workers
    ├── deep-learning workers
    └── foundation-model GPU service
```

## Lightweight deployment

`requirements-lite.txt` omits PyTorch. With that environment the interface still supports all classical/ML models and Deep MLP, while automatically hiding LSTM/TCN/Transformer.

## Foundation models

Do not add large foundation-model checkpoints directly to the public Streamlit cold-start path. Use `requirements-foundation.txt` in a separate service or environment and expose the model through an adapter/API.

## Secrets

For the optional AI analyst, configure:

```toml
OPENAI_API_KEY="..."
```

Never commit API keys to the repository.

## Pre-deployment checks

```bash
python -m compileall streamlit_app.py src
pytest -q
```

After deployment, manually check:

- keyboard navigation and focus visibility
- 200% browser zoom
- mobile/narrow layout
- file upload failure messages
- stale-result warning after changing configuration
- deep-model on/off behavior
- model failure disclosure
- AI analyst behavior with and without a configured key
