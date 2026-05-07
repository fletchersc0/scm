# Hidden Machine browser integration

This folder contains both parts of the project:

1. **Python modelling/generation code**: `run_models.py` and the modelling modules.
2. **Static browser experiment**: `index.html` plus adjacent `machine_world_trials.json`.

The browser does not run any Python code. The Python package is used offline to generate `machine_world_trials.json`; participants only need the static browser files.

## Generate the browser corpus from the Python package

```bash
python build_machine_world_trials.py \
  --seed 2026 \
  --n_episodes 48 \
  --bag_search_limit 5000 \
  --alpha 1.0 \
  --out machine_world_trials.json
```

The generated JSON is consumed by `index.html`.

## Run the browser experiment locally

From this folder:

```bash
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000/index.html?PROLIFIC_PID=test_pid&condition=mixed
```

For the built-in short interface demo:

```text
http://localhost:8000/index.html?demo=1&debug=1&PROLIFIC_PID=test_pid
```

Do not double-click `index.html` for normal mode, because browsers often block `fetch()` from loading the adjacent JSON file under `file://`.

## Run the model-analysis outputs

```bash
python run_models.py \
  --seed 2026 \
  --n_episodes 48 \
  --bag_search_limit 5000 \
  --beam_width 20 \
  --alpha 1.0 \
  --lambda_complexity 1.0 \
  --association_decay 0.95 \
  --checkpoints 12 30 48 \
  --outdir outputs
```

`run_models.py` writes analysis CSVs and plots under `outputs/`. It is complementary to, but not the same as, `build_machine_world_trials.py`.

## Deployment notes

Before real deployment, replace the placeholder values in `index.html` for:

- `CONFIG.dataPipeID`
- `CONFIG.prolificCompletionURL`
- `CONFIG.prolificScreenoutURL`

Keep `index.html` and `machine_world_trials.json` in the same deployed folder.
