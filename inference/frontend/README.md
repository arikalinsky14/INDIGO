# INDIGO frontend

Single-page UI + FastAPI server over `inference.src.solve.solve()`. All
aesthetic assets live in `static/`; the API is a thin glue layer.

## Layout

```
inference/frontend/
├── README.md           you are here
├── requirements.txt    fastapi + uvicorn + pydantic
├── server.py           FastAPI app + entry point
└── static/             ← every visual asset lives here
    ├── index.html
    ├── styles.css
    └── app.js
```

The pure-function boundary is `solve(model, pool, spec) → Result`. The
server wraps it; the static page renders the result.

## Install (one-time, into the existing env)

```bash
source $HOME/envs/llm-env/bin/activate
pip install -r inference/frontend/requirements.txt
```

`torch`, `jax`, `jaxlayerlumos`, `numpy`, `matplotlib` are already in
that env from training / inference. `openai` is optional — without
`OPENAI_API_KEY` the LLM parse falls back to the keyword `mock` backend.

## Run

```bash
export OPENAI_API_KEY=sk-...    # optional; mock backend works without
python -m inference.frontend.server \
    --checkpoint data/checkpoints/<tag>/latest \
    --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

### Running on a compute node + SSH tunneling

```bash
# on your laptop
ssh -L 8000:<node>:8000 user@cluster

# on the cluster — submit an interactive session and run inside it
srun --gres=gpu:1 --pty bash
cd /ix1/.../INDIGO
python -m inference.frontend.server \
    --checkpoint <ckpt> --host 0.0.0.0 --port 8000
```

## API

| Method | Path           | Body / Notes |
|--------|----------------|--------------|
| GET    | `/api/status`  | loaded checkpoint, pool fingerprint, device, OpenAI key flag |
| GET    | `/api/pool`    | canonical-name list of the loaded pool |
| POST   | `/api/solve`   | `{ prompt: str }` or `{ target_lab: [L,a,b], constraints?: [...] }`, plus optional `knobs: {...}` |
| GET    | `/`            | UI (static/index.html) |
| GET    | `/styles.css`, `/app.js` | static assets |

`POST /api/solve` returns the same JSON envelope the CLI persists as
`result_seed<N>.json` — see `inference/src/schema.py:Result`.

## UI notes

- Two input modes: free-text **Prompt** (LLM parsed) or structured
  **Lab + JSON constraints**. Tabs switch between them.
- Live sRGB preview as you type Lab values.
- Advanced knobs (ensemble N, temperature, tolerance, λ, top-k, seed)
  collapsed by default.
- Result panel: target / achieved swatches with a centre ΔE callout,
  SVG reflectance chart with the chosen line + faint top-k alternatives,
  proportional layer-bar stack diagram with hover tooltips, robustness
  stats, and a per-run details JSON drawer.
- Modern dark theme. Tailwind via CDN for utility classes; bespoke
  bits (chart / bars / glass cards) in `styles.css`.
- Zero build step. Pure vanilla JS. Open the files; edit; reload.

## Deferred to a follow-up

- Streaming progress over SSE so long solves show "decoding step k/L".
- Pool inspector (browse canonical names, n,k previews per material).
- Run history persisted to `localStorage`.
- Light-theme toggle.
- Mobile-tuned breakpoints (current layout is desktop-first).
