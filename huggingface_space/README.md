---
title: INDIGO — Flexible-Material RGB → Structure
emoji: 🎨
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Thin-film stack designer that hits a target color
---

# INDIGO — Hosted Demo

A cross-attention model that takes a target color (free-text prompt or CIE Lab) plus a pool of optical materials and returns a layered thin-film stack that hits the target color, with physics-based ΔE validation.

This Space hosts the same FastAPI server + static frontend that runs locally — see the [main repo](https://github.com/arikalinsky14/INDIGO) for training code, dataset generation, and the inference library (`solve(model, pool, spec) -> Result`).

## What you can try
- Free-text prompts: *"a deep crimson with no silver, between 3 and 5 layers"*
- Structured input: CIE Lab + a JSON constraint array
- Material picker: any subset of the JLL library (≤ 32) or upload a CSV of n/k spectra
- Speed presets: **fast** (~5–10 s), **balanced** (~15–25 s), **best** (~2–5 min)

## How it works
See the **"how it works ⓘ"** button in the UI for the 7-stage pipeline reference. Short version:

```
parse prompt → encode pool → sample N candidates → simulate ΔE
 → score + filter → refine top-k → MC robustness → rank → return
```

The model runs on CPU (HF Spaces free tier). Cold start on first request takes ~30 s while JAX warms up; subsequent requests are seconds to minutes depending on the preset.

## OpenAI key
The natural-language Prompt tab uses `gpt-4o-mini` for parsing. Either:
- paste your own key into the Advanced settings panel in the UI (stored only in your browser's localStorage; sent only with prompt requests), or
- skip the Prompt tab and use the Structured tab (no key needed).

If the Space owner has set `OPENAI_API_KEY` as a Space Secret, it's used automatically and you don't need to paste one. The hosted prompt-parser call costs ~$0.001 per request at current rates.
