#!/usr/bin/env bash
# Rebuild static/tailwind.css from the classes index.html and app.js use.
# Needs Node; run after changing classes in either file.
set -euo pipefail
cd "$(dirname "$0")"
npx --yes tailwindcss@3.4.17 -c tailwind.config.js -i tailwind.input.css -o static/tailwind.css --minify
