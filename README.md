# ocr-capture

Screenshot → crop → OCR → translate. Two modes:

## GUI (default)

```
uv run ocr-capture
```

- Select OCR engine from dropdown (meiki, yomitoku, ollama, mistral, deepseek)
- GUI uses Index-Translate-2B by default; translation engine selection is fixed.
- Click **CAPTURE** or press **Ctrl+S** to grab the screen
- Drag a crop region on the overlay; **Escape** cancels
- First use pulls the recommended Q4_K_M model through Ollama and caches it there. Ollama must be installed and running; errors appear in Status without fallback.

Windows only (Win32 global hotkey).

## CLI

```
uv run python main.py
```

Edit `OCR = "..."` in `main.py` to pick engine. Clicks a predefined region.

The CLI also defaults to Index-Translate-2B.
