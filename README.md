# ocr-capture

Screenshot → crop → OCR → translate. Two modes:

## GUI (default)

```
uv run python -m src.ui
```

- Select OCR engine from dropdown (meiki, yomitoku, ollama, mistral, deepseek)
- Click **CAPTURE** or press **Alt+Shift+T** to grab the screen
- Drag a crop region on the overlay; **Escape** cancels
- Result is OCR'd and translated via Gemma (Ollama)

Windows only (Win32 global hotkey).

## CLI

```
uv run python main.py
```

Edit `OCR = "..."` in `main.py` to pick engine. Clicks a predefined region.
