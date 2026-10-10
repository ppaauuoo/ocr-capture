# ocr-capture

Screenshot → crop → OCR → translate. Two modes:

## GUI (default)

```
uv run ocr-capture
```

- Select OCR engine from dropdown (meiki, yomitoku, ollama, mistral, deepseek)
- Select translation engine: Hy-MT2 (default), TranslateGemma (Ollama), or Index-Translate-2B
- Click **CAPTURE** or press **Ctrl+S** to grab the screen
- Drag a crop region on the overlay; **Escape** cancels
- First Hy-MT2 use builds its pinned llama.cpp revision and verifies the 462 MB STQ model. First Index-Translate-2B use builds a pinned upstream llama.cpp revision and downloads and verifies the recommended 1.31 GB Q4_K_M model. Both local runtimes start automatically and reuse cached files; initialization and translation errors appear in Status without fallback.

Windows only (Win32 global hotkey).

The app builds llama.cpp from source; on Windows, install Git, CMake, and Visual Studio Build Tools with the C++ workload before selecting either local model. Hy-MT2 also needs the STQ-capable fork because that kernel is not in llama.cpp mainline; its ARM NEON optimization uses the generic CPU path on Windows x64 and may be slower.

## CLI

```
uv run python main.py
```

Edit `OCR = "..."` in `main.py` to pick engine. Clicks a predefined region.

The CLI also defaults to Hy-MT2.
