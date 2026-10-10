# ocr-capture

Screenshot → crop → OCR → translate. Two modes:

## GUI (default)

```
uv run python -m src.ui
```

- Select OCR engine from dropdown (meiki, yomitoku, ollama, mistral, deepseek)
- Select translation engine: Hy-MT2 (default) or TranslateGemma (Ollama)
- Click **CAPTURE** or press **Ctrl+S** to grab the screen
- Drag a crop region on the overlay; **Escape** cancels
- First Hy-MT2 use builds a pinned llama.cpp revision with Git, CMake, and a C++ compiler, then downloads and verifies the 462 MB STQ model. The app starts and stops the local runtime automatically; failures appear in Status without falling back.

Windows only (Win32 global hotkey).

The STQ kernel is not yet in llama.cpp mainline. The app builds the pinned source revision from [PR #22836](https://github.com/ggml-org/llama.cpp/pull/22836); on Windows, install Visual Studio Build Tools with the C++ workload before selecting Hy-MT2.

The pinned STQ dot-product optimization is ARM NEON; Windows x64 uses the generic CPU path and may be slower.

## CLI

```
uv run python main.py
```

Edit `OCR = "..."` in `main.py` to pick engine. Clicks a predefined region.

The CLI also defaults to Hy-MT2.
