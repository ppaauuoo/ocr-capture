"""PySide6 UI: capture screen → drag-crop → OCR → translate."""
import sys
import threading
from PIL import Image

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QComboBox, QLabel, QTextEdit,
)
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush

from src.capture import Capture
from src.translate.ollama import TranslateGemma

# ── OCR registry ──────────────────────────────────────────
OCR_OPTIONS = {
    "meiki": "src.ocr.meiki_ocr:MeikiOCREngine",
    "yomitoku": "src.ocr.yomitoku_ocr:YomiTokuOCR",
    "ollama": "src.ocr.ollama_vlm:OllamaVisionOCR",
    "mistral": "src.ocr.mistral:MistralOCR",
    "deepseek": "src.ocr.deepseekocr2:DeepseekOCR2",
}


def _load_ocr(name: str):
    path = OCR_OPTIONS[name]
    mod_path, cls_name = path.split(":")
    import importlib
    return getattr(importlib.import_module(mod_path), cls_name)()


# ── Overlay ───────────────────────────────────────────────
class OverlayWindow(QWidget):
    crop_done = Signal(int, int, int, int)

    def __init__(self, pixmap: QPixmap, geo, parent=None):
        super().__init__(parent)
        self._pixmap = pixmap
        self._drag_start = None
        self._drag_end = None
        self._dragging = False

        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setGeometry(geo)
        self.setCursor(Qt.CrossCursor)

    def showEvent(self, event):
        self.showFullScreen()

    def paintEvent(self, event):
        painter = QPainter(self)
        # Fill background, then draw screenshot scaled to fill the window
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        painter.drawPixmap(self.rect(), self._pixmap)

        if self._drag_start and self._drag_end:
            rect = QRectF(self._drag_start, self._drag_end).normalized()
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setBrush(QBrush(QColor(0, 255, 0, 30)))
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.position()
            self._drag_end = event.position()
            self._dragging = True

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._drag_end = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            if self._drag_start and self._drag_end:
                r = QRectF(self._drag_start, self._drag_end).normalized().toRect()
                if r.width() > 10 and r.height() > 10:
                    sx = self._pixmap.width() / self.width()
                    sy = self._pixmap.height() / self.height()
                    self.crop_done.emit(
                        int(r.x() * sx), int(r.y() * sy),
                        int(r.right() * sx), int(r.bottom() * sy),
                    )
                    self.close()
                    return
            self.close()


# ── Main window ───────────────────────────────────────────
class MainWindow(QMainWindow):
    status_changed = Signal(str)
    result_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self._ocr = None
        self._ocr_name = "meiki"

        self.setWindowTitle("OCR Capture")
        self.setFixedSize(320, 300)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(QLabel("OCR Engine:"))
        self._combo = QComboBox()
        self._combo.addItems(list(OCR_OPTIONS.keys()))
        self._combo.setCurrentText(self._ocr_name)
        self._combo.currentTextChanged.connect(self._on_ocr_changed)
        layout.addWidget(self._combo)

        self._btn = QPushButton("CAPTURE")
        self._btn.setMinimumHeight(40)
        self._btn.clicked.connect(self._do_capture)
        layout.addWidget(self._btn)

        layout.addWidget(QLabel("Status:"))
        self._status = QLabel("Ready. Click CAPTURE.")
        layout.addWidget(self._status)

        layout.addWidget(QLabel("Result:"))
        self._result = QTextEdit()
        self._result.setReadOnly(True)
        self._result.setMaximumHeight(120)
        layout.addWidget(self._result)

        self.status_changed.connect(self._status.setText)
        self.result_changed.connect(self._result.setText)

    def _on_ocr_changed(self, name):
        self._ocr_name = name
        self._ocr = None

    def _do_capture(self):
        self.status_changed.emit("Capturing…")
        Capture().run()

        qimg = QPixmap("temp/image.png")
        geo = self.screen().geometry()
        self._overlay = OverlayWindow(qimg, geo)
        self._overlay.crop_done.connect(self._on_crop)
        self._overlay.show()

    def _on_crop(self, x1, y1, x2, y2):
        self.status_changed.emit(f"Cropped {x2-x1}x{y2-y1}, running OCR…")
        img = Image.open("temp/image.png")
        img.crop((x1, y1, x2, y2)).save("temp/image.png")
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        try:
            if self._ocr is None:
                self._ocr = _load_ocr(self._ocr_name)
            src = self._ocr.run()
            texts = self._ocr.extract(src)
            if not texts or all(len(t) < 3 for t in texts):
                self.status_changed.emit("No text found.")
                return
            lines = [TranslateGemma().run(t) for t in texts]
            self.result_changed.emit("\n".join(lines))
            self.status_changed.emit(f"Done ({len(lines)} lines)")
        except Exception as e:
            self.status_changed.emit(f"Error: {e}")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
