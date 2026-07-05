import ctypes
import ctypes.wintypes
from dataclasses import dataclass
from time import sleep

user32 = ctypes.windll.user32
VK_LBUTTON = 0x01

@dataclass
class Region:
    x1: int
    y1: int
    x2: int
    y2: int


def get_region() -> Region:
    """Wait for two left-clicks → return normalised (top-left, bottom-right) region."""
    clicks = []

    print("Click top-left corner of crop area...", end="", flush=True)

    while len(clicks) < 2:
        # wait for press
        while not user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000:
            sleep(0.03)

        pt = ctypes.wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        clicks.append((pt.x, pt.y))
        print(f" ({pt.x}, {pt.y})")

        # wait for release
        while user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000:
            sleep(0.03)

        if len(clicks) == 1:
            print("Click bottom-right corner...", end="", flush=True)

    x1, y1 = clicks[0]
    x2, y2 = clicks[1]

    return Region(
        x1=min(x1, x2),
        y1=min(y1, y2),
        x2=max(x1, x2),
        y2=max(y1, y2),
    )
