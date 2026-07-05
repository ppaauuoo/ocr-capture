from PIL import ImageGrab


class Capture:
    def run(self, name="image.png"):
        img = ImageGrab.grab()
        img.save("temp/" + name)
