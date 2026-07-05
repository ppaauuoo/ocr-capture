import ollama


class TranslateGemma:
    def __init__(self, src="Japanese", target="English"):
        self.src = src
        self.target = target

    def run(self, text):
        response = ollama.chat(
            model="translategemma",
            messages=[
                {
                    "role": "user",
                    "content": f"You are a professional {self.src} to {self.target} translator. Your goal is to accurately convey the meaning and nuances of the original {self.src} text while adhering to {self.target} grammar, vocabulary, and cultural sensitivities. Produce only the {self.target} translation, without any additional explanations or commentary. Please translate the following {self.src} text into {self.target}: {text}",
                },
            ],
        )
        return response["message"]["content"]
