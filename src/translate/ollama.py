import re

import ollama


class TranslateGemma:
    def __init__(self, src="Japanese", target="English"):
        self.src = src
        self.target = target

    def run(self, text):
        return self.run_many([text])[0]

    def run_many(self, texts):
        if not texts:
            return []

        numbered = "\n".join(f"[{i}] {text}" for i, text in enumerate(texts, 1))
        prompt = (
            f"Translate each {self.src} line into {self.target}. "
            "Keep every line separate and return exactly one numbered line for each input. "
            "Preserve the numbering and output no explanations.\n\n"
            f"{numbered}"
        )
        response = ollama.chat(
            model="translategemma",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response["message"]["content"]
        translated = {}
        valid = True
        for line in raw.splitlines():
            if not line.strip():
                continue
            match = re.fullmatch(r"\s*\[?(\d+)\]?\s*[:.)-]?\s*(.+?)\s*", line)
            if not match:
                valid = False
                break
            number = int(match.group(1))
            if number in translated or not 1 <= number <= len(texts):
                valid = False
                break
            translated[number] = match.group(2)

        # Never silently return merged, missing, duplicate, or malformed translations.
        if not valid or len(translated) != len(texts):
            return [self._run_one(text) for text in texts]
        return [translated[i] for i in range(1, len(texts) + 1)]

    def _run_one(self, text):
        response = ollama.chat(
            model="translategemma",
            messages=[{
                "role": "user",
                "content": f"You are a professional {self.src} to {self.target} translator. Your goal is to accurately convey the meaning and nuances of the original {self.src} text while adhering to {self.target} grammar, vocabulary, and cultural sensitivities. Produce only the {self.target} translation, without any additional explanations or commentary. Please translate the following {self.src} text into {self.target}: {text}",
            }],
        )
        return response["message"]["content"]
