from transformers import AutoModel, AutoTokenizer
import torch
from typing import List

model_name = "deepseek-ai/DeepSeek-OCR-2"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.float16,
)
model = model.eval().cuda()


class DeepseekOCR2:
    def run(self, image_path="temp/image.png"):
        return model.infer(
            tokenizer,
            prompt="<image>\nFree OCR.",
            image_file=image_path,
            output_path="temp/res",
            base_size=1024, image_size=768, crop_mode=True,
            save_results=False, eval_mode=True,
        )

    def extract(self, ocr_response) -> List[str]:
        if isinstance(ocr_response, str):
            lines = [
                line.strip()
                for line in ocr_response.split("\n")
                if line.strip() and len(line.strip()) > 3
            ]
            return lines
        return []
