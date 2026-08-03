import json
import time

import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


model = build_sam3_image_model().float().eval()
proc = Sam3Processor(model)
img = Image.open("assets/test.jpg")


def timed(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return {"median": ts[len(ts) // 2], "min": ts[0], "max": ts[-1]}


enc = timed(lambda: proc.set_image(img))
state = proc.set_image(img)
prompts = ["person", "object", "chair"]
prompt_results = {
    prompt: timed(lambda prompt=prompt: proc.set_text_prompt(state=state, prompt=prompt))
    for prompt in prompts
}

result = {
    "model": "sam3",
    "dtype": "float32",
    "encode_s": enc,
    "prompt_s": prompt_results,
    "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
}
print(json.dumps(result, indent=2))
