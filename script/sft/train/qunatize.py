# quantize_fp8_qwen3vl.py
import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForVision2Seq,
    AutoProcessor,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

BASE = "/hub_data4/seohyun/saves/ecva_instruct_1223/full/sft/checkpoint-350"
OUT  = "/hub_data4/seohyun/saves/ecva_instruct_1223/full/sft/checkpoint-350-fp8"

os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────
# 1) tokenizer / processor 로드
# ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)

# ─────────────────────────────────────────────
# 2) Qwen3-VL 모델 로드 (Vision2Seq)
# ─────────────────────────────────────────────
model = AutoModelForVision2Seq.from_pretrained(
    BASE,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

# ─────────────────────────────────────────────
# 3) FP8_DYNAMIC 양자화 (데이터 불필요)
# ─────────────────────────────────────────────
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],  # 출력층은 보통 제외
)

oneshot(
    model=model,
    recipe=recipe,
)

# ─────────────────────────────────────────────
# 4) 저장 (중요: processor까지 같이 저장!)
# ─────────────────────────────────────────────
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
processor.save_pretrained(OUT)   # ⭐ 이 줄이 핵심

print("✅ Saved FP8 Qwen3-VL model (with processor) to:", OUT)
