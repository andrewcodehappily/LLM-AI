import mlx.core as mx
from transformers import AutoTokenizer
from gpt2 import GPT, GPTConfig
from inference import generate_text, VOCAB_SIZE

model = GPT(GPTConfig(vocab_size=VOCAB_SIZE))
model.apply(lambda x: x.astype(mx.bfloat16))

# 只載 weight，不用 optimizer
import os, glob
ckpt_dirs = sorted(glob.glob("checkpoints/*/"))
if ckpt_dirs:
    latest = ckpt_dirs[-1]
    weights_path = os.path.join(latest, "model_weights.safetensors")
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        mx.eval(model.parameters())
        print(f"載入：{latest}")

enc = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
prompt = mx.array(enc.encode("在一座遙遠的島嶼上，"), dtype=mx.int32)
output = generate_text(model, prompt.reshape(1, -1), max_new_tokens=100, temperature=0)
print(output)
