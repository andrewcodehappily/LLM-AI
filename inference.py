import mlx.core as mx
from transformers import AutoTokenizer

VOCAB_SIZE = 151680  # Qwen 2.5 tokenizer vocab size (151643 padded to 128)

# Load Qwen 2.5 tokenizer once at module level
_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B", trust_remote_code=True
        )
    return _tokenizer


def generate_text(model, prompt, max_new_tokens=100, temperature=1.0, top_k=None):
    enc = get_tokenizer()
    input_ids = prompt.astype(mx.int32)
    initial_length = input_ids.shape[1]

    while input_ids.shape[1] < max_new_tokens + initial_length:
        logits = model(input_ids)
        logits = logits[:, -1, :] / temperature

        logits = logits[:, :VOCAB_SIZE]

        if top_k is not None:
            top_k_values = mx.topk(logits, k=top_k, axis=-1)
            min_top_k = top_k_values[:, -1].reshape(-1, 1)
            logits = mx.where(logits < min_top_k, mx.array(float("-inf")), logits)

        probs = mx.softmax(logits, axis=-1)
        next_token = mx.random.categorical(probs, num_samples=1)

        input_ids = mx.concat([input_ids, next_token], axis=1)

    output_texts = []
    for i in range(input_ids.shape[0]):
        output_ids = input_ids[i].tolist()
        output_text = enc.decode(output_ids)
        output_texts.append(output_text)

    if len(output_texts) == 1:
        return output_texts[0]
    else:
        return "\n".join(f"[{i+1}] {text}" for i, text in enumerate(output_texts))
