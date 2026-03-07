import sys
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Phi-3-mini-4k-instruct-4bit"

def main():
    prompt = sys.argv[1]
    loaded = load(MODEL_ID)

    model, tokenizer = loaded[0], loaded[1]

    sampler = make_sampler(
        temp=0.7,
        top_p=0.9,
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    for response in stream_generate(
        model,
        tokenizer,
        prompt_ids,
        max_tokens=256,
        sampler=sampler,
    ):
        if response.text:
            print(response.text, end="", flush=True)

if __name__ == "__main__":
    main()
