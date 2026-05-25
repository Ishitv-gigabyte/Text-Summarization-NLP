import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ── Load model once at startup ──────────────────────────────────────────
MODEL_NAME = "transformersbook/pegasus-samsum"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


def summarize(text: str) -> str:
    """Generate a concise summary of the input dialogue."""
    if not text or not text.strip():
        return "⚠️ Please enter some text to summarize."

    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            length_penalty=0.8,
            num_beams=8,
            max_length=128,
        )

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output


# ── Examples ────────────────────────────────────────────────────────────
examples = [
    [
        "Amanda: Hey, are we still meeting for lunch tomorrow?\n"
        "John: Yes! I was thinking 12:30 at the Italian place on Main Street.\n"
        "Amanda: Sounds perfect. Should I invite Sarah too?\n"
        "John: Definitely, the more the merrier.\n"
        "Amanda: Great, I'll text her now. See you tomorrow!"
    ],
    [
        "Bob: Did you finish the quarterly report?\n"
        "Alice: Almost, I need the sales figures from Dave.\n"
        "Bob: I'll ping him right now. When's the deadline?\n"
        "Alice: Friday end of day. But I'd like to have it reviewed by Thursday.\n"
        "Bob: Got it. I'll make sure Dave sends the numbers by tonight.\n"
        "Alice: Thanks Bob, you're a lifesaver."
    ],
    [
        "Mom: Don't forget to pick up groceries on your way home.\n"
        "Jake: What do we need?\n"
        "Mom: Milk, eggs, bread, and some vegetables. Maybe broccoli and carrots.\n"
        "Jake: Anything else?\n"
        "Mom: Oh, and get some chicken for dinner tonight.\n"
        "Jake: Got it. I'll stop by the store after work."
    ],
]


# ── Gradio Interface ───────────────────────────────────────────────────
description = """
> Powered by **Google's Pegasus** model, fine-tuned on the **SAMSum** dialogue dataset.
>
> Paste a conversation below and get a concise summary in seconds.
"""

demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Paste a dialogue or conversation here...",
        label="💬 Dialogue Input",
    ),
    outputs=gr.Textbox(
        lines=4,
        label="📋 Summary",
    ),
    title="📝 Text Summarizer — Dialogue Edition",
    description=description,
    examples=examples,
    flagging_mode="never",
    submit_btn="Summarize ✨",
    clear_btn="Clear 🗑️",
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8080,
        theme=gr.themes.Soft(),
    )

