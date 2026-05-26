import gradio as gr
import sys
import os
from textSummarizer.pipeline.prediction import PredictionPipeline

def summarize_text(text):
    if not text or not text.strip():
        return "Please enter some text or dialogue to summarize."
    try:
        pipeline = PredictionPipeline()
        output = pipeline.predict(text)
        return output
    except Exception as e:
        return f"An error occurred: {str(e)}\nMake sure you have downloaded or trained the model successfully."

# Custom high-end CSS for glassmorphism and modern UI feel
custom_css = """
body {
    background-color: #0B0F19;
}
.gradio-container {
    font-family: 'Outfit', 'Inter', sans-serif !important;
    background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.05), transparent), 
                radial-gradient(circle at bottom left, rgba(168, 85, 247, 0.05), transparent);
}
.header-box {
    text-align: center;
    padding: 2.5rem 1rem;
    background: rgba(255, 255, 255, 0.02);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 2rem;
    border-radius: 16px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
}
.header-box h1 {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.header-box p {
    color: #9CA3AF;
    font-size: 1.1rem;
}
.footer {
    text-align: center;
    margin-top: 3rem;
    padding: 1.5rem 0;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    color: #4B5563;
    font-size: 0.9rem;
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_color="rgba(255, 255, 255, 0.08)",
    button_primary_background_fill="linear-gradient(135deg, #6366F1 0%, #A855F7 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4F46E5 0%, #9333EA 100%)",
    button_primary_text_color="white",
)

with gr.Blocks(title="NLP Dialogue Summarizer") as demo:
    # Header Section
    gr.HTML(
        """
        <div class="header-box">
            <h1>✨ NLP Dialogue Summarizer</h1>
            <p>Paste a conversation transcript or dialogue and get an elegant, concise summary instantly.</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_box = gr.Textbox(
                label="💬 Dialogue / Conversation",
                placeholder="Enter or paste your dialogue here...",
                lines=12,
                max_lines=20,
                elem_id="input-text"
            )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                submit_btn = gr.Button("✨ Summarize", variant="primary")
                
        with gr.Column(scale=1):
            output_box = gr.Textbox(
                label="📝 Generated Summary",
                placeholder="The summary will appear here...",
                lines=12,
                interactive=False,
                elem_id="output-text"
            )

    # Examples Section
    gr.Markdown("### 💡 Try these examples:")
    examples = [
        [
            "Hannah: Hey, do you have any plans for tonight?\n"
            "Mark: Not really, just planning to order some pizza and watch a movie. Why?\n"
            "Hannah: I have two extra tickets to the Coldplay concert! Someone cancelled last minute.\n"
            "Mark: Are you kidding me?! Coldplay?! I'd love to go!\n"
            "Hannah: Awesome! The show starts at 8 PM. Let's meet at the arena entrance around 7:30.\n"
            "Mark: Perfect, I'll be there! Thank you so much!"
        ],
        [
            "Manager: Team, we need to finalize the Q3 project deliverables by Friday.\n"
            "Developer: I have finished implementing the core APIs, but front-end integration is pending.\n"
            "Designer: The UI mockups are ready. I will share them with the developer today.\n"
            "Manager: Perfect. Let's sync tomorrow morning at 10 AM to review integration progress."
        ]
    ]
    
    gr.Examples(
        examples=examples,
        inputs=input_box,
        outputs=output_box,
        fn=summarize_text,
        cache_examples=False
    )
    
    # Interactions
    submit_btn.click(
        fn=summarize_text,
        inputs=input_box,
        outputs=output_box
    )
    
    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[input_box, output_box]
    )

    # Footer Section
    gr.HTML(
        """
        <div class="footer">
            <p>Powered by Pegasus SAMSum & Gradio 6.x • Purely Local Inference</p>
        </div>
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080, theme=theme, css=custom_css)
