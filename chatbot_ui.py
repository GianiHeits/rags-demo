import time
from pathlib import Path

import gradio as gr

from main import RAG, update_env_vars


rag = RAG()

def handler(message, history, task_type, additional_info):
    match task_type:
        case "Article Summary":
            response = rag.query_links([additional_info], message)
        case "YouTube Summary":
            response = rag.query_youtube_video(additional_info, message)
        case "Image Summary":
            image_paths = [str(path) for path in Path("data/images/").glob("*")]
            response = rag.query_images(image_paths, message)
        case _:
            response = f"I currently do not support '{task_type}' functionality"

    for i in range(len(response)):
        time.sleep(0.01)
        yield response[: i + 1]

demo = gr.ChatInterface(
    handler,
    additional_inputs=[
        gr.Dropdown(
            ["Article Summary", "YouTube Summary", "Image Summary"], label="Data Source", info="Choose what is the source for your RAG"
        ),
        gr.Textbox("https://heits.digital/articles/gpt3-overview", label="Additional Information", info="Choose what is the source for your RAG"),
    ],
    examples=[
        ["Why was Elon Musk afraid to make ChatGPT public?", "Article Summary", "https://heits.digital/articles/gpt3-overview"],
        ["Why is google chrome slow based on the transcript?", "YouTube Summary", "https://www.youtube.com/watch?v=lW7Mxj8KUJE&ab_channel=LinusTechTips"], 
        ["Are there any pokemon cards? And if so, what colors are the pokemons?", "Image Summary", "data/images/"]
    ],
    cache_examples=True
)

if __name__ == "__main__":
    update_env_vars(".env")
    demo.queue().launch(share=True)
