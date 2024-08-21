import gradio as gr

with gr.Blocks() as demo:
    input_text = gr.Textbox(label="Input Text", elem_id="input_text")

    # @gr.render(inputs=input_text)
    # def show_split(text):
    #     if len(text) == 0:
    #         gr.Markdown("## No Input Provided")
    #     else:
    #         for letter in text:
    #             with gr.Row():
    #                 text = gr.Textbox(letter)
    #                 btn = gr.Button("Clear")
    #                 btn.click(lambda: gr.Textbox(value=""), None, text)

demo.launch(server_name="0.0.0.0", share=True)