import gradio as gr
import qrcode
from main import ask_question

def run_interface():
    gr_interface = gr.Interface(
            fn=ask_question,
            inputs=[
                # gr.Checkbox(label = "Chat History Awareness"),
                # gr.Dropdown(label = 'Username', value = [name for name in history.chat_history.keys()], allow_custom_value = True),
                gr.Dropdown(['LLaMA 3.2 3B', 'TYPHOON AI 2 3B', 'TYPHOON AI 2 70B'], label = "Model", allow_custom_value = False),
                gr.Textbox(label = "Input", info = 'Question')
            ],
            outputs=[
                gr.Textbox(label = "Response"),
                gr.Textbox(label = "Related Documents", max_lines = 100)
            ],
            flagging_mode = 'auto',
        )

    gr_interface.launch(server_port=1234, share=True, prevent_thread_lock = True)

    if gr_interface.share_url:
        img = qrcode.make(gr_interface.share_url)
        img.save(r'.\data\QRCode.png')
        print(f"Shareable URL: {gr_interface.share_url}")
        print('QR code generated.')


if __name__ == "__main__":
    print('start')
    run_interface()
