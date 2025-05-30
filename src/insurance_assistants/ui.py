import mimetypes
import os
import re
import shutil

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import login
from smolagents.gradio_ui import stream_to_gradio
from src.insurance_assistants.agents import manager_agent
from src.insurance_assistants.consts import PRIMARY_HEADING, PROMPT_PREFIX

load_dotenv(override=True)
# login(token=os.getenv(key="HF_TOKEN"))







class UI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(self, file_upload_folder: str | None = None):
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

    def interact_with_agent(self, prompt, messages, session_state):
        # Get or create session-specific agent
        if "agent" not in session_state:
            session_state["agent"] = manager_agent
        prompt = PROMPT_PREFIX + prompt
        # Adding monitoring
        try:
            # log the existence of agent memory
            has_memory = hasattr(session_state["agent"], "memory")
            print(f"Agent has memory: {has_memory}")
            if has_memory:
                print(f"Memory type: {type(session_state['agent'].memory)}")

            messages.append(gr.ChatMessage(role="user", content=prompt))
            yield messages

            for msg in stream_to_gradio(
                agent=session_state["agent"],
                task=prompt,
                reset_agent_memory=False,
            ):
                messages.append(msg)
                yield messages
            yield messages
        except Exception as e:
            print(f"Error in interaction: {str(e)}")
            raise

    def upload_file(
        self,
        file,
        file_uploads_log,
        allowed_file_types=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ],
    ):
        """
        Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """
        if file is None:
            return gr.Textbox("No file uploaded", visible=True), file_uploads_log

        try:
            mime_type, _ = mimetypes.guess_type(file.name)
        except Exception as e:
            return gr.Textbox(f"Error: {e}", visible=True), file_uploads_log

        if mime_type not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        type_to_ext = {}
        for ext, t in mimetypes.types_map.items():
            if t not in type_to_ext:
                type_to_ext[t] = ext

        # Ensure the extension correlates to the mime type
        sanitized_name = sanitized_name.split(".")[:-1]
        sanitized_name.append("" + type_to_ext[mime_type])
        sanitized_name = "".join(sanitized_name)

        # Save the uploaded file to the specified folder
        file_path = os.path.join(
            self.file_upload_folder, os.path.basename(sanitized_name)
        )
        shutil.copy(file.name, file_path)

        return gr.Textbox(
            f"File uploaded: {file_path}", visible=True
        ), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        return (
            text_input
            + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0
                else ""
            ),
            gr.Textbox(
                value="",
                interactive=False,
                placeholder="Please wait while the agent answers your question",
            ),
            gr.Button(interactive=False),
        )

    

    def launch(self, **kwargs):
        with gr.Blocks(fill_height=True) as demo:
            @gr.render()
            def layout(request: gr.Request):
                # Render layout with sidebar
                with gr.Blocks(
                    fill_height=True,
                ):
                    file_uploads_log = gr.State([])
                    with gr.Sidebar():
                        gr.Markdown(value=PRIMARY_HEADING)
                        with gr.Group():
                            gr.Markdown("**Your question, please...**", container=True)
                            text_input = gr.Textbox(
                                lines=3,
                                label="Your question, please...",
                                container=False,
                                placeholder="Enter your prompt here and press Shift+Enter or press the button",
                                # value=PROMPT_PREFIX
                            )
                            launch_research_btn = gr.Button(
                                value="Run", variant="primary"
                            )

                        # If an upload folder is provided, enable the upload feature
                        if self.file_upload_folder is not None:
                            upload_file = gr.File(label="Upload a file")
                            upload_status = gr.Textbox(
                                label="Upload Status",
                                interactive=False,
                                visible=False,
                            )
                            upload_file.change(
                                fn=self.upload_file,
                                inputs=[upload_file, file_uploads_log],
                                outputs=[upload_status, file_uploads_log],
                            )

                        gr.HTML("<br><br><h4><center>Powered by:</center></h4>")
                        with gr.Row():
                            gr.HTML("""<div style="display: flex; align-items: center; gap: 8px; font-family: system-ui, -apple-system, sans-serif;">
                    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png" style="width: 32px; height: 32px; object-fit: contain;" alt="logo">
                    <a target="_blank" href="https://github.com/huggingface/smolagents"><b>huggingface/smolagents</b></a>
                    </div>""")

                    # Add session state to store session-specific data
                    session_state = gr.State(
                        {}
                    )  # Initialize empty state for each session
                    stored_messages = gr.State([])
                    chatbot = gr.Chatbot(
                        label="Health Insurance Agent",
                        type="messages",
                        avatar_images=(
                            None,
                            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                        ),
                        resizeable=False,
                        scale=1,
                        elem_id="Insurance-Agent",
                    )

                    text_input.submit(
                        fn=self.log_user_message,
                        inputs=[text_input, file_uploads_log],
                        outputs=[stored_messages, text_input, launch_research_btn],
                    ).then(
                        fn=self.interact_with_agent,
                        # Include session_state in function calls
                        inputs=[stored_messages, chatbot, session_state],
                        outputs=[chatbot],
                    ).then(
                        fn=lambda: (
                            gr.Textbox(
                                interactive=True,
                                placeholder="Enter your prompt here and press the button",
                            ),
                            gr.Button(interactive=True),
                        ),
                        inputs=None,
                        outputs=[text_input, launch_research_btn],
                    )
                    launch_research_btn.click(
                        fn=self.log_user_message,
                        inputs=[text_input, file_uploads_log],
                        outputs=[stored_messages, text_input, launch_research_btn],
                    ).then(
                        fn=self.interact_with_agent,
                        # Include session_state in function calls
                        inputs=[stored_messages, chatbot, session_state],
                        outputs=[chatbot],
                    ).then(
                        fn=lambda: (
                            gr.Textbox(
                                interactive=True,
                                placeholder="Enter your prompt here and press the button",
                            ),
                            gr.Button(interactive=True),
                        ),
                        inputs=None,
                        outputs=[text_input, launch_research_btn],
                    )

        demo.launch(debug=True, **kwargs)


# UI().launch(share=False)
