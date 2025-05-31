    from dotenv import load_dotenv, find_dotenv
    from src.insurance_assistants.ui import UI
    from src.insurance_assistants.consts import PROJECT_ROOT_DIR
    import logging

    logging.basicConfig(level=logging.INFO)
    _ = load_dotenv(dotenv_path=find_dotenv())# Load .env variables

    if __name__ == "__main__":
        # Ensure the allowed_paths correctly points to where your PDFs are for the viewer
        # The viewer itself loads PDFs from PROJECT_ROOT_DIR / "data/policy_wordings/"
        # as per ui.py's list_pdfs and display_pdf methods.
        # The allowed_paths is more for Gradio's security if it needs to serve files from there directly.
        ui_app = UI()
        ui_app.launch(
            share=False, # Set to True for a public Gradio link (temporary)
            allowed_paths=[(PROJECT_ROOT_DIR / "data/policy_wordings").as_posix()]
        )
