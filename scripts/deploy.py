import sys
from huggingface_hub import HfApi

def deploy():
    print("Preparing to deploy to Hugging Face Spaces...")

    # Get user inputs
    hf_token = input("1. Enter your Hugging Face Access Token (from https://huggingface.co/settings/tokens): ").strip()
    space_id = input("2. Enter your Space ID (e.g. your-username/anime-rag): ").strip()

    if not hf_token or not space_id:
        print("Token and Space ID are required.")
        return

    print("\nUploading files... (this might take a minute depending on your internet speed)")
    
    api = HfApi(token=hf_token)
    
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=space_id,
            repo_type="space",
            # We explicitly ignore files that shouldn't go to the cloud
            ignore_patterns=[
                ".venv/*",
                "venv/*",
                "__pycache__/*",
                ".env",
                "*.db",          # Ignore local chat_history.db
                "token.json",    # Ignore local Google Calendar token
                "credentials.json",
                ".git/*",
                "charts/*",
                ".vscode/*",
                "*.ipynb"
            ]
        )
        print("\nDeployment successful!")
        print(f"Your app is now building at: https://huggingface.co/spaces/{space_id}")
    except Exception as e:
        print(f"\nDeployment failed: {e}")

if __name__ == "__main__":
    deploy()
