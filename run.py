import main
import subprocess

def run_sequence():
    # Step 1: Run the main analysis pipeline
    print("🚀 Starting main analysis...")
    main.main()
    print("✅ Main analysis completed")

    # Step 2: Start the FastAPI server
    print("🌐 Starting FastAPI server...")
    subprocess.run([
        "uvicorn",
        "fastapi_main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])

if __name__ == "__main__":
    run_sequence()