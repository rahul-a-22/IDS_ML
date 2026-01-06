import subprocess
import sys
import time

def run_flask():
    return subprocess.Popen(
        [sys.executable, "-m", "api.app"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

def run_streamlit():
    return subprocess.Popen(
        ["streamlit", "run", "ui/dashboard.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

if __name__ == "__main__":
    print("Starting Flask API...")
    flask_process = run_flask()

    time.sleep(5)

    print("Starting Streamlit Dashboard...")
    streamlit_process = run_streamlit()

    try:
        flask_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("Stopping applications...")
        flask_process.terminate()
        streamlit_process.terminate()
