# demo.py
import sys
from streamlit.web import cli as stcli # streamlit<=1.39 uses this
# from streamlit.web import cli as stcli  # for newer versions if needed

def run_app(script_path="app.py"):
    # Ensure sys.argv is set up correctly to tell streamlit which script to run
    # You might need to check if "streamlit" is already in sys.argv or clear it first
    if "streamlit" not in sys.argv:
         sys.argv = ["streamlit", "run", script_path]
    elif sys.argv[1] != "run":
         sys.argv.insert(1, "run")
         sys.argv.insert(2, script_path)
    
    stcli.main()

if __name__ == "__main__":
    # Make sure app.py exists in the same directory, or specify the correct path
    run_app(script_path="app.py")
