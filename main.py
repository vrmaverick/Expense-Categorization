# demo.py
import sys
from streamlit import cli as stcli  # streamlit<=1.39 uses this
# from streamlit.web import cli as stcli  # for newer versions if needed

def run_app(script_path="app.py"):
    sys.argv = ["streamlit", "run", script_path]
    stcli.main()

if __name__ == "__main__":
    run_app()
