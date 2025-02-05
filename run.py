import os
import subprocess
import sys

# Ensure a script name is provided
if len(sys.argv) < 2:
    print("Usage: python manage.py [scriptname]")
    sys.exit(1)

script_name = f"{sys.argv[1]}.py"  # Get script name from command line

# Get absolute paths
parent_dir = os.path.abspath(os.path.dirname(__file__))  # Parent directory
examples_dir = os.path.join(parent_dir, "backtests")
qstrader_dir = os.path.join(parent_dir, "qstrader")

# Ensure qstrader is available in Python's module path
sys.path.insert(0, parent_dir)  # Add parent directory to sys.path

# Construct full script path
script_path = os.path.join(examples_dir, script_name)

# Check if the script exists
if not os.path.isfile(script_path):
    print(f"Error: {script_name} not found in {examples_dir}")
    sys.exit(1)

# Run the script using subprocess
subprocess.run(["python", script_path], env={**os.environ, "PYTHONPATH": parent_dir})
