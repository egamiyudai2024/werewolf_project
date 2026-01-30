# check_env.py
import trl
import transformers
import sys

print(f"--- Environment Check ---")
print(f"Python Executable: {sys.executable}")
print(f"TRL Version:       {trl.__version__}")
print(f"TRL Path:          {trl.__file__}")
print(f"Transformers Version: {transformers.__version__}")
print(f"Transformers Path:    {transformers.__file__}")
print(f"-------------------------")