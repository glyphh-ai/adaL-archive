import sys
from pathlib import Path

# Add model root to path so `from encoder import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
