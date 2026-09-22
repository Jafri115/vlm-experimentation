"""Add the repository script package to direct-execution imports."""
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
