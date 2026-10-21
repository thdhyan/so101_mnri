"""Make the repo's `scripts/` importable from the tests.

The scripts are standalone programs, not an installed package, and they import
each other by bare name (`import episode_outcome_keys`). Putting `scripts/` on
the path is what lets a test drive them the way the scripts do.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
