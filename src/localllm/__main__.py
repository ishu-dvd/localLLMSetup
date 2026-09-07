"""Lets `python -m localllm` work, not just the installed `localllm` script.

The console entry point only exists after an install. During setup — which is
the entire point of this tool — the most likely way to run it is from a checkout,
and `python -m localllm` failing there sends people to debug their PATH.
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
