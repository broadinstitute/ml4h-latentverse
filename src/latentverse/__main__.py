"""Enable ``python -m latentverse`` to invoke the CLI."""

import sys

from latentverse.cli import main

if __name__ == "__main__":
    sys.exit(main())
