import os
import sys

# Add src to path so we can import tgp
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

if __name__ == "__main__":
    from tgp.__main__ import main

    main()
