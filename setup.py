"""
Install project dependencies.
Run: python setup.py
"""
import subprocess
import sys


def main():
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=__file__.rsplit("/", 1)[0] or ".",
    )
    print("All dependencies installed.")


if __name__ == "__main__":
    main()
