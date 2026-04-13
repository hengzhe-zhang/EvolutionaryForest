import os
import shutil

# Targets to delete relative to the core submodule root
targets = ["evolutionary_forest/__init__.py", "docs", "setup.py", "tutorial"]


def main():
    for target in targets:
        if os.path.exists(target):
            if os.path.isdir(target):
                shutil.rmtree(target)
                print(f"Deleted directory: {target}")
            else:
                os.remove(target)
                print(f"Deleted file: {target}")
        else:
            print(f"Target not found: {target}")


if __name__ == "__main__":
    main()
