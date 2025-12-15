#!/usr/bin/env python
"""
One-click script to build and publish evolutionary_forest to PyPI.

Usage:
    python scripts/publish_to_pypi.py [--test] [--skip-build] [--skip-clean]

Options:
    --test          Upload to TestPyPI instead of PyPI
    --skip-build    Skip building the package (use existing dist files)
    --skip-clean    Skip cleaning old build artifacts
"""

import os
import sys
import shutil
import subprocess
import argparse
import re
from pathlib import Path

# Fix encoding issues on Windows
if sys.platform == "win32":
    # Set UTF-8 encoding for Python I/O
    os.environ["PYTHONIOENCODING"] = "utf-8"

    # Reconfigure stdout/stderr if possible (Python 3.7+)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    # Set Windows console code page to UTF-8 (critical for twine/rich library)
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8 code page
        kernel32.SetConsoleCP(65001)  # UTF-8 code page
    except (AttributeError, OSError):
        pass


def run_command(cmd, check=True, shell=False, capture_output=False):
    """Run a shell command and return the result."""
    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print(f"{'=' * 60}\n")

    # Set UTF-8 encoding in environment for subprocess
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    if isinstance(cmd, str):
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=capture_output,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    else:
        result = subprocess.run(
            cmd,
            check=check,
            shell=shell,
            capture_output=capture_output,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

    if result.returncode != 0 and check:
        if capture_output:
            print(f"Error output: {result.stderr}")
        print(f"Error: Command failed with return code {result.returncode}")
        sys.exit(1)

    return result


def get_current_version():
    """Get the current version from setup.py."""
    setup_py = Path("setup.py")
    if not setup_py.exists():
        return None

    content = setup_py.read_text(encoding="utf-8")
    match = re.search(r"version=['\"]([^'\"]+)['\"]", content)
    if match:
        return match.group(1)
    return None


def increment_minor_version(version):
    """Increment the minor version number (e.g., 0.2.4 -> 0.2.5)."""
    parts = version.split(".")
    if len(parts) >= 3:
        # Increment patch version (last number)
        parts[-1] = str(int(parts[-1]) + 1)
    elif len(parts) == 2:
        # Add patch version if it doesn't exist
        parts.append("1")
    else:
        # If format is unexpected, just increment the last part
        parts[-1] = str(int(parts[-1]) + 1) if parts[-1].isdigit() else "1"

    return ".".join(parts)


def update_version(new_version):
    """Update version in setup.py and __init__.py."""
    print(f"\n🔄 Updating version to {new_version}...")

    # Update setup.py
    setup_py = Path("setup.py")
    if setup_py.exists():
        content = setup_py.read_text(encoding="utf-8")
        content = re.sub(
            r"version=['\"]([^'\"]+)['\"]", f"version='{new_version}'", content
        )
        setup_py.write_text(content, encoding="utf-8")
        print(f"  ✅ Updated setup.py")

    # Update __init__.py
    init_py = Path("evolutionary_forest/__init__.py")
    if init_py.exists():
        content = init_py.read_text(encoding="utf-8")
        content = re.sub(
            r"__version__ = ['\"]([^'\"]+)['\"]",
            f"__version__ = '{new_version}'",
            content,
        )
        init_py.write_text(content, encoding="utf-8")
        print(f"  ✅ Updated evolutionary_forest/__init__.py")

    print(f"✅ Version updated to {new_version}!")


def clean_build_artifacts():
    """Remove old build artifacts."""
    print("\n🧹 Cleaning old build artifacts...")

    dirs_to_remove = ["build", "dist", "*.egg-info"]
    for pattern in dirs_to_remove:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                print(f"  Removing directory: {path}")
                shutil.rmtree(path, ignore_errors=True)
            elif path.is_file():
                print(f"  Removing file: {path}")
                path.unlink()

    # Also remove any .egg-info directories
    for path in Path(".").rglob("*.egg-info"):
        if path.is_dir():
            print(f"  Removing directory: {path}")
            shutil.rmtree(path, ignore_errors=True)

    print("✅ Cleanup complete!")


def build_package():
    """Build the package (sdist and wheel)."""
    print("\n📦 Building package...")

    # Check if setuptools and wheel are installed
    try:
        import setuptools
        import wheel
    except ImportError:
        print("⚠️  Installing build dependencies...")
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "setuptools",
                "wheel",
                "twine",
            ]
        )

    # Build source distribution and wheel
    run_command([sys.executable, "setup.py", "sdist", "bdist_wheel"])

    print("✅ Build complete!")


def check_distribution():
    """Check the distribution files."""
    print("\n🔍 Checking distribution files...")

    try:
        import twine
    except ImportError:
        print("⚠️  Installing twine...")
        run_command([sys.executable, "-m", "pip", "install", "--upgrade", "twine"])

    dist_files = list(Path("dist").glob("*"))
    if not dist_files:
        print("❌ Error: No distribution files found in dist/")
        sys.exit(1)

    print(f"Found {len(dist_files)} distribution file(s):")
    for f in dist_files:
        print(f"  - {f.name} ({f.stat().st_size / 1024:.2f} KB)")

    # Check the package
    run_command([sys.executable, "-m", "twine", "check", "dist/*"])

    print("✅ Distribution check complete!")


def upload_to_pypi(test=False, max_retries=3):
    """Upload the package to PyPI or TestPyPI with automatic version bumping."""
    print("\n🚀 Uploading to PyPI...")

    try:
        import twine
    except ImportError:
        print("⚠️  Installing twine...")
        run_command([sys.executable, "-m", "pip", "install", "--upgrade", "twine"])

    repository = "testpypi" if test else "pypi"
    repository_url = (
        "https://test.pypi.org/legacy/" if test else "https://upload.pypi.org/legacy/"
    )

    for attempt in range(max_retries):
        print(
            f"\n📤 Uploading to {'TestPyPI' if test else 'PyPI'}... (Attempt {attempt + 1}/{max_retries})"
        )
        print(f"   Repository URL: {repository_url}")
        if attempt == 0:
            print("\n⚠️  You will be prompted for your PyPI credentials.")
            print("   Username: __token__")
            print("   Password: Your PyPI API token (pypi-...)\n")

        # Upload using twine
        cmd = [
            sys.executable,
            "-m",
            "twine",
            "upload",
            "--repository",
            repository,
            "dist/*",
        ]

        # Run upload command and capture output
        # Set UTF-8 encoding in environment for twine/rich library
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        if process.returncode == 0:
            print(process.stdout)
            print(f"\n✅ Successfully uploaded to {'TestPyPI' if test else 'PyPI'}!")
            if test:
                print("\n💡 To install from TestPyPI, use:")
                print(
                    "   pip install -i https://test.pypi.org/simple/ evolutionary-forest"
                )
            else:
                print("\n💡 To install, use:")
                print("   pip install evolutionary-forest")
            return True

        # Check if error is due to file already existing
        error_output = process.stderr if process.stderr else process.stdout
        print(error_output)  # Show the error to user

        if error_output and (
            "File already exists" in error_output
            or "already been registered" in error_output
            or "already exists" in error_output.lower()
            or "HTTPError: 400" in error_output
        ):
            print(f"\n⚠️  Version already exists on PyPI!")
            current_version = get_current_version()
            if current_version:
                new_version = increment_minor_version(current_version)
                print(
                    f"🔄 Automatically bumping version: {current_version} -> {new_version}"
                )

                # Update version in files
                update_version(new_version)

                # Clean and rebuild
                print("\n🧹 Cleaning old build artifacts...")
                clean_build_artifacts()
                print("\n📦 Rebuilding package with new version...")
                build_package()
                print("\n🔍 Rechecking distribution...")
                check_distribution()

                # Continue to next attempt
                continue
            else:
                print("❌ Could not determine current version. Please update manually.")
                return False
        else:
            # Different error, print and exit
            print(f"\n❌ Upload failed with error:")
            print(error_output)
            return False

    print(f"\n❌ Failed to upload after {max_retries} attempts.")
    return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Build and publish evolutionary_forest to PyPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/publish_to_pypi.py              # Build and upload to PyPI
  python scripts/publish_to_pypi.py --test       # Build and upload to TestPyPI
  python scripts/publish_to_pypi.py --skip-clean # Skip cleaning old artifacts
  python scripts/publish_to_pypi.py --skip-build # Use existing dist files
        """,
    )

    parser.add_argument(
        "--test", action="store_true", help="Upload to TestPyPI instead of PyPI"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building the package (use existing dist files)",
    )
    parser.add_argument(
        "--skip-clean", action="store_true", help="Skip cleaning old build artifacts"
    )

    args = parser.parse_args()

    # Change to the core directory (parent of scripts folder)
    script_dir = Path(__file__).parent.absolute()
    core_dir = script_dir.parent  # Go up one level from scripts/ to core/
    os.chdir(core_dir)
    print(f"📁 Working directory: {core_dir}")

    # Verify we're in the right directory
    if not Path("setup.py").exists():
        print(
            "❌ Error: setup.py not found. Please run this script from the core directory."
        )
        sys.exit(1)

    try:
        # Step 1: Clean old build artifacts
        if not args.skip_clean:
            clean_build_artifacts()

        # Step 2: Build the package
        if not args.skip_build:
            build_package()

        # Step 3: Check the distribution
        check_distribution()

        # Step 4: Upload to PyPI
        success = upload_to_pypi(test=args.test)

        if success:
            print("\n" + "=" * 60)
            print("🎉 All done! Package successfully published!")
            print("=" * 60 + "\n")
        else:
            print("\n" + "=" * 60)
            print("❌ Upload failed. Please check the error messages above.")
            print("=" * 60 + "\n")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
