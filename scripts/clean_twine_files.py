#!/usr/bin/env python
"""
Script to clean twine-generated files and build artifacts.

This script removes:
- dist/ directory (contains .whl and .tar.gz distribution files)
- build/ directory (contains build artifacts)
- *.egg-info directories (package metadata)

Usage:
    python scripts/clean_twine_files.py [--dry-run] [--verbose]

Options:
    --dry-run    Show what would be deleted without actually deleting
    --verbose    Show detailed information about each file/directory removed
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def clean_twine_files(dry_run=False, verbose=False):
    """
    Remove twine-generated files and build artifacts.
    
    Args:
        dry_run: If True, only show what would be deleted without deleting
        verbose: If True, show detailed information about each item
    """
    print("\n🧹 Cleaning twine-generated files and build artifacts...")
    print("=" * 60)
    
    # Get the core directory (parent of scripts folder)
    script_dir = Path(__file__).parent.absolute()
    core_dir = script_dir.parent
    os.chdir(core_dir)
    
    if verbose:
        print(f"📁 Working directory: {core_dir}")
    
    removed_count = 0
    total_size = 0
    
    # Directories and patterns to remove
    items_to_remove = [
        ("dist", "Distribution files (.whl, .tar.gz)"),
        ("build", "Build artifacts"),
    ]
    
    # Remove specific directories
    for dir_name, description in items_to_remove:
        dir_path = Path(core_dir) / dir_name
        if dir_path.exists():
            if verbose:
                print(f"\n📦 {description}: {dir_path}")
            
            if dir_path.is_dir():
                # Calculate size before removal
                size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                total_size += size
                
                if dry_run:
                    print(f"  [DRY RUN] Would remove directory: {dir_path}")
                    if verbose:
                        print(f"  Size: {size / 1024:.2f} KB")
                else:
                    try:
                        shutil.rmtree(dir_path, ignore_errors=True)
                        print(f"  ✅ Removed directory: {dir_path}")
                        if verbose:
                            print(f"  Size: {size / 1024:.2f} KB")
                        removed_count += 1
                    except Exception as e:
                        print(f"  ❌ Error removing {dir_path}: {e}")
            elif dir_path.is_file():
                size = dir_path.stat().st_size
                total_size += size
                
                if dry_run:
                    print(f"  [DRY RUN] Would remove file: {dir_path}")
                    if verbose:
                        print(f"  Size: {size / 1024:.2f} KB")
                else:
                    try:
                        dir_path.unlink()
                        print(f"  ✅ Removed file: {dir_path}")
                        if verbose:
                            print(f"  Size: {size / 1024:.2f} KB")
                        removed_count += 1
                    except Exception as e:
                        print(f"  ❌ Error removing {dir_path}: {e}")
        else:
            if verbose:
                print(f"\n📦 {description}: {dir_path} (not found)")
    
    # Remove .egg-info directories recursively
    print("\n📦 Package metadata (.egg-info directories):")
    egg_info_dirs = list(Path(core_dir).rglob("*.egg-info"))
    
    if egg_info_dirs:
        for egg_info_path in egg_info_dirs:
            if verbose:
                print(f"  Found: {egg_info_path}")
            
            if egg_info_path.is_dir():
                # Calculate size before removal
                size = sum(f.stat().st_size for f in egg_info_path.rglob('*') if f.is_file())
                total_size += size
                
                if dry_run:
                    print(f"  [DRY RUN] Would remove directory: {egg_info_path}")
                    if verbose:
                        print(f"  Size: {size / 1024:.2f} KB")
                else:
                    try:
                        shutil.rmtree(egg_info_path, ignore_errors=True)
                        print(f"  ✅ Removed directory: {egg_info_path}")
                        if verbose:
                            print(f"  Size: {size / 1024:.2f} KB")
                        removed_count += 1
                    except Exception as e:
                        print(f"  ❌ Error removing {egg_info_path}: {e}")
    else:
        if verbose:
            print("  No .egg-info directories found")
    
    # Summary
    print("\n" + "=" * 60)
    if dry_run:
        print(f"📊 [DRY RUN] Would remove {removed_count} item(s)")
        print(f"📊 Total size: {total_size / 1024:.2f} KB ({total_size / (1024 * 1024):.2f} MB)")
        print("\n⚠️  No files were actually deleted. Run without --dry-run to delete.")
    else:
        print(f"✅ Cleanup complete!")
        print(f"📊 Removed {removed_count} item(s)")
        print(f"📊 Total size freed: {total_size / 1024:.2f} KB ({total_size / (1024 * 1024):.2f} MB)")
    print("=" * 60 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Clean twine-generated files and build artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/clean_twine_files.py              # Clean all twine files
  python scripts/clean_twine_files.py --dry-run   # Show what would be deleted
  python scripts/clean_twine_files.py --verbose   # Show detailed information
        """,
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information about each file/directory removed",
    )
    
    args = parser.parse_args()
    
    try:
        clean_twine_files(dry_run=args.dry_run, verbose=args.verbose)
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

