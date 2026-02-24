"""
Command Line Interface for exowrap.

Handles downloading, compiling, and configuring the ExoREM Fortran backend.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

# Constants and Configuration
EXOREM_REPO_URL = "https://gitlab.obspm.fr/Exoplanet-Atmospheres-LESIA/exorem.git"

USER_HOME = Path.home()
EXOWRAP_DIR = USER_HOME / ".exowrap"
CONFIG_FILE = EXOWRAP_DIR / "config.json"
DEFAULT_INSTALL_DIR = EXOWRAP_DIR / "exorem_source"


def check_dependencies():
    """
    Check if git, make, tar, and a Fortran compiler are installed.

    Exits the script with status 1 if any required system tools are missing.
    """
    print("Checking system dependencies...")
    missing = []

    for req in ['git', 'make', 'tar']:
        if shutil.which(req) is None:
            missing.append(req)

    # Check for at least one Fortran compiler
    if shutil.which('gfortran') is None and shutil.which('ifort') is None:
        missing.append('gfortran (or ifort)')

    if missing:
        print(f"\n❌ Missing required system tools: {', '.join(missing)}")
        print("Please install them using your package manager.")
        sys.exit(1)

    print("✅ All basic dependencies found.")


def clone_repo(install_dir: Path):
    """
    Clone the ExoREM repository into the specified directory.

    Args:
        install_dir (Path): The target directory for the repository clone.
    """
    if install_dir.exists():
        prompt = (
            f"⚠️ Directory {install_dir} already exists.\n"
            "Do you want to delete it and re-download? (y/n): "
        )
        choice = input(prompt).strip().lower()
        if choice == 'y':
            shutil.rmtree(install_dir)
        else:
            print("Skipping download.")
            return

    print(f"📥 Cloning ExoREM from {EXOREM_REPO_URL} into {install_dir}...")
    try:
        subprocess.run(
            ["git", "clone", EXOREM_REPO_URL, str(install_dir)],
            check=True
        )
        print("✅ Download complete.")
    except subprocess.CalledProcessError:
        print(
            "\n❌ Failed to clone the repository. "
            "Check your internet connection or git permissions."
        )
        sys.exit(1)


def compile_exorem(install_dir: Path) -> Path:
    """
    Extract the nested tarball, patch the Makefile, and compile ExoREM.

    Args:
        install_dir (Path): The root directory where the repo was cloned.

    Returns:
        Path: The absolute path to the directory containing the Makefile.
    """
    tarball_path = install_dir / "dist" / "exorem.tar.gz"
    extract_dir = install_dir / "dist"

    # 1. Extract the nested tarball
    if tarball_path.exists():
        print(f"📦 Found nested tarball at {tarball_path}. Extracting...")
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
    else:
        print(f"❌ Could not find {tarball_path}. Repo structure may vary.")
        sys.exit(1)

    # 2. Dynamically find the actual root directory containing the Makefile
    makefile_paths = list(extract_dir.rglob("Makefile"))
    if not makefile_paths:
        print("❌ Could not find Makefile. Tarball might be malformed.")
        sys.exit(1)

    makefile_paths.sort(key=lambda p: len(p.parts))
    make_dir = makefile_paths[0].parent
    makefile_target = makefile_paths[0]
    print(f"📂 Resolved ExoREM root to: {make_dir}")

    # 3. Patch the Makefile for Apple Silicon / Custom Architectures
    print("🛠️ Patching Makefile to remove incompatible architecture flags...")
    with open(makefile_target, 'r') as f:
        makefile_content = f.read()

    # Strip out flags which crash ARM64/Apple Silicon gfortran
    patched_content = re.sub(r'-mcmodel=(large|medium)', '', makefile_content)

    with open(makefile_target, 'w') as f:
        f.write(patched_content)

    # 4. Environment Fix
    custom_env = os.environ.copy()
    fc_path = shutil.which('gfortran') or shutil.which('ifort')
    if fc_path:
        print(f"🔧 Forcing HDF5 wrapper to use compiler: {fc_path}")
        custom_env['HDF5_FC'] = fc_path
        custom_env['FC'] = fc_path

    # 5. Compile the code (Streaming Output)
    print(f"🔨 Compiling ExoREM in {make_dir}... (Streaming output below)\n")
    print("=" * 60)
    try:
        result = subprocess.run(
            ["make", "exorem"],
            cwd=make_dir,
            env=custom_env
        )
        print("=" * 60)

        if result.returncode != 0:
            print("\n❌ Compilation failed! (See the make output above)")
            print("💡 Tip: Anaconda's h5fc is looking for its own compiler.")
            print("   Mixing Homebrew gfortran with Conda HDF5 often fails.")
            print("   Try running this to fix your Conda environment:")
            print("   conda install -c conda-forge fortran-compiler")
            sys.exit(1)

        print("✅ Compilation successful!")
        return make_dir

    except Exception as e:
        print(f"\n❌ An unexpected error occurred during compilation: {e}")
        sys.exit(1)


def setup_data(exorem_base: Path, res: int = 50):
    """
    Download and extract required K-tables for a specific resolution.

    Args:
        exorem_base (Path): The root directory of the compiled ExoREM installation.
        res (int): The resolution of the K-tables to download (e.g., 50, 500).
    """
    k_table_url = f"https://lesia.obspm.fr/exorem/ktables/default/xz/R{res}.tar.xz"
    k_table_dir = exorem_base / "data" / "k_coefficients_tables"
    tar_dest = k_table_dir / f"R{res}.tar.xz"
    res_dir = k_table_dir / f"R{res}"

    if res_dir.exists():
        print(f"✅ R{res} K-tables already exist at {res_dir}. Skipping download.")
        return

    print(f"📥 Downloading R{res} K-tables from {k_table_url}...")
    k_table_dir.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(k_table_url, tar_dest)
        print("📦 Extracting K-tables... (This may take a moment for high resolutions)")
        subprocess.run(
            ["tar", "xJvf", f"R{res}.tar.xz"],
            cwd=k_table_dir,
            capture_output=True,
            check=True
        )
        tar_dest.unlink()  # Clean up the tarball
        print(f"✅ R{res} K-tables successfully downloaded and extracted.")
    except Exception as e:
        print(f"⚠️ Failed to download/extract K-tables: {e}")
        print(f"Ensure the requested resolution (R{res}) exists on the ExoREM server.")


def save_config(make_dir: Path):
    """Save the backend paths to a JSON configuration file."""
    exe_path = make_dir / "bin" / "exorem.exe"
    data_path = make_dir / "data"

    config_data = {
        "EXOREM_BASE_PATH": str(make_dir),
        "EXOREM_EXE": str(exe_path),
        "EXOREM_DATA": str(data_path)
    }

    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)

    print(f"⚙️ Configuration saved to {CONFIG_FILE}")
    print("🚀 exowrap is now ready to use!")


def init_backend(args: argparse.Namespace):
    """Main sequence for the 'init' command."""
    install_dir = Path(args.path).resolve()

    print("--- Initializing exowrap backend ---")
    check_dependencies()
    clone_repo(install_dir)
    make_dir = compile_exorem(install_dir)
    setup_data(make_dir, res=50) # Always fetch R50 as the default baseline
    save_config(make_dir)


def download_tables_cmd(args: argparse.Namespace):
    """Main sequence for the 'download-tables' command."""
    if not CONFIG_FILE.exists():
        print("❌ Backend not initialized! Run 'exowrap init' first.")
        sys.exit(1)
        
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
        
    exorem_base = Path(config["EXOREM_BASE_PATH"])
    setup_data(exorem_base, res=args.res)


def main():
    """Parse arguments and execute the appropriate CLI sub-command."""
    parser = argparse.ArgumentParser(
        description="exowrap CLI: Manage the ExoREM backend."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Download and compile the ExoREM Fortran code."
    )
    init_parser.add_argument(
        "--path",
        type=str,
        default=str(DEFAULT_INSTALL_DIR),
        help=f"Where to install the ExoREM source (default: {DEFAULT_INSTALL_DIR})"
    )

    # Download Tables command
    table_parser = subparsers.add_parser(
        "download-tables",
        help="Download specific resolution K-tables (e.g., 50, 500, 20000)."
    )
    table_parser.add_argument(
        "--res",
        type=int,
        required=True,
        help="Resolution of the tables to download (e.g., 500)"
    )

    args = parser.parse_args()

    if args.command == "init":
        init_backend(args)
    elif args.command == "download-tables":
        download_tables_cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()