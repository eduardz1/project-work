import os
import site
import sys
from importlib.util import find_spec
from pathlib import Path


def configure_cuda_env():
    """
    Adds NVIDIA CUDA libraries to LD_LIBRARY_PATH. By default, they are not
    found when installed in a virtual environment.
    """

    if "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ:
        return

    if os.environ.get("TGP_CUDA_CONFIGURED"):
        return

    if find_spec("numba.cuda") is None:
        return

    libs_to_add = []
    site_packages = site.getsitepackages()
    try:
        site_packages.append(site.getusersitepackages())
    except AttributeError:
        pass

    for sp in site_packages:
        sp_path = Path(sp)
        nvidia_path = sp_path / "nvidia"
        if nvidia_path.exists():
            for lib_dir in nvidia_path.glob("*/lib"):
                if lib_dir.is_dir():
                    libs_to_add.append(str(lib_dir))

    if not libs_to_add:
        return

    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    current_paths = current_ld_path.split(":") if current_ld_path else []

    new_paths = []
    for lib in libs_to_add:
        if lib not in current_paths:
            new_paths.append(lib)

    if new_paths:
        final_ld_path = (
            ":".join(new_paths + current_paths)
            if current_paths
            else ":".join(new_paths)
        )

        os.environ["LD_LIBRARY_PATH"] = final_ld_path
        os.environ["TGP_CUDA_CONFIGURED"] = "1"

        try:
            # Re-execute the current script
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except OSError as e:
            print(f"Warning: Failed to re-execute with new environment: {e}")


configure_cuda_env()
