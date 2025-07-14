import glob
import os


def find_rmb_files(base_path):
    if base_path.rstrip("/").endswith((".rmb", ".hdf5")):
        rmb_path_list = [base_path]
    elif os.path.isdir(base_path):
        rmb_path_list = sorted(
            [
                f
                for f in glob.glob(f"{base_path}/**/*.*", recursive=True)
                if f.endswith(".rmb")
                or (f.endswith(".hdf5") and not f.endswith(".rmb.hdf5"))
            ]
        )
    else:
        raise ValueError(f"[find_rmb_files] RMB file not found: {base_path}")
    return rmb_path_list
