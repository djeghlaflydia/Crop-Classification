import os

def get_unique_path(filename, base_dir=os.path.join("results", "plots")):
    """
    Returns a unique path for the given filename by appending _v1, _v2, etc. if it exists.
    Creates base_dir if it doesn't exist.
    """
    os.makedirs(base_dir, exist_ok=True)
    name, ext = os.path.splitext(filename)
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        return path
    
    version = 1
    while True:
        v_path = os.path.join(base_dir, f"{name}_v{version}{ext}")
        if not os.path.exists(v_path):
            return v_path
        version += 1
