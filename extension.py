import os


def get_latest_pth_file(base_dir, extension):
    latest_path = None
    latest_mtime = -1

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(extension):
                full_path = os.path.join(root, file)
                mtime = os.path.getmtime(full_path)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_path = full_path

    if latest_path:
        return os.path.relpath(latest_path, base_dir)  # 상대경로로 반환
    return None