import os
import shutil

def rename_data(folder_name:str):
    fname_list = os.listdir(folder_name)
    start_num = 855
    for fname in fname_list:
        old_path = os.path.join(folder_name, fname)
        if fname.endswith(".jpg") or fname.endswith(".png"):
            new_fname = f"{start_num}.jpg"
            start_num += 1
            new_path = os.path.join("/home/user/PycharmProjects/iron_bar_sample_project/data", new_fname)

            shutil.copy2(old_path, new_path)

if __name__ == "__main__":
    rename_data("/home/user/Desktop/0912 ironbar")