import os


def rename_files_in_folder(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 筛选出所有jpg和txt文件
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]
    txt_files = [f for f in files if f.lower().endswith('.txt')]

    # 分别排序
    jpg_files.sort()
    txt_files.sort()

    # 分别重命名jpg文件
    for index, old_name in enumerate(jpg_files):
        # 构造新的文件名
        new_name = f"2_{index:03d}.jpg"

        # 获取旧文件的完整路径
        old_path = os.path.join(folder_path, old_name)
        # 获取新文件的完整路径
        new_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} to {new_name}")

    # 分别重命名txt文件
    for index, old_name in enumerate(txt_files):
        # 构造新的文件名
        new_name = f"2_{index:03d}.txt"

        # 获取旧文件的完整路径
        old_path = os.path.join(folder_path, old_name)
        # 获取新文件的完整路径
        new_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} to {new_name}")


# 输入文件夹路径
folder_path = r"你的文件夹路径"
rename_files_in_folder(folder_path)
