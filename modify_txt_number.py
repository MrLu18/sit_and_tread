import os

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 处理每一行
    new_lines = []
    for line in lines:
        if line.startswith('3') or line.startswith('2') : #根据实际情况改
            continue  # 删除以1或2开头的行
        elif line.startswith('3'):
            new_lines.append('1' + line[1:])  # 将3开头的行改为1
        else:
            new_lines.append(line)  # 其他行保持不变

    # 将处理后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(new_lines)

def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):  # 只处理文本文件
                file_path = os.path.join(root, file)
                process_file(file_path)

# 输入文件夹路径
folder_path = input("请输入文件夹路径: ")
process_folder(folder_path)
print("处理完成！")