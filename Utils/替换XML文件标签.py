import os
import glob

from tqdm import tqdm


# 传入文件(file),将旧内容(old_content)替换为新内容(new_content)
def replace(file, old_content, new_content):
    content = read_file(file)
    content = content.replace(old_content, new_content)
    rewrite_file(file, content)

# 读文件内容
def read_file(file):
    with open(file, encoding='UTF-8') as f:
        read_all = f.read()
        f.close()
    return read_all

# 写内容到文件
def rewrite_file(file, data):
    with open(file, 'w', encoding='UTF-8') as f:
        f.write(data)
        f.close()

if __name__ == '__main__':
    # folder_path = r"D:\data\jcxfbjc\nolabeled"  # 文件夹路径
    # file_type = ".xml"  # 文件类型
    # files = glob.glob(os.path.join(folder_path, f"*{file_type}"))
    files = glob.iglob(r"//172.202.50.65/data/6C/日照港-数据/日照港-道口栏木机/cat_dog/cat_dog_datasets/*.xml", recursive=True)

    for file in tqdm(files,desc='转换：'):
        replace(file, "dog", "异物")
        replace(file, "cat", "异物")

