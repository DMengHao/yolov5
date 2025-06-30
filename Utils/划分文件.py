import math
import os
import shutil
from tqdm import tqdm

image_path = r'//172.202.50.65/data/6C/日照港-数据/日照港-道口栏木机/道口2_20250619'
image_number = 100

files = os.listdir(image_path)
temp_files = []
for file in files:
    if os.path.splitext(file)[1] == '.jpg':
        temp_files.append(os.path.join(image_path, file))
people_numbers = math.ceil(len(temp_files)/image_number)
result = []
print('开始拆分文件夹：')
for i in range(people_numbers):
    temp = temp_files[i*image_number:(i+1)*image_number]
    result.append(temp)
for i in range(len(result)):
    if not os.path.exists(image_path+f'/folder_{i+1}'):
        os.makedirs(image_path+f'/folder_{i+1}')
    for file in tqdm(result[i], desc=f'拷贝文件至folder_{i+1}'):
        shutil.move(file, os.path.join(image_path+f'/folder_{i+1}',os.path.basename(file)))
        if os.path.exists(os.path.splitext(file)[0]+'.xml'):
            shutil.move(os.path.splitext(file)[0]+'.xml', os.path.join(image_path + f'/folder_{i+1}', os.path.splitext(os.path.basename(file))[0]+'.xml'))
print('拆分文件夹完成！')