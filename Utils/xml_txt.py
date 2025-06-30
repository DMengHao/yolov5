# -*- coding: utf-8 -*-
import argparse
import xml.etree.ElementTree as ET
import os
import shutil
import random
import logging
from tqdm import tqdm

# 创建保存文件夹
error_name_file = './Results/Error/'
if not os.path.exists(error_name_file):
    os.makedirs(error_name_file)
train_txt_path = r'./Results/labels/train/'
val_txt_path = r'./Results/labels/val/'
if not os.path.exists(train_txt_path):
    os.makedirs(train_txt_path)
if not os.path.exists(val_txt_path):
    os.makedirs(val_txt_path)
train_image_path = r'./Results/images/train/'
val_image_path = r'./Results/images/val/'
if not os.path.exists(train_image_path):
    os.makedirs(train_image_path)
if not os.path.exists(val_image_path):
    os.makedirs(val_image_path)

"1.核查图片是否缺少xml文件和xml文件是否有对应的图片"
def check_xml_image_name(path, classes_image):
    L1 = []
    L2 = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        file_type = os.path.splitext(file_path)[1]
        T = os.path.splitext(file_path)[0]
        file_name = os.path.join(path, os.path.splitext(file)[0])
        if file_type in classes_image:
            L1.append(T+file_type)
            file_name += '.xml'
            if not os.path.exists(file_name):
                logging.warning(f"Image file {file} has no matching XML file. Moving to error directory.")
                shutil.move(file_path, os.path.join(error_name_file, file))
        elif file_type == '.xml':
            L2.append(T+file_type)
            if not os.path.exists(file_name + '.jpg') and not os.path.exists(file_name + '.png') and not os.path.exists(
                    file_name + '.bmp') and not os.path.exists(file_name + '.psd') and not os.path.exists(file_name + '.gif') and not os.path.exists(file_name + '.webp') and not os.path.exists(file_name + '.svg'):
                logging.warning(f"XML file {file} has no matching image file. Moving to error directory.")
                shutil.move(file_path, os.path.join(error_name_file, file))
        else:
            shutil.move(file_path, os.path.join(error_name_file, file))


'2.得到训练和验证图片路径列表'
def getFile_name(file_dir, proportion):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] !='.xml':
                L.append(file)
    Length = len(L)
    random.shuffle(L)
    train = L[0:int(proportion*Length)]
    val = L[int(proportion*Length):]
    return train,val


# xml的(x,y,x,y)->(x,y,w,h)
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


# 3.转换xml文件为txt文件
def convert_annotation(xml_image_data_path,classes, image_id,train=False):
    if train:
        Temp = train_txt_path
    else:
        Temp = val_txt_path
    in_file = xml_image_data_path+image_id+'.xml'

    with open(in_file,'r', encoding='utf-8-sig') as file:
        content = file.read()
    if not content.strip():# 如果是空的则返回
        # shutil.move(Temp+image_id+'.jpg', os.path.join(error_name_file, image_id+'.jpg'))
        return None
    out_file = open(os.path.join(Temp,f'{image_id}.txt'), 'w', encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    out_file.close()
    return 1


# 3.保存图片
def save_image(xml_image_data_path,L,T:bool)->None:
    if T:
        image_path = train_image_path
    else:
        image_path = val_image_path
    L = [os.path.join(xml_image_data_path, i) for i in L]
    desc = 'save_train_image:' if T else 'save_val_image:'
    for i in tqdm(L, desc=desc):
        if os.path.exists(i):
            shutil.copy2(i, image_path)
        else:
            print("Warning: Source file %s does not exist!" % i)

def remove_bom_from_txt(folder):
    for file in os.listdir(folder):
        if file.endswith('.txt'):
            path = os.path.join(folder, file)
            with open(path, 'r', encoding='utf-8-sig') as f:
                context = f.read()
            with open(path, 'w', encoding='utf-8') as f:
                f.write(context)


def main(opt):
    print('#################### Start xml convert txt! ####################')
    # 1.核查iamge和xml
    check_xml_image_name(opt.xml_image_data_path, opt.classes_image)
    # 2.得到训练验证图片列表
    image_ids_train, image_ids_val = getFile_name(opt.xml_image_data_path, opt.proportion)
    # 3.保存训练验证图片
    save_image(opt.xml_image_data_path,image_ids_train, True)
    save_image(opt.xml_image_data_path,image_ids_val, False)
    # 4.转换训练验证为txt文件并保存
    for image_id in tqdm(image_ids_train,desc='save_train_txt:'):
        convert_annotation(opt.xml_image_data_path, opt.classes, os.path.splitext(image_id)[0], train=True)
    for image_id in tqdm(image_ids_val, desc='save_val_txt'):
        convert_annotation(opt.xml_image_data_path, opt.classes, os.path.splitext(image_id)[0], train=False)
    # 5. 核查训练与验证txt文件名称数量是否匹配
    dir1_list = os.listdir('./Results/images/train')
    dir2_list = os.listdir('./Results/labels/train')
    set1 = {os.path.splitext(m)[0] for m in dir1_list}
    set2 = {os.path.splitext(m)[0] for m in dir2_list}
    if len(dir1_list) > len(dir2_list):
        results = set1 - set2
        for i, result in enumerate(results):
            shutil.move('./Results/images/train/' + result+'.jpg', os.path.join('Results/Error/', result + '.jpg'))
    else:
        results = set2 - set1
        for i, result in enumerate(results):
            shutil.move('./Results/labels/train/' + result + '.txt', os.path.join('Results/Error/', result + '.txt'))

    dir1_list_ = os.listdir('./Results/images/val')
    dir2_list_ = os.listdir('./Results/labels/val')
    set1_ = {os.path.splitext(m)[0] for m in dir1_list_}
    set2_ = {os.path.splitext(m)[0] for m in dir2_list_}
    if len(dir1_list_) > len(dir2_list_):
        results_ = set1_ - set2_
        for i, result_ in enumerate(results_):
            shutil.move('./Results/images/val/' + result_ + '.jpg', os.path.join('Results/Error/', result_ + '.jpg'))
    else:
        results_ = set2_ - set1_
        for i, result_ in enumerate(results_):
            shutil.move('./Results/labels/val/' + result_ + '.txt', os.path.join('Results/Error/', result_ + '.txt'))
    print('#################### Xml convert txt end! ####################')

    remove_bom_from_txt('./Results/labels/train/')
    remove_bom_from_txt('./Results/labels/val/')



# ['坠砣', '支柱', '轻硬飘物','施工','危树','分段绝缘器异常','隔离开关','公路桥','电连接线','背后坠砣','隧道口']
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_image_data_path', type=str, default=r'//172.202.50.65/data/6C/日照港-数据/日照港-道口栏木机/道口2_20250619/道口2_20250619-整合/',help='xml和image路径')
    # '坠砣', '支柱', '轻硬飘物','施工','危树','分段绝缘器异常','隔离开关','公路桥','电连接线','背后坠砣','隧道口', '上跨线'
    # '火车','异物','栏木机-开启','栏木机-下落'
    # 'cat','dog'
    # '挂载车头','未挂载车头'
    parser.add_argument('--classes', type=list, default=['火车','异物','栏木机-开启','栏木机-下落'], help='xml转换为txt的标签列表，按照列表索引进行生成对应类别。')
    parser.add_argument('--proportion', type=float, default=0.8, help='训练验证比例,如0.9相当于训练集占总的90%')
    parser.add_argument('--classes_image',type=list, default=['.jpg','.png','bmp','.psd','.gif','.webp','.svg','.png'],help='图片类型')
    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt(True)
    main(opt)