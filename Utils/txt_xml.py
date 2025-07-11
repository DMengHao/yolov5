# .txt-->.xml
# ! /usr/bin/python
# -*- coding:UTF-8 -*-
import os
import numpy as np
from PIL import Image
list = []
def txt_to_xml(txt_path, img_path, xml_path):
    if not os.path.exists(xml_path):
        os.makedirs(xml_path)
    # Label actual sequential numbering, No.:Label name.
    dict = {
        '0': "bicycle",
        '1': "bus",
        '2': "car",
        '3': "motorbike",
        '4': "truck",
        '5':"train",
        '6':"person",
    }
    files = os.listdir(txt_path)
    # 用于存储 "老图"
    pre_img_name = ''
    img_list = os.listdir(img_path)
    # 3.遍历文件夹
    for i, name in enumerate(files):
        # 许多人文件夹里有该文件，默认的也删不掉，那就直接pass
        # if name == "desktop.ini":
        #     continue
        print(name)
        # 4.打开txt
        txtFile = open(txt_path + name)
        # 读取所有内容
        txtList = txtFile.readlines()
        if not txtList:  # 如果txt文件为空
            img_name = name[:-4]
            # 生成一个空的VOC格式的XML文件
            xml_file = open(os.path.join(xml_path, img_name + '.xml'), 'w', encoding='utf-8')
            pil_image = Image.open(os.path.join(img_path, img_name + ".jpg"))
            pic = np.array(pil_image)
            Pheight, Pwidth, Pdepth = pic.shape

            xml_file.write('<annotation>\n')
            xml_file.write('    <folder>VOC2007</folder>\n')
            xml_file.write('    <filename>' + img_name + '.jpg' + '</filename>\n')
            xml_file.write('<source>\n')
            xml_file.write('<database>orgaquant</database>\n')
            xml_file.write('<annotation>organoids</annotation>\n')
            xml_file.write('</source>\n')
            xml_file.write('    <size>\n')
            xml_file.write('        <width>' + str(Pwidth) + '</width>\n')
            xml_file.write('        <height>' + str(Pheight) + '</height>\n')
            xml_file.write('        <depth>' + str(Pdepth) + '</depth>\n')
            xml_file.write('    </size>\n')
            xml_file.write('    <segmented>0</segmented>\n')
            xml_file.write('</annotation>')

            xml_file.close()
            list.append(os.path.join(xml_path, img_name + '.xml'))
            continue  # 跳过当前循环的剩余部分

        # 读取图片名称
        img_name = name[:-4]
        pil_image = Image.open(os.path.join(img_path, img_list[i]))

        # 将 PIL 图像对象转换为 NumPy 数组
        pic = np.array(pil_image)
        # pic = cv2.imread(os.path.join(img_path, img_name + ".jpg"))
        # 获取图像大小信息
        Pheight, Pwidth, Pdepth = pic.shape
        # 5.遍历txt文件中每行内容
        for row in txtList:
            # 按' '分割txt的一行的内容
            oneline = row.strip().split(" ")
            # 遇到的是一张新图片
            if img_name != pre_img_name:
                # 6.新建xml文件
                xml_file = open(os.path.join(xml_path + img_name + '.xml'), 'w', encoding='utf-8')
                xml_file.write('<annotation>\n')
                xml_file.write('    <folder>VOC2007</folder>\n')
                xml_file.write('    <filename>' + img_name + '.jpg' + '</filename>\n')
                xml_file.write('    <source>\n')
                xml_file.write('        <database>orgaquant</database>\n')
                xml_file.write('        <annotation>organoids</annotation>\n')
                xml_file.write('    </source>\n')
                xml_file.write('    <size>\n')
                xml_file.write('        <width>' + str(Pwidth) + '</width>\n')
                xml_file.write('        <height>' + str(Pheight) + '</height>\n')
                xml_file.write('        <depth>' + str(Pdepth) + '</depth>\n')
                xml_file.write('    </size>\n')
                xml_file.write('    <segmented>0</segmented>\n')
                xml_file.write('    <object>\n')
                xml_file.write('        <name>' + dict[oneline[0]] + '</name>\n')
                xml_file.write('        <pose>Unspecified</pose>\n')
                xml_file.write('        <truncated>0</truncated>\n')
                xml_file.write('        <difficult>0</difficult>\n')
                xml_file.write('        <bndbox>\n')
                xml_file.write('            <xmin>' + str(
                    int(((float(oneline[1])) * Pwidth) - (float(oneline[3])) * 0.5 * Pwidth)) + '</xmin>\n')
                xml_file.write('            <ymin>' + str(
                    int(((float(oneline[2])) * Pheight) - (float(oneline[4])) * 0.5 * Pheight)) + '</ymin>\n')
                xml_file.write('            <xmax>' + str(
                    int(((float(oneline[1])) * Pwidth) + (float(oneline[3])) * 0.5 * Pwidth)) + '</xmax>\n')
                xml_file.write('            <ymax>' + str(
                    int(((float(oneline[2])) * Pheight) + (float(oneline[4])) * 0.5 * Pheight)) + '</ymax>\n')
                xml_file.write('        </bndbox>\n')
                xml_file.write('    </object>\n')
                xml_file.close()
                pre_img_name = img_name  # 将其设为"老"图
            else:  # 不是新图而是"老图"
                # 7.同一张图片，只需要追加写入object
                xml_file = open((xml_path + img_name + '.xml'), 'a', encoding='utf-8')
                xml_file.write('    <object>\n')
                xml_file.write('        <name>' + dict[oneline[0]] + '</name>\n')
                '''  按需添加这里和上面
                xml_file.write('        <pose>Unspecified</pose>\n')
                xml_file.write('        <truncated>0</truncated>\n')
                xml_file.write('        <difficult>0</difficult>\n')
                '''
                xml_file.write('        <bndbox>\n')
                xml_file.write('            <xmin>' + str(
                    int(((float(oneline[1])) * Pwidth) - (float(oneline[3])) * 0.5 * Pwidth)) + '</xmin>\n')
                xml_file.write('            <ymin>' + str(
                    int(((float(oneline[2])) * Pheight) - (float(oneline[4])) * 0.5 * Pheight)) + '</ymin>\n')
                xml_file.write('            <xmax>' + str(
                    int(((float(oneline[1])) * Pwidth) + (float(oneline[3])) * 0.5 * Pwidth)) + '</xmax>\n')
                xml_file.write('            <ymax>' + str(
                    int(((float(oneline[2])) * Pheight) + (float(oneline[4])) * 0.5 * Pheight)) + '</ymax>\n')
                xml_file.write('        </bndbox>\n')
                xml_file.write('    </object>\n')
                xml_file.close()

        # 8.读完txt文件最后写入</annotation>
        xml_file1 = open((xml_path + pre_img_name + '.xml'), 'a', encoding='utf-8')
        xml_file1.write('</annotation>')
        list.append((xml_path + pre_img_name + '.xml'))
        xml_file1.close()
    print("Done !")
    print(len(list))


# 修改成自己的文件夹 注意文件夹最后要加上/
txt_to_xml("C:/Users/hhkj/Desktop/Temp/laobao/labels/", "C:/Users/hhkj/Desktop/Temp/laobao/images/", "C:/Users/hhkj/Desktop/Temp/laobao/xml/")
