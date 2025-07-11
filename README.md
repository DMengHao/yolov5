# yolov5
## 1.配置yolov5所需环境：

下载安装cuda12.1、cudnn8.9.7、TensorRT8.6.1参考链接：https://blog.csdn.net/qq_45754436/article/details/140323984?spm=1011.2415.3001.5331

创建虚拟环境机器配置需要的环境：

conda create -n yolov5 python=3.9

conda activate yolov5

cd ./yolov5

pip install -r requirements.txt

## 2.制作数据集：

使用labelimg进行标注图片（注意先生成对应的xml文件，如果生成txt文件之后生成的标签是数字，而xml文件标签还是原始标签，便于查看）

参考链接：https://blog.csdn.net/knighthood2001/article/details/125883343?fromshare=blogdetail&sharetype=blogdetail&sharerId=125883343&sharerefer=PC&sharesource=qq_45754436&sharefrom=from_link

xml生成txt脚本放于：./utils/xml_txt.py

txt生成xml脚本放于：./utils/txt_xml.py

## 3.训练

3.1.官网下载对应的预训练权重以yolov5s6为例，如果已经下载放置到yolov5根目录下

3.2.修改配置文件

在data路径下创建文件填写以下内容：自己的类别数量和对应的训练验证数据集路径

```python
path:  # dataset root dir
train:
  - /train1
  - /train2
  - /train3
  - /train4
val:
  - /val1
  - /val2
  - /val3
  - /val4
test: # test images (optional)
nc: 4  # number of classes names列表当中存放类型列表
names: ['h','o','k','b']
```

3.3.修改train.sh里面文件

python -m torch.distributed.run --nproc_per_node 4 train.py --batch-size 128 --weights ./yolov5s6.pt --cfg ./models/hub/yolov5s6.yaml --epochs 100 --project ./runs_dk01/train --data ./data/dk01.yaml --device 0,1,2,3 --cache disk 

3.4.运行train.sh

```python
# 1.使用sh train.sh
sh train.sh
# 2.使用nohup命令 该命令会在服务器关机之后继续训练
nohup sh train.sh > ./log/train.log 2>&1 &
tail -f ./log/train.log
exit
```

## 4.TensorRT推理

4.1根据所需配置需要的onnx包

4.2导出engine

```python
sh export.sh

# 根据参数进行更换自己训练好的路径
#! /bin/bash
# python ./export.py --data ./data/dk01.yaml --weights ./weights/crossing01.pt --include onnx engine --batch-size 1 --inplace --half --imgsz 1280
```

4.3推理

打开yolov5_tensorrt_Demo将导出的engine放置到新建的weights文件夹下，修改yolov5_tensorrt_Demo/models/yolov5_trt.py下的self.det_type=="dk01":下一行的self.names列表，修改为自己训练的标签名称。

自己查看调试yolov5_tensorrt_Demo代码





