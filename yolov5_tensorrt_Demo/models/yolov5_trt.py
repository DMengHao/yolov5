# -*- coding:utf-8 -*-
import os
import time
import logging
import cv2
import numpy as np
import torch
import tensorrt as trt
from pycuda import driver
import pycuda.driver as cuda0
from collections import OrderedDict, namedtuple
from tqdm import tqdm
import sys
sys.path.append('/home/hhkj/dmh/RZG/AnalysisServer/utils')
from utils.utils import letterbox_xu,nms,scale_coords,save_pic,get_images_tensor,draw
import pycuda.autoinit
import yaml
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH 为了导入下方的models和utils
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 创建日志记录器并设置级别
logger = logging.getLogger('crossingdemo')
logger.setLevel(logging.INFO)
# 定义日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 创建一个控制台输出的处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
# 添加处理器
logger.addHandler(console_handler)

with open('/home/hhkj/dmh/RZG/AnalysisServer/cfg/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

points = np.array(config['crossing1_area'])
points_k = np.array(config['crossing2_area_k'])
points_b = np.array(config['crossing2_area_b'])
class yolov5_trt:
    def __init__(self, weights='./weights/new_data_20250222.engine', det_type='task', imgsz=1280, dev='cuda:0',
                 conf_thresh=0.25, iou_thresh=0.45, max_det=200, half=True):
        self.imgsz = imgsz
        self.device = int(dev.split(':')[-1])
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det
        self.half = half
        self.stride = 64 if self.imgsz == 1280 else 32
        self.det_type = det_type
        self.ctx = cuda0.Device(self.device).make_context()
        self.stream = driver.Stream()
        self.names = []
        if self.det_type == 'dk01':
            self.names = ['h','o','k','b']
        if self.det_type == 'dk02':
            self.names = ['h','o','k','b']
        if self.det_type == 'clly':
            self.names = ['ct','cx']
        if self.det_type == 'xf': # 巡防
            self.names = ['bird', 'person', 'train', 'animal', 'stone', 'yw', 'human', 'flood', 'mudslide', 'tree', 'float', 'square', 'box', 'light', 'cat', 'dog', 'traintop']
        if self.det_type == 'fire':
            self.names = ['fire', 'smoke']
        if self.det_type == 'km':
            self.names = ['km', 'person']
        if self.det_type == 'lb':
            # [橙色反光衣，    蓝色工装， 其他反光衣， 头盔,  未穿反光衣, 未戴头盔, ]
            self.names = ['orange_cloth', 'blue_cloth', 'cloth', 'helmet', 'nocloth', 'nohelmet', '']
        if self.det_type == 'hj':
            self.names = ['yw', 'cx']
        logger = trt.Logger(trt.Logger.INFO)
        with open(weights, 'rb') as f:
            self.runtime = trt.Runtime(logger)
            self.model = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.model.create_execution_context()
        self.bindings = OrderedDict()
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        for index in range(self.model.num_bindings):
            if trt.__version__ >= '8.6.1':
                name = self.model.get_tensor_name(index)
                dtype = trt.nptype(self.model.get_tensor_dtype(name))
                shape = tuple(self.model.get_tensor_shape(name))
            else:
                name = self.model.get_binding_name(index)
                dtype = trt.nptype(self.model.get_binding_dtype(index))
                shape = tuple(self.model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            del data
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())


    def preprocess(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float().to(self.device)

        temp = (torch.flip(img, dims=[2]).permute(2, 0, 1)).unsqueeze(0)
        im = letterbox_xu(temp, self.imgsz,auto=False, stride=self.stride)[0]
        img_shape = (img.shape[0], img.shape[1])
        im_shape = (im.shape[2], im.shape[3])

        im = torch.divide(im, 255.0)
        im = im.half() if self.half else im.float()
        return im, img_shape, im_shape


    def postprocess(self, pred, img_list, im_list):
        pred = nms(pred, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh, max_det=self.max_det)
        detections = []
        for k, det in enumerate(pred):
            H, W = img_list
            det[:, :4] = scale_coords(im_list, det[:, :4], img_list).round()
            coords = det[:, :4].long()
            coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, W - 1)
            coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, H - 1)

            classes = det[:, 5]
            confs = det[:, 4]

            detection = []
            for x1, y1, x2, y2, conf, cls in zip(*coords.T, confs, classes):
                cls_str = self.names[int(cls)]
                detection.append({'id': int(cls),
                                  'class': cls_str,
                                  'conf': float(conf),
                                  'box': [x1.item(), y1.item(), x2.item(), y2.item()]})
            detections.append(detection)
        return detections


    def infer(self, tensor_data):
        try:
            self.ctx.push()
            process_start_time = time.time()
            input_data, img_shape, im_shape = self.preprocess(tensor_data)
            logger.info(f'前处理用时：{(time.time() - process_start_time) * 1000:.4f}ms')
            infer_start_time = time.time()
            self.binding_addrs['images'] = int(input_data.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            self.stream.synchronize()  # 确保所有计算已完成，阻塞调用
            logger.info(f'推理用时：{(time.time() - infer_start_time) * 1000:.4f}ms')
            post_start_time = time.time()
            preds = self.bindings['output'].data
            detections = self.postprocess(preds, img_shape, im_shape)
            logger.info(f'后处理用时：{(time.time() - post_start_time) * 1000:.4f}ms')
            return detections
        finally:
            self.ctx.pop()

    
    def logical_dk01(self, image, detections):
        logical_start_time = time.time()
        temp = False
        result = []

        for i, detection in enumerate(detections):
            counts = 0
            for j, dets in enumerate(detection):
                if dets['class'] == 'b':
                    counts = counts + 1
            if counts >= 2:
                for j, dets in enumerate(detection):
                    if dets['class'] == 'o' and dets['box'][2]-dets['box'][0] >= 20 and dets['box'][3] - dets['box'][1] >= 20:
                        box = dets['box']
                        center_point = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                        if cv2.pointPolygonTest(points, center_point, False) >= 0:
                            temp = True
                            result.append(dets)
        logger.info(f'逻辑处理用时:{(time.time() - logical_start_time) * 1000 :.4f}ms')
        return temp, result


    def logical_dk02(self, image, detections):
        logical_start_time = time.time()
        temp = False
        result = []
        for i,detection in enumerate(detections):
            counts_k = 0
            counts_b = 0
            for j,dets in enumerate(detection):
                if dets['class'] == 'k':
                    counts_k = counts_k + 1
                if dets['class'] == 'b':
                    counts_b = counts_b + 1
            if counts_k >= 1:
                for j,dets in enumerate(detection):
                    if dets['class'] == 'o':
                        box = dets['box']
                        center_point = (int((box[0]+box[2])/2), int((box[1]+box[3])/2))
                        if cv2.pointPolygonTest(points_k, center_point, False)>=0:
                            temp = True
                            result.append(dets)
            if counts_b >= 1:
                for j,dets in enumerate(detection):
                    if dets['class'] == 'o':
                        box = dets['box']
                        center_point = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                        if cv2.pointPolygonTest(points_b, center_point, False) >= 0:
                            temp = True
                            result.append(dets)
        logger.info(f'逻辑处理用时：{(time.time() - logical_start_time) * 1000 :.4f}ms')
        return temp, result


    def logical_clly_sort(self, detections, h, w):
        filtration_results = []
        for i, detection in enumerate(detections[0]):
            if detection['class'] == 'cx' and abs(h-detection['box'][3]) >=50:
                filtration_results.append(detection)
        return filtration_results


    def logical_km(self, detections):
        logical_start_time = time.time()
        temp = False
        result = []
        for i,detection in enumerate(detections):
            for j,dets in enumerate(detection):
                if dets['class'] == 'person':
                    temp = False
                    result = detections[0]
                    break
                if dets['class'] == 'km':
                    temp = True
                    result.append(dets)
        logger.info(f'逻辑处理用时：{(time.time() - logical_start_time) * 1000 :.4f}ms')
        return temp, result
    

    def logical_xf(self,detections):
        logical_start_time = time.time()
        temp = False
        result = []
        for i,detection in enumerate(detections):
            for j,dets in enumerate(detection):
                if dets['class'] == 'person':
                    temp = True
                    result.append(dets)
        logger.info(f'逻辑处理用时：{(time.time() - logical_start_time) * 1000 :.4f}ms')
        return temp, result


    def __del__(self):
        self.ctx.pop()
        del self.context
        del self.model
        del self.runtime


if __name__ == '__main__':

    Video = False
    if Video:
        ENGINE_PATH = ROOT / "weights/20250616_dk01.engine"
        yolov5_api = yolov5_trt(weights=ENGINE_PATH)
        color_map = {'o': (0, 0, 255)}
        logger.info('开始推理视频')
        
        cap = cv2.VideoCapture('/home/hhkj/dmh/RZG/crossingdemo/video/dk01.mp4')
        i =0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                result = yolov5_api.infer(frame)
                is_, last_results = yolov5_api.logical(frame, result)
                print('##############################################################################################################', is_)
                if is_:
                    logger.info('#########################################################################################################################')
                    draw(frame, last_results, os.path.join('/home/hhkj/dmh/RZG/crossingdemo/results/video_dk01',str(i) + '.jpg'), color_map)
            else: 
                break
            i = i + 1
            
    else:
        ENGINE_PATH = ROOT / "weights/20250616_dk01.engine"
        color_map = {'o': (0, 0, 255)}
        IMAGE_PATH = ROOT / "images/dk01"  # 待检测图片路径
        OUTPUT_PATH = ROOT / "results/dk01"  # 输出图片路径
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        yolov5_api = yolov5_trt(weights=ENGINE_PATH)
        image_path = os.listdir(IMAGE_PATH)
        print('开始测试：')
        for i,img_path in tqdm(enumerate(image_path), total=len(image_path)):
            image = cv2.imdecode(np.fromfile(os.path.join(IMAGE_PATH,img_path), dtype=np.uint8), -1)
            results = yolov5_api.infer(image)
            _,last_results = yolov5_api.logical(image, results)
            detections = last_results
            draw(image, detections, os.path.join(OUTPUT_PATH,img_path), color_map)
        print('测试完毕！')