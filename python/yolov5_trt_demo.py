# -*- coding:utf-8 -*-
import argparse
import logging
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
import tensorrt as trt
from pycuda import driver
import pycuda.driver as cuda0
import torchvision
from torch.nn import functional as F
from collections import OrderedDict, namedtuple
from torchvision.transforms import Resize
import time
import itertools
from torchvision.ops import box_iou
import pycuda.autoinit


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


# 创建日志记录器并设置级别
logger = logging.getLogger('yolov5_trt_demo')
logger.setLevel(logging.INFO)
# 定义日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 创建一个控制台输出的处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
# 添加处理器
logger.addHandler(console_handler)


def save_pic(image, save_path):
    cv2.imencode('.jpg', image)[1].tofile(save_path)


def draw_pic(img_read, text, x1, y1, x2, y2):
    cv2.rectangle(img_read, (x1, y1), (x2, y2), (0, 0, 220), 2)
    cv2.putText(img_read, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 220), 2)


def draw(img_tensor, detections, img_names, image_name):
    save_path = ROOT /'results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, img in enumerate(img_tensor):
        image = img.cpu().numpy().astype(np.uint8)
        if os.path.splitext(img_names[i])[1] in ['.jpg', '.JPG', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.svg', '.pfg']:
            for item in detections[i]:
                text = item['class']
                box = item['box']
                conf = item['conf']
                label = f'{text} ({conf:.2f})'
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                draw_pic(image, label, x1, y1, x2, y2)  # 使用方法打包汇框和标签，便于维护
            pic_save_path = f'{save_path}/{image_name+os.path.basename(img_names[i])}'
            save_pic(image, pic_save_path)
            print(f'保存推理结果图路径：{pic_save_path}')


def IoU(box1, box2) -> float:
    weight = max(min(box1[2], box2[2]) - max(box1[0], box2[0]), 0)
    height = max(min(box1[3], box2[3]) - max(box1[1], box2[1]), 0)
    s_inter = weight * height
    s_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    s_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    s_union = s_box1 + s_box2 - s_inter
    return s_inter / s_union


def nms(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    min_wh, max_wh = 2, 7680
    max_nms = 30000
    time_limit = 10.0
    redundant = True
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        # 0.2375ms vs 0.1187ms
        # idxs = torch.arange(boxes.shape[0], device=boxes.device)
        # i = torchvision.ops.batched_nms(boxes=boxes, scores=scores, idxs=idxs, iou_threshold=iou_thres)
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break
    return output


def clip_coords(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])
        boxes[:, 1].clamp_(0, shape[0])
        boxes[:, 2].clamp_(0, shape[1])
        boxes[:, 3].clamp_(0, shape[0])
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]) # 计算输入图像相较于原始图像的缩放比例
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def letterbox_xu(im, new_shape=(640, 640), auto=True, scaleFill=False, scaleup=True, stride=32): # im->(B,3,1280,1280)
    shape = im.shape[2:]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) #四舍五入，取整
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        torch_resize = Resize([new_unpad[1], new_unpad[0]])
        im = torch_resize(im)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    p2d = (left, right, top, bottom)
    out = F.pad(im, p2d, 'constant', 114.0 / 255.0)
    out = out.contiguous()
    return out, ratio, (dw, dh)


class yolov5_trt_demo:
    def __init__(self, weights='./weights/new_data_20250222.engine', det_type='task1', imgsz=1280, dev='cuda:0',
                 conf_thresh=0.25, iou_thresh=0.45, max_det=200, half=True):
        # 参数初始化
        self.imgsz = imgsz
        self.device = int(dev.split(':')[-1])
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det
        self.half = half
        self.stride = 64 if self.imgsz == 1280 else 32
        self.det_type = det_type
        self.names = []
        if self.det_type == 'task2':
            self.names = ["", "", ""]
        # CUDA上下文和创建
        self.ctx = cuda0.Device(self.device).make_context() # 为当前线程在指定GPU设备上创建独立的执行环境，每个需要GPU操作的线程必须有自己的上下文
        
        self.stream = driver.Stream() # 创建异步操作队列
        # 引擎反序列化
        logger = trt.Logger(trt.Logger.INFO)
        with open(weights, 'rb') as f:
            self.runtime = trt.Runtime(logger)
            self.model = self.runtime.deserialize_cuda_engine(f.read())
        # 创建执行上下文
        self.context = self.model.create_execution_context()
        # 绑定输入输出内容
        self.bindings = OrderedDict() # 创建有序字典
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr')) #
        for index in range(self.model.num_bindings):
            if trt.__version__ <= '8.6.1':
                name = self.model.get_tensor_name(index)
                dtype = trt.nptype(self.model.get_tensor_dtype(name))
                shape = tuple(self.model.get_tensor_shape(name))
            else:
                name = self.model.get_binding_name(index)
                dtype = trt.nptype(self.model.get_binding_dtype(index))
                shape = tuple(self.model.get_binding_shape(index))
            # 分配GPU内存
            data = torch.empty(shape, dtype=torch.float16 if dtype == np.float16 else torch.float32, device=self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            del data
        # 创建绑定地址字典
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())


    def preprocess(self, img):
        img_shape =[]
        im_shape = []
        if isinstance(img, list):
            temp = [(torch.flip(data, dims=[2]).permute(2, 0, 1)).unsqueeze(0) for data in img]
            temp = torch.cat(temp, dim=0)
        else:
            temp = (torch.flip(img, dims=[2]).permute(2, 0, 1)).unsqueeze(0)
        im = letterbox_xu(temp, self.imgsz,auto=False, stride=self.stride)[0]
        for i in img:
            img_shape.append((i.shape[0], i.shape[1]))
        for i in im:
            im_shape.append((i.shape[1],i.shape[2]))
        im = torch.divide(im, 255.0)
        im = im.half() if self.half else im.float()
        return im, img_shape, im_shape


    def postprocess(self, pred, img_list, im_list):
        pred = nms(pred, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh, max_det=self.max_det)
        detections = []
        for k, det in enumerate(pred):
            H, W = img_list[k]
            det[:, :4] = scale_coords(im_list[k], det[:, :4], img_list[k]).round()
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
            self.ctx.push() # 激活当前线程的CUDA上下文
            input_data, img_shape, im_shape = self.preprocess(tensor_data)
            self.context.execute_async_v2(
                bindings=[int(input_data.data_ptr()), int(self.bindings['output'].ptr)],
                stream_handle=self.stream.handle
            )
            self.stream.synchronize()  # 确保所有计算已完成，阻塞调用
            detections = self.postprocess(self.bindings['output'].data, img_shape, im_shape)
            return detections
        finally:
            self.ctx.pop()

    def __del__(self):
        self.ctx.pop()
        del self.context
        del self.model
        del self.runtime


def get_images_tensor(path):
    device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')
    tensor_data = []
    image_path_list = os.listdir(path)
    all_image_list = [os.path.join(path,i) for i in image_path_list]
    for i, image_path in enumerate(all_image_list):
        img_read = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        img_read = torch.from_numpy(img_read).float()
        img_read = img_read.to(device)
        tensor_data.append(img_read)
    return tensor_data, image_path_list


def parse_args(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_path', type=str, default=ROOT /'weights/20250408_task2_b1.engine', help='task2 engine path')
    parser.add_argument('--detect_path', type=str, default=ROOT / 'images', help='detect path')
    parser.add_argument('--save_path', type=str, default=ROOT / 'results', help='save path')
    parser.add_argument('--input_image_numbers', type=int, default=1, help='batch size')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    input_image_numbers = opt['input_image_numbers']
    engine_path = opt['engine_path']
    tensor_data, img_names = get_images_tensor(opt['detect_path'])
    detect_api_task2 = yolov5_trt_demo(weights=engine_path, det_type='task2', imgsz=1280, dev='cuda:0', conf_thresh=0.25)
    for i in range(len(tensor_data)+1):
        if i == 0 or i % input_image_numbers != 0:
            continue
        if i % input_image_numbers == 0:
            start_time = time.time()
            result2 = detect_api_task2.infer(tensor_data[i - input_image_numbers:i])
            logger.info(f"#########################{(time.time()-start_time)*1000:.2f}ms")
            if True:
                draw(tensor_data[i - input_image_numbers:i], result2, img_names[i - input_image_numbers:i], 'task2__')

if __name__ == '__main__':
    opt = parse_args(known=True)
    main(vars(opt))