import os
import cv2
import json
import sys
import logging
from pathlib import Path
from models.yolov5_trt import yolov5_trt
from utils.rtsp_stream import RTSPVideoStream
from utils.utils import draw, save_box_image

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

with open(ROOT /"cfg/camera.json", "r") as f:
    config = json.load(f)

stream_url = config["cameras"][0]['cameraRtsp']


# 初始化模型
yolov5_api_dk01 = yolov5_trt(weights='/home/hhkj/dmh/RZG/AnalysisServer/weights/20250616_dk01.engine', det_type='dk01', dev='cuda:0')
yolov5_api_dk02 = yolov5_trt(weights='/home/hhkj/dmh/RZG/AnalysisServer/weights/20250616_dk02.engine', det_type='dk02', dev='cuda:0')
yolov5_api_clly = yolov5_trt(weights='/home/hhkj/dmh/RZG/AnalysisServer/weights/20250617_clly.engine', det_type='clly', dev='cuda:0')
yolov5_api_km = yolov5_trt(weights='/home/hhkj/dmh/RZG/AnalysisServer/weights/door_0616.engine', det_type='km', dev='cuda:0')
yolov5_api_xf = yolov5_trt(weights='/home/hhkj/dmh/RZG/AnalysisServer/weights/xf.engine', det_type='xf', dev='cuda:0')

# 初始化拉流器
stream = RTSPVideoStream(rtsp_url=stream_url)
h, w = stream.get_resolution()
# h, w = 1920, 1080
stream.start()

OUTPUT_PATH_dk01 = ROOT /"results/dk01"  # 道口1输出图片路径
if not os.path.exists(OUTPUT_PATH_dk01):
    os.makedirs(OUTPUT_PATH_dk01)
OUTPUT_PATH_dk02 = ROOT /"results/dk02"  # 道口2输出图片路径
if not os.path.exists(OUTPUT_PATH_dk02):
    os.makedirs(OUTPUT_PATH_dk02)
OUTPUT_PATH_clly = ROOT /"results/clly"  # 车辆溜逸输出图片路径
if not os.path.exists(OUTPUT_PATH_clly):
    os.makedirs(OUTPUT_PATH_clly)
OUTPUT_PATH_km = ROOT /"results/km" # 开门输出图片路径
if not os.path.exists(OUTPUT_PATH_km):
    os.makedirs(OUTPUT_PATH_km)
OUTPUT_PATH_xf = ROOT /"results/xf" # 开门输出图片路径
if not os.path.exists(OUTPUT_PATH_xf):
    os.makedirs(OUTPUT_PATH_xf)



color_map = {'o': (0, 255, 0)}



vehicle_tracks = []
vehicle_images = []
k = 0
while True:
    frame = stream.read()
    if frame is None:
        print("⚠️ 拉流结束或帧错误")
        break
    # 是否保存逐帧拉流图片
    if False:
        output_dir = ROOT / "output_frames"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'frame_{k}.jpg')
        cv2.imwrite(save_path, frame)

    # dk01 推理
    dk01_ok, dk01_results = yolov5_api_dk01.logical_dk01(frame, yolov5_api_dk01.infer(frame))
    if dk01_ok:
        draw(frame, dk01_results, os.path.join(OUTPUT_PATH_dk01,f"{k}.jpg"), color_map)
    # dk02 推理
    dk02_ok, dk02_results = yolov5_api_dk02.logical_dk02(frame, yolov5_api_dk02.infer(frame))
    if dk02_ok:
        draw(frame, dk02_results, os.path.join(OUTPUT_PATH_dk02,f"{k}.jpg"), color_map)
    # clly 推理
    clly_results = yolov5_api_clly.logical_clly_sort(yolov5_api_clly.infer(frame), h, w)
    # km 推理
    km_ok, km_results = yolov5_api_km.logical_km(yolov5_api_km.infer(frame))
    if km_ok:
        draw(frame, km_results, os.path.join(OUTPUT_PATH_km,f"{k}.jpg"), color_map)
    # xf 推理
    xf_ok, xf_results = yolov5_api_xf.logical_xf(yolov5_api_xf.infer(frame))
    if xf_ok:
        draw(frame, xf_results, os.path.join(OUTPUT_PATH_xf,f"{k}"), color_map)
        save_box_image(frame, xf_results, k)
    if k%5 == 0:
        vehicle_images.append(frame)
        vehicle_tracks.append(clly_results)
    if len(vehicle_images) == 2 and len(vehicle_tracks) == 2:
        tracks0 = vehicle_tracks[0]
        tracks1 = vehicle_tracks[1]
        zip_tuple = []
        for i, tracks0_box in enumerate(tracks0):
            for j, tracks1_box in enumerate(tracks1):
                if abs(tracks0_box['box'][0]-tracks1_box['box'][0])<=100 and abs(tracks0_box['box'][1]-tracks1_box['box'][1])<=100 and abs(tracks0_box['box'][2]-tracks1_box['box'][2])<=100 and abs(tracks0_box['box'][3]-tracks1_box['box'][3])<=100:
                    zip_tuple.append((tracks1_box,tracks0_box))
        for i, Temp in enumerate(zip_tuple):
            temp0,temp1 = Temp
            if abs((temp0['box'][2]-temp0['box'][0])-(temp1['box'][2]-temp1['box'][0]))>40 or abs((temp0['box'][3]-temp0['box'][1])-(temp1['box'][3]-temp1['box'][1]))>40:
                draw(vehicle_images[0], vehicle_tracks[0], os.path.join(OUTPUT_PATH_clly,'帧数'+ str(k-5) + '原图' + '.jpg'), color_map)
                draw(vehicle_images[1], vehicle_tracks[1], os.path.join(OUTPUT_PATH_clly,'帧数'+ str(k) + '车辆溜逸' + '.jpg'), color_map)
                break
        del vehicle_images[0]
        del vehicle_tracks[0]
    k += 1

stream.stop()