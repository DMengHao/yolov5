import os
import cv2
from tqdm import tqdm

video_path = "F:/手指口述/20250606_102722清晰工人手指1.mp4"
frame_save_path = "//172.202.50.65/data/6C/日照港-数据/日照港-手指口述/person/"

if not os.path.exists(frame_save_path):
    os.makedirs(frame_save_path)

print("==========Transform Start!==========")
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

i = 0
j = 0
for i in tqdm(range(frame_count), desc='处理'):
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {i + 1}")
        break
    if i%15 == 0:
        frame_save_path_temp = os.path.join(frame_save_path, f"20250606_102722清晰工人手指1{j+1}" + ".jpg")
        cv2.imencode('.jpg', frame)[1].tofile(frame_save_path_temp)
    j = j + 1
print("==========Transform End!==========")
cap.release()