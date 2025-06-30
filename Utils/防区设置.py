import cv2
import numpy as np


print('防区设置测试开始：')
# 加载图像
image_path = './道口防区图片/test_szks.jpg'
image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
height, width = image.shape[:2]

# 定义多边形顶点
points = np.array( [
    [ 906, 611 ],
    [ 975, 618 ],
    [ 952, 934 ],
    [ 836, 924 ]
  ])

# 判断点是否在多边形内的函数
def is_point_in_polygon(point, poly_points):
    """检查点是否在多边形内部或边界上"""
    result = cv2.pointPolygonTest(poly_points, point, False)
    return result >= 0  # 返回值>=0表示在内部或边界上

# 创建掩码
mask = np.zeros((height, width), dtype=np.uint8)
cv2.fillPoly(mask, [points], 255)

# 提取ROI区域
polygon_roi = cv2.bitwise_and(image, image, mask=mask)

# 标记多边形边界
marked_image = image.copy()
cv2.polylines(marked_image, [points], True, (0, 0, 255), thickness=2)


# 测试点坐标 (可以修改为需要检测的点)
test_point = (700, 600)  # 示例测试点

# 在图像上标记测试点并显示结果
point_color = (0, 255, 0)  # 默认绿色（在内部）
position_text = f"({test_point[0]}, {test_point[1]}) - Inside"

if not is_point_in_polygon(test_point, points):
    point_color = (0, 0, 255)  # 红色（在外部）
    position_text = f"({test_point[0]}, {test_point[1]}) - Outside"

# 绘制测试点和结果文本
cv2.circle(marked_image, test_point, 8, point_color, -1)
cv2.putText(marked_image, position_text, (test_point[0] + 15, test_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, point_color, 2)

cv2.imwrite("marked_polygon.jpg", marked_image)


# 控制台输出结果
print(f"Point {test_point} is inside polygon: {is_point_in_polygon(test_point, points)}")
print('防区设置测试结束！')