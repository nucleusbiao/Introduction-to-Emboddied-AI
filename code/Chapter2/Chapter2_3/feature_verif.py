import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional, Dict
import math


class DetectionResult:
    def __init__(self):
        self.position = (0, 0)  # (u, v) - 检测框底部中点坐标
        self.confidence = 0.0
        self.bbox = (0, 0, 0, 0)  # (x, y, width, height) - 完整边界框信息
        self.world_position = (0.0, 0.0, 0.0)  # (x, y, z) - 世界坐标系中的位置
        self.class_id = 0  # 类别ID
        self.class_name = ""  # 类别名称


class Point3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class Point2D:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class CameraIntrinsics:
    def __init__(self, fx=500.0, fy=500.0, cx=640.0, cy=360.0,
                 k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2


class TransformMatrix:
    def __init__(self):
        self.data = np.eye(4, dtype=np.float64)

    def __matmul__(self, other):
        result = TransformMatrix()
        result.data = self.data @ other.data
        return result


class BackProjector:
    def __init__(self, intrinsics: CameraIntrinsics, extrinsic: TransformMatrix):
        self.intrinsics_ = intrinsics
        self.extrinsic_ = extrinsic
        self.calculate_inverse_extrinsic()

    def calculate_inverse_extrinsic(self):
        R = self.extrinsic_.data[0:3, 0:3]
        t = self.extrinsic_.data[0:3, 3]

        Rt_t = R.T @ t

        self.extrinsic_inv_ = TransformMatrix()
        self.extrinsic_inv_.data[0:3, 0:3] = R.T
        self.extrinsic_inv_.data[0:3, 3] = -Rt_t
        self.extrinsic_inv_.data[3, 3] = 1.0

    def pixel_to_image(self, pixel_point: Point2D) -> Point2D:
        x = pixel_point.x - self.intrinsics_.cx
        y = pixel_point.y - self.intrinsics_.cy
        return Point2D(x, y)

    def remove_distortion(self, distorted_point: Point2D) -> Point2D:
        x_distorted = distorted_point.x / self.intrinsics_.fx
        y_distorted = distorted_point.y / self.intrinsics_.fy

        x = x_distorted
        y = y_distorted

        for _ in range(10):
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2

            radial_factor = 1.0 + self.intrinsics_.k1 * r2 + self.intrinsics_.k2 * r4 + self.intrinsics_.k3 * r6
            delta_x = 2 * self.intrinsics_.p1 * x * y + self.intrinsics_.p2 * (r2 + 2 * x * x)
            delta_y = self.intrinsics_.p1 * (r2 + 2 * y * y) + 2 * self.intrinsics_.p2 * x * y

            x_ideal = (x_distorted - delta_x) / radial_factor
            y_ideal = (y_distorted - delta_y) / radial_factor

            x = x_ideal
            y = y_ideal

        x_ideal = x * self.intrinsics_.fx
        y_ideal = y * self.intrinsics_.fy

        return Point2D(x_ideal, y_ideal)

    def image_to_camera_ray(self, ideal_point: Point2D) -> Point3D:
        x = ideal_point.x / self.intrinsics_.fx
        y = ideal_point.y / self.intrinsics_.fy
        return Point3D(x, y, 1.0)

    def camera_to_world(self, camera_point: Point3D) -> Point3D:
        camera_homogeneous = np.array([camera_point.x, camera_point.y, camera_point.z, 1.0])
        world_homogeneous = self.extrinsic_inv_.data @ camera_homogeneous

        if abs(world_homogeneous[3]) > 1e-10:
            return Point3D(
                world_homogeneous[0] / world_homogeneous[3],
                world_homogeneous[1] / world_homogeneous[3],
                world_homogeneous[2] / world_homogeneous[3]
            )
        else:
            return Point3D(
                world_homogeneous[0],
                world_homogeneous[1],
                world_homogeneous[2]
            )

    def estimate_ground_position(self, pixel_point: Point2D, verbose=False) -> Point3D:
        """估计地面上的位置（假设物体在地面上）"""
        if verbose:
            print("=== 利用地面约束估计位置 ===")

        # 步骤1-3: 得到相机坐标系中的射线方向
        distorted_image_point = self.pixel_to_image(pixel_point)
        ideal_image_point = self.remove_distortion(distorted_image_point)
        camera_ray = self.image_to_camera_ray(ideal_image_point)

        if verbose:
            print(f"像素坐标: ({pixel_point.x}, {pixel_point.y})")
            print(f"相机射线方向: ({camera_ray.x}, {camera_ray.y}, {camera_ray.z})")

        # 提取外参逆矩阵
        M = self.extrinsic_inv_.data

        # 求解lambda使得 P_world.z = 0（地面假设）
        A = M[2, 0] * camera_ray.x + M[2, 1] * camera_ray.y + M[2, 2] * camera_ray.z
        B = M[2, 3]

        if abs(A) < 1e-10:
            print("警告：射线与地面平行，无交点")
            return Point3D(0, 0, 0)

        lambda_val = -B / A

        if verbose:
            print(f"射线参数 lambda: {lambda_val}")

        # 计算交点
        camera_point = Point3D(
            camera_ray.x * lambda_val,
            camera_ray.y * lambda_val,
            camera_ray.z * lambda_val
        )
        world_point = self.camera_to_world(camera_point)

        if verbose:
            print(f"相机坐标系交点: ({camera_point.x}, {camera_point.y}, {camera_point.z})")
            print(f"世界坐标系位置: ({world_point.x}, {world_point.y}, {world_point.z})")

        return world_point

    def estimate_object_height(self, pixel_point: Point2D, estimated_height=1.0, verbose=False) -> Point3D:
        """估计具有一定高度的物体的位置"""
        if verbose:
            print("=== 估计具有一定高度的物体位置 ===")

        # 步骤1-3: 得到相机坐标系中的射线方向
        distorted_image_point = self.pixel_to_image(pixel_point)
        ideal_image_point = self.remove_distortion(distorted_image_point)
        camera_ray = self.image_to_camera_ray(ideal_image_point)

        if verbose:
            print(f"像素坐标: ({pixel_point.x}, {pixel_point.y})")
            print(f"相机射线方向: ({camera_ray.x}, {camera_ray.y}, {camera_ray.z})")

        # 提取外参逆矩阵
        M = self.extrinsic_inv_.data

        # 求解lambda使得 P_world.z = estimated_height
        A = M[2, 0] * camera_ray.x + M[2, 1] * camera_ray.y + M[2, 2] * camera_ray.z
        B = M[2, 3] - estimated_height

        if abs(A) < 1e-10:
            print("警告：射线与水平面平行，无交点")
            return Point3D(0, 0, estimated_height)

        lambda_val = -B / A

        if verbose:
            print(f"射线参数 lambda: {lambda_val}")

        # 计算交点
        camera_point = Point3D(
            camera_ray.x * lambda_val,
            camera_ray.y * lambda_val,
            camera_ray.z * lambda_val
        )
        world_point = self.camera_to_world(camera_point)

        if verbose:
            print(f"相机坐标系交点: ({camera_point.x}, {camera_point.y}, {camera_point.z})")
            print(f"世界坐标系位置: ({world_point.x}, {world_point.y}, {world_point.z})")

        return world_point


class MultiClassFootballDetector:
    def __init__(self, model_path: str, confidence_threshold=0.25, nms_threshold=0.4):
        self.model_path = model_path
        self.confidence_ = confidence_threshold
        self.nms_threshold_ = nms_threshold

        # 自定义类别映射
        self.class_names = {
            0: "Ball",
            1: "Goalpost",
            2: "Person",
            3: "LCross",
            4: "TCross",
            5: "XCross",
            6: "PenaltyPoint",
            7: "Opponent",
            8: "BRMarker"
        }
        self.num_classes = len(self.class_names)

        # 定义不同类别的估计高度（米）
        self.class_heights = {
            0: 0.0,  # Ball - 地面
            1: 2.44,  # Goalpost - 球门高度
            2: 1.7,  # Person - 人的高度
            3: 0.0,  # LCross - 地面标记
            4: 0.0,  # TCross - 地面标记
            5: 0.0,  # XCross - 地面标记
            6: 0.0,  # PenaltyPoint - 地面标记
            7: 1.7,  # Opponent - 人的高度
            8: 0.0  # BRMarker - 地面标记
        }

        print(f"正在加载ONNX模型: {model_path}")

        # 初始化ONNX Runtime会话
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            # 获取输出名称
            self.output_names = [output.name for output in self.session.get_outputs()]
            print(f"模型加载成功! 输入名称: {self.input_name}, 输出名称: {self.output_names}")

            # 打印模型输入输出信息
            for i, input_info in enumerate(self.session.get_inputs()):
                print(f"输入 {i}: {input_info.name}, 形状: {input_info.shape}")
            for i, output_info in enumerate(self.session.get_outputs()):
                print(f"输出 {i}: {output_info.name}, 形状: {output_info.shape}")

        except Exception as e:
            raise RuntimeError(f"加载ONNX模型失败: {e}")

    def inference(self, img: np.ndarray) -> List[DetectionResult]:
        # 预处理图像
        input_img = self.preprocess(img)

        # 运行推理
        print("正在进行推理...")
        outputs = self.session.run(self.output_names, {self.input_name: input_img})

        # 处理输出
        if len(outputs) == 1:
            output_data = outputs[0]
        else:
            output_data = outputs[0]

        print(f"推理完成! 输出形状: {output_data.shape}")

        # 后处理 - 返回所有类别的检测结果
        detections = self.postprocess(output_data, img.shape)
        return detections

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # 调整大小并归一化
        input_size = (640, 640)  # YOLOv8默认输入尺寸
        resized = cv2.resize(img, input_size)
        normalized = resized.astype(np.float32) / 255.0

        # 转换通道顺序: HWC to NCHW
        input_img = np.transpose(normalized, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0)

        return input_img

    def nms(self, boxes: List[Tuple], scores: List[float], class_ids: List[int]) -> List[int]:
        """非极大值抑制算法"""
        if len(boxes) == 0:
            return []

        # 将边界框转换为(x1, y1, x2, y2)格式
        boxes_array = np.array(boxes)
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 0] + boxes_array[:, 2]
        y2 = boxes_array[:, 1] + boxes_array[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(scores)[::-1]  # 按置信度降序排序

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # 计算当前框与剩余框的IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留IoU低于阈值的框
            inds = np.where(iou <= self.nms_threshold_)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, outputs: np.ndarray, orig_shape: Tuple) -> List[DetectionResult]:
        """
        处理YOLOv8输出格式
        输出形状: (1, 13, 8400) - 13 = 4(bbox) + 9(classes)
        """
        orig_h, orig_w = orig_shape[:2]
        input_size = 640

        # YOLOv8输出格式: (1, 13, 8400)
        # 13 = 4(bbox) + 9(classes)
        output_data = outputs  # 形状: (1, 13, 8400)
        print(f"原始输出形状: {output_data.shape}")

        # 转置为更易处理的格式: (8400, 13)
        predictions = output_data.transpose(0, 2, 1).squeeze(0)
        print(f"转换后预测形状: {predictions.shape}")

        all_detections = []
        boxes = []
        scores = []
        class_ids = []

        # 遍历所有检测框 (8400个)
        for i in range(predictions.shape[0]):
            prediction = predictions[i]

            # 前4个值是边界框坐标 (x_center, y_center, width, height)
            bbox = prediction[0:4]

            # 剩余的值是9个类别的概率
            class_scores = prediction[4:4 + self.num_classes]

            # 找到最高类别分数和对应的类别ID
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            # 应用置信度阈值
            if confidence > self.confidence_:
                x_center, y_center, w, h = bbox

                # 坐标转换到原始图像尺寸
                x_center_orig = (x_center / input_size) * orig_w
                y_center_orig = (y_center / input_size) * orig_h
                w_orig = (w / input_size) * orig_w
                h_orig = (h / input_size) * orig_h

                # 计算边界框的左上角坐标
                x1 = int(max(0, x_center_orig - w_orig / 2))
                y1 = int(max(0, y_center_orig - h_orig / 2))
                w_int = int(w_orig)
                h_int = int(h_orig)

                # 计算底部中点坐标 (u, v)
                u = int(x_center_orig)
                v = int(y1 + h_int)

                all_detections.append({
                    'position': (u, v),
                    'confidence': confidence,
                    'bbox': (x1, y1, w_int, h_int),
                    'class_id': class_id
                })

                boxes.append((x1, y1, w_int, h_int))
                scores.append(float(confidence))
                class_ids.append(class_id)

        # 如果没有检测到任何目标，返回空列表
        if len(all_detections) == 0:
            print("未检测到任何目标")
            return []

        print(f"应用NMS前检测到 {len(all_detections)} 个目标")

        # 应用NMS
        keep_indices = self.nms(boxes, scores, class_ids)

        # 构建最终结果
        results = []
        for idx in keep_indices:
            detection = all_detections[idx]

            result = DetectionResult()
            result.position = detection['position']
            result.confidence = detection['confidence']
            result.bbox = detection['bbox']
            result.class_id = detection['class_id']
            result.class_name = self.class_names.get(detection['class_id'], f"Class_{detection['class_id']}")

            results.append(result)

            print(f"检测结果: {result.class_name} 位置{result.position}, 置信度{result.confidence:.3f}")

        print(f"经过NMS后保留 {len(results)} 个检测框")
        return results

    def estimate_world_positions(self, detections: List[DetectionResult],
                                 back_projector: BackProjector) -> List[DetectionResult]:
        """为所有检测结果估计世界坐标位置"""
        for detection in detections:
            pixel_point = Point2D(detection.position[0], detection.position[1])

            # 根据类别选择合适的高度估计方法
            class_height = self.class_heights.get(detection.class_id, 0.0)

            if class_height > 0:
                # 对于有一定高度的物体，使用高度估计
                world_position = back_projector.estimate_object_height(
                    pixel_point, class_height, verbose=False)
            else:
                # 对于地面物体，使用地面约束
                world_position = back_projector.estimate_ground_position(
                    pixel_point, verbose=False)

            detection.world_position = (world_position.x, world_position.y, world_position.z)

            print(
                f"{detection.class_name} 世界坐标: ({world_position.x:.2f}, {world_position.y:.2f}, {world_position.z:.2f})")

        return detections

    def draw_detections_with_3d(self, image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """绘制检测结果，包含3D坐标信息"""
        result_image = image.copy()

        if not detections:
            print("没有检测结果可绘制")
            return result_image

        # 为不同类别定义不同颜色
        colors = {
            0: (0, 255, 0),  # Ball - 绿色
            1: (255, 0, 0),  # Goalpost - 蓝色
            2: (0, 0, 255),  # Person - 红色
            3: (255, 255, 0),  # LCross - 青色
            4: (255, 0, 255),  # TCross - 洋红色
            5: (0, 255, 255),  # XCross - 黄色
            6: (128, 0, 128),  # PenaltyPoint - 紫色
            7: (0, 128, 128),  # Opponent - 橄榄色
            8: (128, 128, 0)  # BRMarker - 深黄色
        }

        for detection in detections:
            # 获取类别对应的颜色
            color = colors.get(detection.class_id, (255, 255, 255))  # 默认白色

            # 绘制检测框
            x, y, w, h = detection.bbox
            cv2.rectangle(result_image,
                          (x, y),
                          (x + w, y + h),
                          color, 2)

            # 绘制底部中点
            u, v = detection.position
            cv2.circle(result_image, (u, v), 5, color, -1)

            # 构建显示文本
            world_x, world_y, world_z = detection.world_position
            display_text = f"{detection.class_name}: {detection.confidence:.2f}"
            position_text = f"World: ({world_x:.1f}, {world_y:.1f}, {world_z:.1f})"

            # 绘制文本
            font_scale = 0.5
            thickness = 1
            text_color = (255, 255, 255)  # 白色文本

            # 第一行文本：类别和置信度
            text_size1 = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_org1 = (x, max(20, y - 10))

            # 绘制文本背景
            cv2.rectangle(result_image,
                          (text_org1[0] - 2, text_org1[1] - text_size1[1] - 2),
                          (text_org1[0] + text_size1[0] + 2, text_org1[1] + 2),
                          color, -1)

            cv2.putText(result_image, display_text, text_org1,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

            # 第二行文本：世界坐标
            text_size2 = cv2.getTextSize(position_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_org2 = (x, text_org1[1] + text_size1[1] + 5)

            # 绘制文本背景
            cv2.rectangle(result_image,
                          (text_org2[0] - 2, text_org2[1] - text_size2[1] - 2),
                          (text_org2[0] + text_size2[0] + 2, text_org2[1] + 2),
                          color, -1)

            cv2.putText(result_image, position_text, text_org2,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

            print(f"绘制: {detection.class_name} 像素({u},{v}) 世界({world_x:.1f},{world_y:.1f},{world_z:.1f})")

        return result_image


# 工具函数
def create_rotation_x(angle_degrees: float) -> TransformMatrix:
    rot = TransformMatrix()
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rot.data[1, 1] = cos_a
    rot.data[1, 2] = sin_a
    rot.data[2, 1] = -sin_a
    rot.data[2, 2] = cos_a

    return rot


def create_rotation_y(angle_degrees: float) -> TransformMatrix:
    rot = TransformMatrix()
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rot.data[0, 0] = cos_a
    rot.data[0, 2] = -sin_a
    rot.data[2, 0] = sin_a
    rot.data[2, 2] = cos_a

    return rot


def create_rotation_z(angle_degrees: float) -> TransformMatrix:
    rot = TransformMatrix()
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rot.data[0, 0] = cos_a
    rot.data[0, 1] = sin_a
    rot.data[1, 0] = -sin_a
    rot.data[1, 1] = cos_a

    return rot


def create_translation(tx: float, ty: float, tz: float) -> TransformMatrix:
    trans = TransformMatrix()
    trans.data[0, 3] = tx
    trans.data[1, 3] = ty
    trans.data[2, 3] = tz
    return trans


def create_extrinsic_matrix(pos_x: float, pos_y: float, pos_z: float,
                            pitch_deg: float, yaw_deg: float, roll_deg: float) -> TransformMatrix:
    rotation_z = create_rotation_z(yaw_deg)
    rotation_y = create_rotation_y(pitch_deg)
    rotation_x = create_rotation_x(roll_deg)

    rotation = rotation_x @ rotation_y @ rotation_z
    translation = create_translation(-pos_x, -pos_y, -pos_z)

    return rotation @ translation


def test_multi_class_detection_with_3d(image_path: str, model_path: str,
                                       confidence_threshold=0.25, nms_threshold=0.4):
    print("=== 多类别目标检测与3D定位测试 ===")
    print(f"置信度阈值: {confidence_threshold}, NMS阈值: {nms_threshold}")

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return

    print(f"图像加载成功! 尺寸: {image.shape[1]} × {image.shape[0]}")

    try:
        # 初始化检测器
        detector = MultiClassFootballDetector(model_path,
                                              confidence_threshold=confidence_threshold,
                                              nms_threshold=nms_threshold)

        # 进行检测
        detections = detector.inference(image)

        if detections:
            print(f"成功检测到 {len(detections)} 个目标!")

            # 设置相机参数
            intrinsics = CameraIntrinsics()
            intrinsics.fx = 645.060547
            intrinsics.fy = 644.257935
            intrinsics.cx = 649.562866
            intrinsics.cy = 373.498932

            # 设置相机外参（位置和姿态）
            camera_x = 6.0  # 相机在世界坐标系中的X位置
            camera_y = -4.5  # 相机在世界坐标系中的Y位置
            camera_z = 1.0  # 相机在世界坐标系中的Z位置（高度）
            yaw_deg = -15.0  # 绕Z轴旋转（偏航角）
            pitch_deg = 0.0  # 绕Y轴旋转（俯仰角）
            roll_deg = -115.0  # 绕X轴旋转（滚转角）

            extrinsic = create_extrinsic_matrix(camera_x, camera_y, camera_z,
                                                pitch_deg, yaw_deg, roll_deg)

            # 创建反投影器
            back_projector = BackProjector(intrinsics, extrinsic)

            # 为所有检测结果估计世界坐标
            detections_with_3d = detector.estimate_world_positions(detections, back_projector)

            # 按类别统计
            class_count = {}
            class_positions = {}

            for detection in detections_with_3d:
                u, v = detection.position
                world_x, world_y, world_z = detection.world_position
                print(
                    f"- {detection.class_name}: 像素({u}, {v}), 世界({world_x:.2f}, {world_y:.2f}, {world_z:.2f}), 置信度{detection.confidence:.3f}")

                # 统计各类别数量
                if detection.class_name in class_count:
                    class_count[detection.class_name] += 1
                else:
                    class_count[detection.class_name] = 1

                # 记录位置信息
                if detection.class_name not in class_positions:
                    class_positions[detection.class_name] = []
                class_positions[detection.class_name].append(detection.world_position)

            print("\n=== 检测结果统计 ===")
            for class_name, count in class_count.items():
                print(f"  {class_name}: {count}个")
                positions = class_positions[class_name]
                for i, pos in enumerate(positions):
                    print(f"    {i + 1}. 世界坐标: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

            # 绘制检测结果（包含世界坐标）
            result_image = detector.draw_detections_with_3d(image, detections_with_3d)

        else:
            print("未检测到任何目标")
            result_image = image.copy()

        # 显示结果
        cv2.imshow("Multi-Class Detection with 3D Positions", result_image)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存结果图像
        output_path = "multi_class_3d_detection_result.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"结果已保存到: {output_path}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 请替换为实际的文件路径
    image_path = "football_image.jpg"
    model_path = "model/best.onnx"

    # 进行多类别检测并计算3D位置
    test_multi_class_detection_with_3d(image_path, model_path,
                                       confidence_threshold=0.25,
                                       nms_threshold=0.4)