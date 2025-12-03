import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional, Dict, Any
import math
import yaml
import time
from dataclasses import dataclass
import json
import os


@dataclass
class DetectionResult:
    position: Tuple[int, int] = (0, 0)  # (u, v) - 检测框底部中点坐标（已校正）
    confidence: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, width, height)
    class_id: int = 0
    class_name: str = ""
    world_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 3D世界坐标 (x, y, z)


@dataclass
class FieldMarker:
    type: str  # 'L', 'T', 'X', 'P'
    x: float  # 相对于机器人的x坐标
    y: float  # 相对于机器人的y坐标
    confidence: float = 0.0


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass
class PoseBox2D:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    thetamin: float
    thetamax: float


@dataclass
class LocateResult:
    success: bool
    code: int
    residual: float
    pose: Pose2D
    msecs: float


@dataclass
class FieldDimensions:
    length: float = 14.0
    width: float = 9.0
    penaltyDist: float = 2.1
    goalWidth: float = 2.6
    circleRadius: float = 1.5
    penaltyAreaLength: float = 3.0
    penaltyAreaWidth: float = 6.0
    goalAreaLength: float = 1.0
    goalAreaWidth: float = 4.0


@dataclass
class CameraParameters:
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 640.0
    cy: float = 360.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    camera_height: float = 0.5  # 相机离地高度（米）
    pitch_angle: float = 0.0  # 相机俯仰角（弧度）


class CameraModel:
    """相机模型，包含内参和畸变校正"""

    def __init__(self, camera_params: CameraParameters):
        self.params = camera_params

        # 构建相机矩阵
        self.camera_matrix = np.array([
            [camera_params.fx, 0, camera_params.cx],
            [0, camera_params.fy, camera_params.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # 构建畸变系数
        self.dist_coeffs = np.array([
            camera_params.k1, camera_params.k2, camera_params.p1,
            camera_params.p2, camera_params.k3
        ], dtype=np.float32)

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """校正图像点的畸变"""
        if len(points.shape) == 1:
            points = points.reshape(1, -1)

        points_2d = points.astype(np.float32)
        undistorted_points = cv2.undistortPoints(
            points_2d, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix
        )
        return undistorted_points.reshape(-1, 2)

    def pixel_to_camera(self, pixel_points: np.ndarray, Z: float = 1.0) -> np.ndarray:
        """将像素坐标转换到相机坐标系（逆投影）"""
        undistorted_points = self.undistort_points(pixel_points)

        camera_points = []
        for point in undistorted_points:
            u, v = point
            X = (u - self.params.cx) * Z / self.params.fx
            Y = (v - self.params.cy) * Z / self.params.fy
            camera_points.append([X, Y, Z])

        return np.array(camera_points)

    def camera_to_pixel(self, camera_points: np.ndarray) -> np.ndarray:
        """将相机坐标系转换到像素坐标"""
        pixel_points = []
        for point in camera_points:
            X, Y, Z = point
            u = (X * self.params.fx / Z) + self.params.cx
            v = (Y * self.params.fy / Z) + self.params.cy
            pixel_points.append([u, v])

        pixel_points = np.array(pixel_points, dtype=np.float32)

        # 应用畸变
        if np.any(self.dist_coeffs != 0):
            points_3d = np.zeros((len(pixel_points), 1, 2), dtype=np.float32)
            points_3d[:, 0, :] = pixel_points
            distorted_points, _ = cv2.projectPoints(
                points_3d, np.zeros(3), np.zeros(3),
                self.camera_matrix, self.dist_coeffs
            )
            return distorted_points.reshape(-1, 2)
        else:
            return pixel_points

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """校正整张图像的畸变"""
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)


class SimpleFootballDetector:
    def __init__(self, model_path: str, camera_model: CameraModel, confidence_threshold=0.25, nms_threshold=0.4):
        self.model_path = model_path
        self.confidence_ = confidence_threshold
        self.nms_threshold_ = nms_threshold
        self.camera_model = camera_model

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

        print(f"正在加载ONNX模型: {model_path}")

        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            print(f"模型加载成功! 输入名称: {self.input_name}, 输出名称: {self.output_names}")

        except Exception as e:
            raise RuntimeError(f"加载ONNX模型失败: {e}")

    def inference(self, img: np.ndarray) -> List[DetectionResult]:
        # 先进行图像畸变校正
        undistorted_img = self.camera_model.undistort_image(img)
        input_img = self.preprocess(undistorted_img)

        print("正在进行推理...")
        outputs = self.session.run(self.output_names, {self.input_name: input_img})

        if len(outputs) == 1:
            output_data = outputs[0]
        else:
            output_data = outputs[0]

        print(f"推理完成! 输出形状: {output_data.shape}")

        detections = self.postprocess(output_data, undistorted_img.shape)
        return detections

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        input_size = (640, 640)
        resized = cv2.resize(img, input_size)
        normalized = resized.astype(np.float32) / 255.0
        input_img = np.transpose(normalized, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0)
        return input_img

    def nms(self, boxes: List[Tuple], scores: List[float], class_ids: List[int]) -> List[int]:
        if len(boxes) == 0:
            return []

        boxes_array = np.array(boxes)
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 0] + boxes_array[:, 2]
        y2 = boxes_array[:, 1] + boxes_array[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(scores)[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= self.nms_threshold_)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, outputs: np.ndarray, orig_shape: Tuple) -> List[DetectionResult]:
        orig_h, orig_w = orig_shape[:2]
        input_size = 640

        output_data = outputs
        predictions = output_data.transpose(0, 2, 1).squeeze(0)

        all_detections = []
        boxes = []
        scores = []
        class_ids = []

        for i in range(predictions.shape[0]):
            prediction = predictions[i]
            bbox = prediction[0:4]
            class_scores = prediction[4:4 + self.num_classes]

            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence > self.confidence_:
                x_center, y_center, w, h = bbox

                x_center_orig = (x_center / input_size) * orig_w
                y_center_orig = (y_center / input_size) * orig_h
                w_orig = (w / input_size) * orig_w
                h_orig = (h / input_size) * orig_h

                x1 = int(max(0, x_center_orig - w_orig / 2))
                y1 = int(max(0, y_center_orig - h_orig / 2))
                w_int = int(w_orig)
                h_int = int(h_orig)

                u = int(x_center_orig)
                v = int(y1 + h_int)

                # 校正检测点的畸变
                pixel_point = np.array([u, v])
                undistorted_point = self.camera_model.undistort_points(pixel_point)[0]
                u_corrected, v_corrected = undistorted_point.astype(int)

                all_detections.append({
                    'position': (u_corrected, v_corrected),
                    'confidence': confidence,
                    'bbox': (x1, y1, w_int, h_int),
                    'class_id': class_id
                })

                boxes.append((x1, y1, w_int, h_int))
                scores.append(float(confidence))
                class_ids.append(class_id)

        if len(all_detections) == 0:
            print("未检测到任何目标")
            return []

        print(f"应用NMS前检测到 {len(all_detections)} 个目标")

        keep_indices = self.nms(boxes, scores, class_ids)

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

        print(f"经过NMS后保留 {len(results)} 个检测框")
        return results

    def draw_detections(self, image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        result_image = image.copy()

        if not detections:
            return result_image

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
            color = colors.get(detection.class_id, (255, 255, 255))

            x, y, w, h = detection.bbox
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)

            u, v = detection.position
            cv2.circle(result_image, (u, v), 5, color, -1)

            display_text = f"{detection.class_name}: {detection.confidence:.2f}"
            font_scale = 0.6
            thickness = 2

            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_bg_x1 = x
            text_bg_y1 = max(0, y - text_size[1] - 10)
            text_bg_x2 = x + text_size[0] + 5
            text_bg_y2 = max(0, y - 5)

            cv2.rectangle(result_image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
            text_org = (x + 2, max(15, y - 5))
            cv2.putText(result_image, display_text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                        thickness)

        return result_image


class PoseEstimator:
    """基于相机几何的位姿估计器"""

    def __init__(self, camera_model: CameraModel):
        self.camera_model = camera_model

    def estimate_field_marker_pose(self, detection: DetectionResult, robot_pose: Pose2D) -> FieldMarker:
        """估计场地标记点在机器人坐标系中的位置"""
        u, v = detection.position

        # 假设标记点在地面上 (Z=0)
        # 使用逆透视变换计算3D位置
        camera_height = self.camera_model.params.camera_height
        pitch_angle = self.camera_model.params.pitch_angle

        # 将像素坐标转换到相机坐标系
        pixel_point = np.array([u, v])
        camera_point = self.camera_model.pixel_to_camera(pixel_point, Z=1.0)[0]
        X_cam, Y_cam, Z_cam = camera_point

        # 考虑相机俯仰角
        # 计算射线与地面的交点
        # 地面方程: Y = -camera_height (在相机坐标系中)
        if Y_cam + math.sin(pitch_angle) > 0:  # 确保射线指向地面
            # 计算射线参数 t
            t = (self.camera_model.params.camera_height) / (Y_cam + math.sin(pitch_angle))

            # 计算地面交点
            X_ground = X_cam * t
            Z_ground = Z_cam * t - math.cos(pitch_angle) * t

            # 转换到机器人坐标系（相机在机器人前方）
            x_robot = Z_ground  # 相机前方为机器人x轴
            y_robot = -X_ground  # 相机左侧为机器人y轴正方向

            marker_type = self.class_to_marker_type(detection.class_name)

            print(f"位姿估计: {detection.class_name} -> 像素({u},{v}) -> 机器人坐标({x_robot:.2f},{y_robot:.2f})")

            return FieldMarker(
                type=marker_type,
                x=x_robot,
                y=y_robot,
                confidence=detection.confidence
            )
        else:
            # 如果射线不指向地面，使用简化模型
            print(f"警告: 使用简化模型估计 {detection.class_name}")
            return self.simple_pose_estimation(detection)

    def simple_pose_estimation(self, detection: DetectionResult) -> FieldMarker:
        """简化的位姿估计（备用方法）"""
        u, v = detection.position
        bbox_area = detection.bbox[2] * detection.bbox[3]

        # 基于检测框大小的距离估计
        estimated_distance = max(1.0, 5000.0 / bbox_area)

        # 基于像素位置的方位估计
        u_center = u - self.camera_model.params.cx
        horizontal_angle = u_center / self.camera_model.params.fx

        x_robot = estimated_distance * math.cos(horizontal_angle)
        y_robot = estimated_distance * math.sin(horizontal_angle)

        marker_type = self.class_to_marker_type(detection.class_name)

        return FieldMarker(
            type=marker_type,
            x=x_robot,
            y=y_robot,
            confidence=detection.confidence
        )

    def class_to_marker_type(self, class_name: str) -> str:
        """将检测类别转换为标记点类型"""
        if "LCross" in class_name:
            return 'L'
        elif "TCross" in class_name:
            return 'T'
        elif "XCross" in class_name:
            return 'X'
        elif class_name == "PenaltyPoint":
            return 'P'
        elif class_name == "BRMarker":
            return 'B'  # 特殊标记
        else:
            return 'U'  # Unknown


class Locator:
    """机器人定位器（粒子滤波实现）"""

    def __init__(self):
        self.fieldMarkers = []
        self.fieldDimensions = FieldDimensions()
        self.minMarkerCnt = 2
        self.residualTolerance = 0.4
        self.convergeTolerance = 0.1
        self.maxIteration = 50
        self.muOffset = 1.0

    def init(self, fd: FieldDimensions, minMarkerCnt=2, residualTolerance=0.4, muOffset=1.0):
        self.fieldDimensions = fd
        self.calc_field_markers(fd)
        self.minMarkerCnt = minMarkerCnt
        self.residualTolerance = residualTolerance
        self.muOffset = muOffset
        print(f"定位器初始化完成，共有 {len(self.fieldMarkers)} 个场地标记点")

    def calc_field_markers(self, fd: FieldDimensions):
        """计算场地标记点的全局坐标"""
        self.fieldMarkers = []

        # 中线上的 X 标志
        self.fieldMarkers.append(FieldMarker('X', 0.0, -fd.circleRadius))
        self.fieldMarkers.append(FieldMarker('X', 0.0, fd.circleRadius))

        # 罚球点
        self.fieldMarkers.append(FieldMarker('P', fd.length / 2 - fd.penaltyDist, 0.0))
        self.fieldMarkers.append(FieldMarker('P', -fd.length / 2 + fd.penaltyDist, 0.0))

        # 边线中心
        self.fieldMarkers.append(FieldMarker('T', 0.0, fd.width / 2))
        self.fieldMarkers.append(FieldMarker('T', 0.0, -fd.width / 2))

        # 禁区标记点
        self.fieldMarkers.append(FieldMarker('L', (fd.length / 2 - fd.penaltyAreaLength), fd.penaltyAreaWidth / 2))
        self.fieldMarkers.append(FieldMarker('L', (fd.length / 2 - fd.penaltyAreaLength), -fd.penaltyAreaWidth / 2))
        self.fieldMarkers.append(FieldMarker('L', -(fd.length / 2 - fd.penaltyAreaLength), fd.penaltyAreaWidth / 2))
        self.fieldMarkers.append(FieldMarker('L', -(fd.length / 2 - fd.penaltyAreaLength), -fd.penaltyAreaWidth / 2))

        # 场地四角
        self.fieldMarkers.append(FieldMarker('L', fd.length / 2, fd.width / 2))
        self.fieldMarkers.append(FieldMarker('L', fd.length / 2, -fd.width / 2))
        self.fieldMarkers.append(FieldMarker('L', -fd.length / 2, fd.width / 2))
        self.fieldMarkers.append(FieldMarker('L', -fd.length / 2, -fd.width / 2))

    def marker_to_field_frame(self, marker_r: FieldMarker, pose_r2f: Pose2D) -> FieldMarker:
        """将机器人坐标系下的标记点转换到场地坐标系"""
        x, y, theta = pose_r2f.x, pose_r2f.y, pose_r2f.theta

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        x_field = x + marker_r.x * cos_theta - marker_r.y * sin_theta
        y_field = y + marker_r.x * sin_theta + marker_r.y * cos_theta

        return FieldMarker(marker_r.type, x_field, y_field, marker_r.confidence)

    def min_dist(self, marker: FieldMarker) -> float:
        """找到最近的同类型场地标记点的距离"""
        min_dist = float('inf')

        for target in self.fieldMarkers:
            if target.type != marker.type:
                continue

            dist = math.sqrt((target.x - marker.x) ** 2 + (target.y - marker.y) ** 2)
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def residual(self, markers_r: List[FieldMarker], pose: Pose2D) -> float:
        """计算残差 - 衡量观测与假设位姿的匹配程度"""
        total_residual = 0.0

        for marker_r in markers_r:
            marker_f = self.marker_to_field_frame(marker_r, pose)
            marker_dist = self.min_dist(marker_f)
            obs_dist = max(math.sqrt(marker_r.x ** 2 + marker_r.y ** 2), 0.1)
            total_residual += marker_dist / obs_dist * 3.0

        return total_residual

    def locate_robot(self, markers_r: List[FieldMarker], constraints: PoseBox2D,
                     num_particles=1000, max_iterations=50) -> LocateResult:
        """基于粒子滤波的机器人定位"""
        start_time = time.time()

        print(f"=== 开始定位 ===")
        print(f"检测到 {len(markers_r)} 个标记点")
        for marker in markers_r:
            print(f"  {marker.type}: ({marker.x:.2f}, {marker.y:.2f}) 置信度: {marker.confidence:.2f}")

        if len(markers_r) < self.minMarkerCnt:
            return LocateResult(False, 1, float('inf'), Pose2D(0, 0, 0),
                                (time.time() - start_time) * 1000)

        particles = self.initialize_particles(num_particles, constraints)
        best_pose = Pose2D(0, 0, 0)
        best_residual = float('inf')

        for iteration in range(max_iterations):
            weights = []
            residuals = []

            for particle in particles:
                res = self.residual(markers_r, particle)
                residuals.append(res)

                if res < best_residual:
                    best_residual = res
                    best_pose = particle

            min_res = min(residuals)
            weights = [math.exp(-(res - min_res)) for res in residuals]
            weight_sum = sum(weights)
            if weight_sum > 0:
                weights = [w / weight_sum for w in weights]
            else:
                weights = [1.0 / len(weights)] * len(weights)

            if self.is_converged(particles):
                avg_residual = best_residual / len(markers_r)
                if avg_residual <= self.residualTolerance:
                    processing_time = (time.time() - start_time) * 1000
                    print(f"定位成功! 残差: {avg_residual:.4f}, 迭代: {iteration}")
                    return LocateResult(True, 0, avg_residual, best_pose, processing_time)

            particles = self.resample_particles(particles, weights, constraints)

        avg_residual = best_residual / len(markers_r)
        processing_time = (time.time() - start_time) * 1000
        print(f"定位失败! 最佳残差: {avg_residual:.4f}")
        return LocateResult(False, 2, avg_residual, best_pose, processing_time)

    def initialize_particles(self, num_particles: int, constraints: PoseBox2D) -> List[Pose2D]:
        """初始化粒子群"""
        particles = []
        for _ in range(num_particles):
            x = np.random.uniform(constraints.xmin, constraints.xmax)
            y = np.random.uniform(constraints.ymin, constraints.ymax)
            theta = np.random.uniform(constraints.thetamin, constraints.thetamax)
            particles.append(Pose2D(x, y, theta))
        return particles

    def resample_particles(self, particles: List[Pose2D], weights: List[float],
                           constraints: PoseBox2D) -> List[Pose2D]:
        """重采样粒子"""
        new_particles = []
        indices = np.random.choice(len(particles), size=len(particles), p=weights)

        for idx in indices:
            old_particle = particles[idx]
            noise_x = np.random.normal(0, 0.1)
            noise_y = np.random.normal(0, 0.1)
            noise_theta = np.random.normal(0, 0.05)

            x = max(constraints.xmin, min(constraints.xmax, old_particle.x + noise_x))
            y = max(constraints.ymin, min(constraints.ymax, old_particle.y + noise_y))
            theta = max(constraints.thetamin, min(constraints.thetamax, old_particle.theta + noise_theta))

            new_particles.append(Pose2D(x, y, theta))

        return new_particles

    def is_converged(self, particles: List[Pose2D]) -> bool:
        """检查粒子是否收敛"""
        if len(particles) == 0:
            return False

        xs = [p.x for p in particles]
        ys = [p.y for p in particles]
        thetas = [p.theta for p in particles]

        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        theta_range = max(thetas) - min(thetas)

        return (x_range < self.convergeTolerance and
                y_range < self.convergeTolerance and
                theta_range < self.convergeTolerance * 2)


class RobotVisionSystem:
    """完整的机器人视觉定位系统"""

    def __init__(self, model_path: str, camera_params: CameraParameters):
        self.camera_model = CameraModel(camera_params)
        self.detector = SimpleFootballDetector(model_path, self.camera_model)
        self.pose_estimator = PoseEstimator(self.camera_model)
        self.locator = Locator()

        field_dim = FieldDimensions()
        self.locator.init(field_dim)

        self.robot_pose = Pose2D(0, 0, 0)

    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """处理单张图像并计算机器人位置"""
        print("=== 开始处理图像 ===")

        # 1. 目标检测
        detections = self.detector.inference(image)
        print(f"检测到 {len(detections)} 个目标")

        # 2. 提取和估计场地标记点
        field_markers = self.extract_field_markers(detections)
        print(f"提取到 {len(field_markers)} 个场地标记点")

        # 3. 机器人定位
        constraints = PoseBox2D(-7, 7, -4.5, 4.5, -math.pi, math.pi)
        localization_result = self.locator.locate_robot(field_markers, constraints)

        # 4. 更新机器人位姿
        if localization_result.success:
            self.robot_pose = localization_result.pose
            print(f"机器人定位成功: 位置({self.robot_pose.x:.2f}, {self.robot_pose.y:.2f}), "
                  f"朝向{math.degrees(self.robot_pose.theta):.1f}°")
        else:
            print("机器人定位失败")

        # 5. 绘制结果
        result_image = self.detector.draw_detections(image, detections)
        result_image = self.draw_robot_position(result_image)

        return {
            'detections': detections,
            'field_markers': field_markers,
            'localization_result': localization_result,
            'result_image': result_image
        }

    def extract_field_markers(self, detections: List[DetectionResult]) -> List[FieldMarker]:
        """从检测结果中提取场地标记点"""
        field_markers = []

        for detection in detections:
            if self.is_field_marker(detection.class_name):
                field_marker = self.pose_estimator.estimate_field_marker_pose(detection, self.robot_pose)
                field_markers.append(field_marker)

        return field_markers

    def is_field_marker(self, class_name: str) -> bool:
        """判断是否为场地标记点"""
        return class_name in ["LCross", "TCross", "XCross", "PenaltyPoint", "BRMarker"]

    def draw_robot_position(self, image: np.ndarray) -> np.ndarray:
        """在图像上绘制机器人位置信息"""
        result_image = image.copy()

        pose_text = f"Robot: ({self.robot_pose.x:.2f}, {self.robot_pose.y:.2f}, {math.degrees(self.robot_pose.theta):.1f}°)"
        cv2.putText(result_image, pose_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return result_image


def main():
    """主函数"""
    model_path = "model/best.onnx"
    image_path = "football_image.jpg"

    # 使用您提供的相机参数
    camera_params = CameraParameters(
        fx=500.0, fy=500.0, cx=640.0, cy=360.0,
        k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
        camera_height=0.5,  # 相机离地高度0.5米
        pitch_angle=math.radians(-10)  # 相机俯仰角-10度（略微向下）
    )

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return

    print(f"图像加载成功: {image.shape[1]}x{image.shape[0]}")
    print(f"相机参数: fx={camera_params.fx}, fy={camera_params.fy}, "
          f"cx={camera_params.cx}, cy={camera_params.cy}")

    # 初始化视觉系统
    vision_system = RobotVisionSystem(model_path, camera_params)

    # 处理图像
    results = vision_system.process_image(image)

    # 显示结果
    cv2.imshow("Robot Vision System", results['result_image'])
    print("按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    output_path = "robot_localization_result.jpg"
    cv2.imwrite(output_path, results['result_image'])
    print(f"结果已保存到: {output_path}")

    # 打印详细结果
    print("\n=== 最终结果 ===")
    if results['localization_result'].success:
        pose = results['localization_result'].pose
        print(f"✅ 定位成功!")
        print(f"   位置: ({pose.x:.3f}, {pose.y:.3f})")
        print(f"   朝向: {math.degrees(pose.theta):.2f}°")
        print(f"   残差: {results['localization_result'].residual:.4f}")
        print(f"   耗时: {results['localization_result'].msecs:.1f}ms")
    else:
        print(f"❌ 定位失败")
        print(f"   错误代码: {results['localization_result'].code}")
        print(f"   最佳残差: {results['localization_result'].residual:.4f}")


if __name__ == "__main__":
    main()