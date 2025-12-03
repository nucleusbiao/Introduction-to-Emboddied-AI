import cv2
import yaml
import numpy as np
import os
import random
import time
import math
import onnxruntime as ort
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Callable


# ===================== 补充依赖类（适配 MultiClassFootballDetector）=====================
@dataclass
class Point2D:
    x: float
    y: float


@dataclass
class Point3D:
    x: float
    y: float
    z: float


@dataclass
class DetectionResult:
    """适配 MultiClassFootballDetector 的检测结果类"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, w, h
    position: Tuple[int, int]  # 底部中点 (u, v)
    world_position: Optional[Tuple[float, float, float]] = None  # 世界坐标


class BackProjector:
    """反投影工具类（将像素坐标转换为世界坐标）"""

    def __init__(self, intrinsics: 'Intrinsics', camera_pose: 'Pose'):
        self.intrinsics = intrinsics
        self.camera_pose = camera_pose  # 相机在世界坐标系中的位姿（eye2base）

        # 相机内参矩阵
        self.K = np.array([
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # 相机外参（旋转矩阵和平移向量）
        self.R = camera_pose.rotation
        self.T = camera_pose.translation.reshape(3, 1)

    def estimate_ground_position(self, pixel_point: Point2D, verbose: bool = False) -> Point3D:
        """
        地面点反投影（假设 z=0）
        :param pixel_point: 像素坐标 (u, v)
        :return: 世界坐标 (x, y, 0)
        """
        # 像素坐标转归一化相机坐标（齐次）
        u, v = pixel_point.x, pixel_point.y
        uv_hom = np.array([u, v, 1], dtype=np.float32).reshape(3, 1)
        cam_coord_hom = np.linalg.inv(self.K) @ uv_hom

        # 相机坐标转世界坐标（假设地面 z=0）
        # 相机坐标: [Xc, Yc, Zc]^T = k * [cam_coord_hom[0], cam_coord_hom[1], cam_coord_hom[2]]
        # 世界坐标: [Xw, Yw, 0]^T = R^T * ([Xc, Yc, Zc]^T - T)
        # 求解 k 使 Zw=0
        R_inv = self.R.T
        T = self.T

        # 展开方程求解 k
        A = R_inv[2, 0] * cam_coord_hom[0] + R_inv[2, 1] * cam_coord_hom[1] + R_inv[2, 2] * cam_coord_hom[2]
        B = R_inv[2, 0] * T[0] + R_inv[2, 1] * T[1] + R_inv[2, 2] * T[2]
        k = B  # 因 Zw=0，k = B/A（A=cam_coord_hom[2]，此处简化为 k=B）

        # 计算世界坐标
        cam_coord = k * cam_coord_hom
        world_coord = R_inv @ (cam_coord - T)

        result = Point3D(world_coord[0, 0], world_coord[1, 0], 0.0)
        if verbose:
            print(f"像素 ({u}, {v}) -> 世界坐标 ({result.x:.2f}, {result.y:.2f}, {result.z:.2f})")
        return result

    def estimate_object_height(self, pixel_point: Point2D, object_height: float, verbose: bool = False) -> Point3D:
        """
        有高度物体的反投影（已知物体高度）
        :param pixel_point: 像素坐标 (u, v)（物体底部中点）
        :param object_height: 物体高度（米）
        :return: 世界坐标（物体底部中心）
        """
        # 简化实现：先反投影到底面，再根据高度调整
        ground_point = self.estimate_ground_position(pixel_point, verbose)
        # 假设物体垂直于地面，z坐标为物体高度（根据实际需求调整）
        return Point3D(ground_point.x, ground_point.y, object_height)


# ===================== 原有数据结构（保持不变）=====================
@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    model: int  # 畸变模型


@dataclass
class Pose:
    rotation: np.ndarray  # 3x3 旋转矩阵
    translation: np.ndarray  # 3x1 平移向量

    def __init__(self, rx=0, ry=0, rz=0, tx=0, ty=0, tz=0):
        self.rotation = self.euler_to_rot(rx, ry, rz)
        self.translation = np.array([tx, ty, tz], dtype=np.float32)

    @staticmethod
    def euler_to_rot(rx, ry, rz):
        """欧拉角转旋转矩阵（ZYX顺序）"""
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(rx), -math.sin(rx)],
                       [0, math.sin(rx), math.cos(rx)]])
        Ry = np.array([[math.cos(ry), 0, math.sin(ry)],
                       [0, 1, 0],
                       [-math.sin(ry), 0, math.cos(ry)]])
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                       [math.sin(rz), math.cos(rz), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx

    def to_cv_mat(self) -> np.ndarray:
        """转换为4x4变换矩阵"""
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.translation
        return mat

    def __mul__(self, other: 'Pose') -> 'Pose':
        """Pose乘法（变换复合）"""
        new_rot = self.rotation @ other.rotation
        new_trans = self.rotation @ other.translation + self.translation
        result = Pose()
        result.rotation = new_rot
        result.translation = new_trans
        return result


@dataclass
class FieldMarker:
    type: str
    x: float
    y: float
    confidence: float


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
class Pose2D:
    x: float
    y: float
    theta: float  # 弧度


@dataclass
class PoseBox2D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    theta_min: float
    theta_max: float


@dataclass
class LocateResult:
    success: bool
    code: int
    residual: float
    msecs: float
    pose: Pose2D


# ===================== 工具类（替换原有 YoloV8Detector 为 MultiClassFootballDetector）=====================
class ImageLoader:
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        img = cv2.imread(path)
        return img if img is not None else np.array([])

    @staticmethod
    def image_exists(path: str) -> bool:
        return os.path.exists(path)


class MultiClassFootballDetector:
    """完整集成的多类别足球场景检测器（替换原有 YoloV8Detector）"""

    def __init__(self, config: Dict):
        # 从配置中读取检测器参数
        self.model_path = config.get("model_path", "")
        self.confidence_threshold = config.get("confidence_threshold", 0.25)
        self.nms_threshold = config.get("nms_threshold", 0.4)

        # 自定义类别映射（与模型输出对应）
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

        print(f"正在加载ONNX模型: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 初始化ONNX Runtime会话
        try:
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
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
        """执行检测，返回所有类别的检测结果"""
        # 预处理图像
        input_img = self.preprocess(img)

        # 运行推理
        print("正在进行推理...")
        outputs = self.session.run(self.output_names, {self.input_name: input_img})

        # 处理输出
        output_data = outputs[0] if len(outputs) > 0 else np.array([])
        print(f"推理完成! 输出形状: {output_data.shape}")

        # 后处理 - 返回所有类别的检测结果
        detections = self.postprocess(output_data, img.shape)
        return detections

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """图像预处理（适配YOLOv8输入格式）"""
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
            inds = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, outputs: np.ndarray, orig_shape: Tuple) -> List[DetectionResult]:
        """处理YOLOv8输出格式，转换为DetectionResult列表"""
        orig_h, orig_w = orig_shape[:2]
        input_size = 640

        if outputs.size == 0:
            print("推理输出为空")
            return []

        # YOLOv8输出格式: (1, 13, 8400) -> 13 = 4(bbox) + 9(classes)
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
            if confidence > self.confidence_threshold:
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
            class_id = detection['class_id']

            result = DetectionResult(
                class_id=class_id,
                class_name=self.class_names.get(class_id, f"Class_{class_id}"),
                confidence=detection['confidence'],
                bbox=detection['bbox'],
                position=detection['position']
            )

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
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)

            # 绘制底部中点
            u, v = detection.position
            cv2.circle(result_image, (u, v), 5, color, -1)

            # 构建显示文本
            if detection.world_position:
                world_x, world_y, world_z = detection.world_position
                display_text = f"{detection.class_name}: {detection.confidence:.2f}"
                position_text = f"({world_x:.1f}, {world_y:.1f}, {world_z:.1f})"
            else:
                display_text = f"{detection.class_name}: {detection.confidence:.2f}"
                position_text = "No 3D Pos"

            # 绘制文本（带背景框）
            font_scale = 0.5
            thickness = 1
            text_color = (255, 255, 255)  # 白色文本

            # 第一行：类别+置信度
            text1_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text1_org = (x, max(20, y - 10))
            cv2.rectangle(result_image,
                          (text1_org[0] - 2, text1_org[1] - text1_size[1] - 2),
                          (text1_org[0] + text1_size[0] + 2, text1_org[1] + 2),
                          color, -1)
            cv2.putText(result_image, display_text, text1_org,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

            # 第二行：世界坐标
            text2_size = cv2.getTextSize(position_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text2_org = (x, text1_org[1] + text1_size[1] + 5)
            cv2.rectangle(result_image,
                          (text2_org[0] - 2, text2_org[1] - text2_size[1] - 2),
                          (text2_org[0] + text2_size[0] + 2, text2_org[1] + 2),
                          color, -1)
            cv2.putText(result_image, position_text, text2_org,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

        return result_image


# ===================== 位姿估计器和定位器（保持不变，适配新检测结果）=====================
class PoseEstimator:
    def __init__(self, intrinsics: Intrinsics):
        self.intrinsics = intrinsics

    def init(self, config: Dict):
        pass

    def estimate_by_color(self, eye2base: Pose, detection: DetectionResult, image: np.ndarray) -> Pose:
        """适配 DetectionResult 的位姿估计"""
        # 直接使用检测器计算的世界坐标
        if detection.world_position:
            pose = Pose()
            pose.translation = np.array(detection.world_position, dtype=np.float32)
            pose.rotation = np.eye(3, dtype=np.float32)  # 简化：默认无旋转
            return pose
        return Pose()


class BallPoseEstimator(PoseEstimator):
    pass


class HumanLikePoseEstimator(PoseEstimator):
    pass


class FieldMarkerPoseEstimator(PoseEstimator):
    pass


class Locator:
    """Python 版本的粒子滤波定位器（完全还原 C++ 逻辑）"""

    def __init__(self):
        # 初始化内部参数（对应 C++ 类的私有成员）
        self.field_length: float = 14.0
        self.field_width: float = 9.0
        self.particle_count: int = 1000
        self.max_residual: float = 0.4
        self.conv_threshold: float = 0.05
        self.use_cache: bool = False
        self.cache_file: str = ""

    def init(self, field_dim: FieldDimensions, particle_count: int,
             max_residual: float, convergence_threshold: float,
             use_cache: bool, cache_file: str):
        """
        初始化定位器（对应 C++ 的 init 方法）
        :param field_dim: 场地尺寸配置
        :param particle_count: 粒子数量
        :param max_residual: 最大允许残差
        :param convergence_threshold: 收敛阈值
        :param use_cache: 是否使用缓存
        :param cache_file: 缓存文件路径
        """
        self.field_length = field_dim.length
        self.field_width = field_dim.width
        self.particle_count = particle_count
        self.max_residual = max_residual
        self.conv_threshold = convergence_threshold
        self.use_cache = use_cache
        self.cache_file = cache_file

        # 缓存功能（如果需要实现，可在此处添加）
        if self.use_cache and self.cache_file:
            print(f"Warning: Cache feature not implemented in Python version")

    def locate_robot(self, markers: List[FieldMarker], constraints: PoseBox2D,
                     max_iter: int, max_residual: float, conv_threshold: float,
                     noise_level: float) -> LocateResult:
        """
        核心定位方法（完全还原 C++ 粒子滤波逻辑）
        :param markers: 检测到的场地标记列表
        :param constraints: 位姿约束范围
        :param max_iter: 最大迭代次数
        :param max_residual: 最大允许残差（覆盖初始化值）
        :param conv_threshold: 收敛阈值（覆盖初始化值）
        :param noise_level: 粒子噪声水平
        :return: 定位结果
        """
        # 记录运行时间（对应 C++ 的 chrono）
        start_time = time.time()

        # 初始化结果（对应 C++ 的 LocateResult）
        result = LocateResult(
            success=False,
            code=1,  # 1: 初始状态
            residual=float('inf'),
            msecs=0.0,
            pose=Pose2D(0.0, 0.0, 0.0)
        )

        # 检查标记点数量（至少 2 个标记点才能定位）
        if not markers or len(markers) < 2:
            result.code = 2  # 2: 标记点不足
            result.msecs = (time.time() - start_time) * 1000
            return result

        # ===================== 粒子滤波核心逻辑（完全还原 C++）=====================
        # 1. 初始化粒子群（在约束范围内随机生成）
        particles = []
        for _ in range(self.particle_count):
            # 随机生成粒子位姿（添加噪声）
            x = random.uniform(constraints.x_min, constraints.x_max) + random.uniform(0, noise_level)
            y = random.uniform(constraints.y_min, constraints.y_max) + random.uniform(0, noise_level)
            theta = random.uniform(constraints.theta_min, constraints.theta_max) + random.uniform(0, noise_level)
            particles.append(Pose2D(x, y, theta))

        current_residual = float('inf')

        # 2. 迭代更新粒子群
        for iter_idx in range(max_iter):
            # 计算每个粒子的权重（基于标记点残差）
            weights = []
            total_weight = 0.0

            for p in particles:
                residual = 0.0
                # 计算粒子与所有标记点的加权残差（置信度反向加权）
                for marker in markers:
                    dx = p.x - marker.x
                    dy = p.y - marker.y
                    # 残差 = 欧氏距离 * (1 - 置信度)（置信度越低，权重越小）
                    residual += np.hypot(dx, dy) * (1.0 - marker.confidence)

                # 权重 = 指数函数（残差越小，权重越大）
                weight = np.exp(-residual / max_residual)
                weights.append(weight)
                total_weight += weight

            # 归一化权重（避免权重和不为 1）
            weights = [w / total_weight for w in weights]

            # 3. 重采样（选择权重高的粒子，保留多样性）
            new_particles = []
            # 根据权重随机选择粒子索引（对应 C++ 的 discrete_distribution）
            particle_indices = random.choices(range(self.particle_count), weights=weights, k=self.particle_count)

            for idx in particle_indices:
                # 复制高权重粒子，并添加少量噪声
                old_p = particles[idx]
                new_x = old_p.x + random.uniform(0, noise_level)
                new_y = old_p.y + random.uniform(0, noise_level)
                new_theta = old_p.theta + random.uniform(0, noise_level)
                new_particles.append(Pose2D(new_x, new_y, new_theta))

            # 更新粒子群
            particles = new_particles

            # 4. 计算当前迭代的平均残差（判断收敛）
            current_residual = 0.0
            for p in particles:
                res = 0.0
                for marker in markers:
                    dx = p.x - marker.x
                    dy = p.y - marker.y
                    res += np.hypot(dx, dy)
                current_residual += res / len(markers)  # 每个粒子的平均残差
            current_residual /= self.particle_count  # 粒子群的平均残差

            # 5. 收敛判断（残差小于阈值则提前退出）
            if current_residual < conv_threshold:
                break

        # 6. 计算最终位姿（粒子群加权平均）
        final_x = 0.0
        final_y = 0.0
        final_theta = 0.0
        total_weight = 0.0

        for p in particles:
            # 权重基于当前残差计算
            weight = np.exp(-current_residual / max_residual)
            final_x += p.x * weight
            final_y += p.y * weight
            final_theta += p.theta * weight
            total_weight += weight

        # 归一化得到最终位姿
        final_x /= total_weight
        final_y /= total_weight
        final_theta /= total_weight

        # 7. 填充定位结果
        result.success = (current_residual < max_residual)
        result.code = 0 if result.success else 3  # 0: 成功，3: 未收敛
        result.residual = current_residual
        result.pose = Pose2D(final_x, final_y, final_theta)
        result.msecs = (time.time() - start_time) * 1000  # 转换为毫秒

        return result

# ===================== 主类（融合新检测器，适配检测流程）=====================
class StandaloneVision:
    def __init__(self):
        self.detector: Optional[MultiClassFootballDetector] = None
        self.pose_estimator_map: Dict[str, PoseEstimator] = {}
        self.intr: Optional[Intrinsics] = None
        self.p_eye2head = Pose()
        self.p_headprime2head = Pose()
        self.classnames: List[str] = []  # 由检测器的class_names自动填充
        self.enable_post_process = False
        self.single_ball_assumption = False
        self.confidence_map: Dict[str, float] = {}
        self.detected_markers: List[FieldMarker] = []

    def init(self, config_path: str) -> bool:
        if not os.path.exists(config_path):
            print(f"Config file not found: {config_path}")
            return False

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"Loaded config from: {config_path}")

        # 初始化相机内参
        cam_config = config.get("camera", {})
        intr_config = cam_config.get("intrin", {})
        self.intr = Intrinsics(
            fx=intr_config.get("fx", 1000.0),  # 默认值，实际从配置读取
            fy=intr_config.get("fy", 1000.0),
            cx=intr_config.get("cx", 320.0),
            cy=intr_config.get("cy", 240.0),
            model=intr_config.get("model", 0)
        )
        print(f"DEBUG - Camera intrinsics: fx={self.intr.fx}, fy={self.intr.fy}, cx={self.intr.cx}, cy={self.intr.cy}")

        # 初始化外参
        extrin = cam_config.get("extrin", [])
        if extrin:
            extrin_mat = np.array(extrin, dtype=np.float32).reshape(4, 4)
            self.p_eye2head.rotation = extrin_mat[:3, :3]
            self.p_eye2head.translation = extrin_mat[:3, 3]

        # 补偿参数
        pitch_comp = cam_config.get("pitch_compensation", 0.0)
        yaw_comp = cam_config.get("yaw_compensation", 0.0)
        z_comp = cam_config.get("z_compensation", 0.0)
        self.p_headprime2head = Pose(
            ry=pitch_comp * math.pi / 180,
            rz=yaw_comp * math.pi / 180,
            tz=z_comp
        )

        # 初始化检测器（替换为 MultiClassFootballDetector）
        det_config = config.get("detection_model", {})
        try:
            self.detector = MultiClassFootballDetector(det_config)
            # 从检测器获取类别名称列表
            self.classnames = list(self.detector.class_names.values())
        except Exception as e:
            print(f"Failed to create detector: {e}")
            return False

        # 后处理配置
        default_threshold = det_config.get("confidence_threshold", 0.25)
        post_process = det_config.get("post_process", {})
        if post_process:
            self.enable_post_process = True
            self.single_ball_assumption = post_process.get("single_ball_assumption", False)
            conf_thresholds = post_process.get("confidence_thresholds", {})
            for cls, thr in conf_thresholds.items():
                self.confidence_map[cls] = thr
            # 设置默认置信度
            for cls in self.classnames:
                if cls not in self.confidence_map:
                    self.confidence_map[cls] = default_threshold

        # 初始化位姿估计器
        self.pose_estimator_map["default"] = PoseEstimator(self.intr)
        self.pose_estimator_map["default"].init({})

        if "ball_pose_estimator" in config:
            self.pose_estimator_map["ball"] = BallPoseEstimator(self.intr)
            self.pose_estimator_map["ball"].init(config["ball_pose_estimator"])

        if "human_like_pose_estimator" in config:
            self.pose_estimator_map["human_like"] = HumanLikePoseEstimator(self.intr)
            self.pose_estimator_map["human_like"].init(config["human_like_pose_estimator"])

        if "field_marker_pose_estimator" in config:
            self.pose_estimator_map["field_marker"] = FieldMarkerPoseEstimator(self.intr)
            self.pose_estimator_map["field_marker"].init(config["field_marker_pose_estimator"])

        print("Standalone vision system initialized successfully")
        return True

    def process_image(self, image_path: str, pose_path: str = ""):
        self.detected_markers.clear()

        # 加载图像
        image = ImageLoader.load_image(image_path)
        if image.size == 0:
            print(f"Failed to load image: {image_path}")
            return

        print(f"DEBUG - Image size: {image.shape[1]}x{image.shape[0]}")
        print(f"DEBUG - Principal point (cx, cy): ({self.intr.cx}, {self.intr.cy})")

        # 加载机器人位姿
        p_head2base = Pose()
        if pose_path and os.path.exists(pose_path):
            with open(pose_path, 'r') as f:
                pose_config = yaml.safe_load(f)
            if "robot_pose" in pose_config:
                pose_data = pose_config["robot_pose"]
                # 支持4x4矩阵和欧拉角两种格式
                if isinstance(pose_data, list) and len(pose_data) == 4 and all(
                        isinstance(row, list) and len(row) == 4 for row in pose_data):
                    extrin_mat = np.array(pose_data, dtype=np.float32)
                    p_head2base.rotation = extrin_mat[:3, :3]
                    p_head2base.translation = extrin_mat[:3, 3]
                    print(f"Loaded robot pose from 4x4 matrix: {pose_path}")
                else:
                    p_head2base = Pose(
                        rx=pose_data.get("rx", 0),
                        ry=pose_data.get("ry", 0),
                        rz=pose_data.get("rz", 0),
                        tx=pose_data.get("tx", 0),
                        ty=pose_data.get("ty", 0),
                        tz=pose_data.get("tz", 0)
                    )
                    print(f"Loaded robot pose from Euler angles: {pose_path}")
            else:
                print(f"No robot_pose found in {pose_path}, using default pose")
        else:
            print("Using default robot pose (origin)")

        # 计算相机到基座的变换（eye2base）
        p_eye2base = p_head2base * self.p_headprime2head * self.p_eye2head

        # ===================== 核心修改：使用新检测器执行检测 =====================
        # 1. 执行检测（获取所有类别结果）
        detections = self.detector.inference(image)
        print(f"Found {len(detections)} raw detections")

        # 2. 后处理过滤（基于置信度和单球假设）
        filtered_detections = self.post_process_detections(detections)
        print(f"After filtering: {len(filtered_detections)} detections")

        # 3. 估计世界坐标（使用 BackProjector）
        back_projector = BackProjector(self.intr, p_eye2base)
        detections_with_3d = self.detector.estimate_world_positions(filtered_detections, back_projector)

        # 4. 处理场地标记（用于机器人定位）
        self.process_field_markers(detections_with_3d)

        # 5. 保存结果和定位
        self.save_marker_positions()
        self.calculate_robot_position()

        # 6. 绘制并显示结果（带3D坐标）
        display_image = self.detector.draw_detections_with_3d(image, detections_with_3d)
        cv2.imshow("Detection & 3D Results", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def post_process_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """后处理检测结果（置信度过滤+单球假设）"""
        if not self.enable_post_process or not detections:
            return detections

        # 1. 置信度过滤
        filtered = []
        for det in detections:
            if det.confidence >= self.confidence_map.get(det.class_name, 0.25):
                filtered.append(det)

        # 2. 单球假设（只保留置信度最高的球）
        if self.single_ball_assumption:
            ball_detections = []
            other_detections = []
            for det in filtered:
                if det.class_name == "Ball":
                    ball_detections.append(det)
                else:
                    other_detections.append(det)

            if len(ball_detections) > 1:
                print("Multiple ball detections found, keeping the one with highest confidence")
                ball_detections.sort(key=lambda x: x.confidence, reverse=True)
                other_detections.append(ball_detections[0])
            else:
                other_detections.extend(ball_detections)
            filtered = other_detections

        return filtered

    def process_field_markers(self, detections: List[DetectionResult]):
        """从检测结果中提取场地标记"""
        for det in detections:
            if self.is_field_marker(det.class_name) and det.world_position:
                marker = FieldMarker(
                    type=self.class_to_marker_type(det.class_name),
                    x=det.world_position[0],
                    y=det.world_position[1],
                    confidence=det.confidence
                )
                self.detected_markers.append(marker)
                print(f"=== FIELD MARKER ===")
                print(f"Marker: {det.class_name} Type: {marker.type}")
                print(f"World Position (x, y): ({marker.x:.2f}, {marker.y:.2f})")
                print(f"Confidence: {marker.confidence:.3f}")
                print(f"====================")

    def class_to_marker_type(self, class_name: str) -> str:
        if "LCross" in class_name:
            return "L"
        elif "TCross" in class_name:
            return "T"
        elif "XCross" in class_name:
            return "X"
        elif class_name == "PenaltyPoint":
            return "P"
        elif class_name == "BRMarker":
            return "B"
        else:
            return "U"

    def is_field_marker(self, class_name: str) -> bool:
        """判断是否为场地标记（用于定位）"""
        return class_name in ["LCross", "TCross", "XCross", "PenaltyPoint", "BRMarker"]

    def save_marker_positions(self):
        timestamp = int(time.time() * 1000)
        os.makedirs("../data", exist_ok=True)
        filename = f"../data/marker_positions_{timestamp}.yaml"

        marker_data = {
            "timestamp": timestamp,
            "markers": []
        }
        for marker in self.detected_markers:
            marker_data["markers"].append({
                "type": marker.type,
                "x": marker.x,
                "y": marker.y,
                "confidence": marker.confidence
            })

        with open(filename, 'w') as f:
            yaml.dump(marker_data, f)

        # 保存最新版本
        with open("../data/marker_positions_latest.yaml", 'w') as f:
            yaml.dump(marker_data, f)

        print(f"Saved {len(self.detected_markers)} markers to: {filename}")

    def calculate_robot_position(self):
        print(f"DEBUG - calculateRobotPosition called")
        print(f"DEBUG - Number of markers detected: {len(self.detected_markers)}")

        if not self.detected_markers:
            print("No field markers detected, cannot calculate robot position.")
            return

        # 初始化定位器
        locator = Locator()
        field_dim = FieldDimensions()
        locator.init(field_dim, particle_count=1000, max_residual=0.4,
                     convergence_threshold=0.05, use_cache=False, cache_file="")

        # 设置约束范围
        constraints = PoseBox2D(-10.0, 10.0, -5.0, 5.0, -math.pi, math.pi)

        # 调用定位算法
        result = locator.locate_robot(
            self.detected_markers,
            constraints,
            max_iter=10000,
            max_residual=1.5,
            conv_threshold=0.01,
            noise_level=0.2
        )

        # 输出定位结果
        print("=== DETAILED LOCALIZATION ANALYSIS ===")
        print(f"Success: {result.success}")
        print(f"Error Code: {result.code}")
        print(f"Residual: {result.residual:.3f}")

        if result.success or result.residual < 0.1:
            self.save_localization_result(result)
            print(f"Robot Position: ({result.pose.x:.2f}, {result.pose.y:.2f})")
            print(f"Robot Orientation: {result.pose.theta * 180 / math.pi:.2f} degrees")
        else:
            print("=== ROBOT LOCALIZATION FAILED ===")

    def save_localization_result(self, result: LocateResult):
        loc_data = {
            "timestamp": int(time.time() * 1000),
            "success": result.success,
            "code": result.code,
            "residual": result.residual,
            "processing_time_ms": result.msecs
        }
        if result.success:
            loc_data["robot_pose"] = {
                "x": result.pose.x,
                "y": result.pose.y,
                "theta": result.pose.theta
            }

        with open("../data/localization_result.yaml", 'w') as f:
            yaml.dump(loc_data, f)
        print("Localization result saved to: ../data/localization_result.yaml")


# ===================== 主函数（保持不变）=====================
def main():
    config_path = "model/vision.yaml"
    image_path = "model/color9.jpg"
    pose_path = "model/pose9.yaml"

    # 命令行参数处理
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    if len(sys.argv) > 2:
        image_path = sys.argv[2]
    if len(sys.argv) > 3:
        pose_path = sys.argv[3]

    vision = StandaloneVision()
    if not vision.init(config_path):
        print("Failed to initialize vision system")
        return -1

    if not ImageLoader.image_exists(image_path):
        print(f"Image not found: {image_path}")
        return -1

    vision.process_image(image_path, pose_path)
    return 0


if __name__ == "__main__":
    main()