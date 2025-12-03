import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional, Dict
import math
import random
import copy


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


class ParticleFilterCameraEstimator:
    def __init__(self, num_particles=50):
        self.num_particles = num_particles
        self.particles = []
        self.best_particle = None
        self.back_projector = None
        self.weights = np.ones(num_particles) / num_particles

        # 已知的固定外参
        self.camera_z = 1.0
        self.pitch_deg = 0.0
        self.roll_deg = -115.0

        # 场地参数
        self.length = 14.0
        self.width = 9.0
        self.penaltyDist = 2.1
        self.circleRadius = 1.5
        self.penaltyAreaLength = 3.0
        self.penaltyAreaWidth = 6.0
        self.goalAreaLength = 1.0
        self.goalAreaWidth = 4.0

        # 过程噪声 - 只对x,y,yaw添加噪声
        self.position_noise = 0.1
        self.yaw_noise = 2.0  # 偏航角噪声

        # 初始粒子中心
        self.initial_center_x = 5.0
        self.initial_center_y = -3
        self.initial_center_yaw = -45.0

        # 高斯采样的标准差
        self.init_std_x = 2
        self.init_std_y = 2
        self.init_std_yaw = 10.0

        # 初始化世界坐标点
        self.world_points_dict = self.calculate_world_points()
        print("场地标记点世界坐标:")
        for name, point in self.world_points_dict.items():
            print(f"  {name}: ({point.x:.2f}, {point.y:.2f}, {point.z:.2f})")

        # 初始化粒子
        self.initialize_particles()

    def calculate_world_points(self):
        """计算所有场地标记点的世界坐标"""
        return {
            # 中线上的 X 标志
            'X1': Point3D(0.0, -self.circleRadius, 0.0),
            'X2': Point3D(0.0, self.circleRadius, 0.0),

            # 罚球点
            'P1': Point3D(self.length / 2 - self.penaltyDist, 0.0, 0.0),
            'P2': Point3D(-self.length / 2 + self.penaltyDist, 0.0, 0.0),

            # 边线中心
            'T1': Point3D(0.0, self.width / 2, 0.0),
            'T2': Point3D(0.0, -self.width / 2, 0.0),

            # 禁区标记点
            'L1': Point3D(self.length / 2 - self.penaltyAreaLength, self.penaltyAreaWidth / 2, 0.0),
            'L2': Point3D(self.length / 2 - self.penaltyAreaLength, -self.penaltyAreaWidth / 2, 0.0),
            'L3': Point3D(-self.length / 2 + self.penaltyAreaLength, self.penaltyAreaWidth / 2, 0.0),
            'L4': Point3D(-self.length / 2 + self.penaltyAreaLength, -self.penaltyAreaWidth / 2, 0.0),
            'T3': Point3D(self.length / 2, self.penaltyAreaWidth / 2, 0.0),
            'T4': Point3D(self.length / 2, -self.penaltyAreaWidth / 2, 0.0),
            'T5': Point3D(-self.length / 2, self.penaltyAreaWidth / 2, 0.0),
            'T6': Point3D(-self.length / 2, -self.penaltyAreaWidth / 2, 0.0),

            # 球门区标记点
            'L5': Point3D(self.length / 2 - self.goalAreaLength, self.goalAreaWidth / 2, 0.0),
            'L6': Point3D(self.length / 2 - self.goalAreaLength, -self.goalAreaWidth / 2, 0.0),
            'L7': Point3D(-self.length / 2 + self.goalAreaLength, self.goalAreaWidth / 2, 0.0),
            'L8': Point3D(-self.length / 2 + self.goalAreaLength, -self.goalAreaWidth / 2, 0.0),
            'T7': Point3D(self.length / 2, self.goalAreaWidth / 2, 0.0),
            'T8': Point3D(self.length / 2, -self.goalAreaWidth / 2, 0.0),
            'T9': Point3D(-self.length / 2, self.goalAreaWidth / 2, 0.0),
            'T10': Point3D(-self.length / 2, -self.goalAreaWidth / 2, 0.0),

            # 场地四角
            'L9': Point3D(self.length / 2, self.width / 2, 0.0),
            'L10': Point3D(self.length / 2, -self.width / 2, 0.0),
            'L11': Point3D(-self.length / 2, self.width / 2, 0.0),
            'L12': Point3D(-self.length / 2, -self.width / 2, 0.0)
        }

    def initialize_particles(self):
        """初始化粒子群"""
        self.particles = []
        for i in range(self.num_particles):
            if i == 0:
                x = self.initial_center_x
                y = self.initial_center_y
                yaw = self.initial_center_yaw
            else:
                # 在初始中心周围进行高斯采样
                x = random.gauss(self.initial_center_x, self.init_std_x)
                y = random.gauss(self.initial_center_y, self.init_std_y)
                yaw = random.gauss(self.initial_center_yaw, self.init_std_yaw)

            particle = {
                'x': x,
                'y': y,
                'yaw': yaw,
                'weight': 1.0 / self.num_particles
            }
            self.particles.append(particle)

        print(f"初始化了 {len(self.particles)} 个粒子")
        print(f"初始粒子范围: x[{min(p['x'] for p in self.particles):.2f}, {max(p['x'] for p in self.particles):.2f}], "
              f"y[{min(p['y'] for p in self.particles):.2f}, {max(p['y'] for p in self.particles):.2f}], "
              f"yaw[{min(p['yaw'] for p in self.particles):.2f}, {max(p['yaw'] for p in self.particles):.2f}]")

    def create_back_projector_for_particle(self, particle, intrinsics):
        """为指定粒子创建反投影器"""
        extrinsic = create_extrinsic_matrix(
            particle['x'], particle['y'], self.camera_z,
            self.pitch_deg, particle['yaw'], self.roll_deg
        )
        return BackProjector(intrinsics, extrinsic)

    def calculate_reprojection_error(self, detection, world_point, back_projector):
        """计算重投影误差 - 将像素坐标投影到世界坐标，再与世界坐标点求误差"""
        try:
            # 获取检测的像素坐标
            u_detect, v_detect = detection.position
            pixel_point = Point2D(u_detect, v_detect)

            # 根据检测类别选择合适的高度估计方法
            class_height = 0.0  # 默认地面高度

            # 如果是球门柱等有高度的物体，使用相应的高度
            if detection.class_name == "Goalpost":
                class_height = 2.44  # 球门高度

            # 将像素坐标投影到世界坐标
            if class_height > 0:
                # 对于有一定高度的物体，使用高度估计
                estimated_world_point = back_projector.estimate_object_height(
                    pixel_point, class_height, verbose=False)
            else:
                # 对于地面物体，使用地面约束
                estimated_world_point = back_projector.estimate_ground_position(
                    pixel_point, verbose=False)

            # 计算估计的世界坐标与真实世界坐标之间的误差
            # 只考虑x和y坐标的误差，忽略z坐标（因为高度可能不同）
            error_xy = math.sqrt(
                (estimated_world_point.x - world_point.x) ** 2 +
                (estimated_world_point.y - world_point.y) ** 2
            )

            # 可选：也可以考虑3D距离误差
            # error_3d = math.sqrt(
            #     (estimated_world_point.x - world_point.x) ** 2 +
            #     (estimated_world_point.y - world_point.y) ** 2 +
            #     (estimated_world_point.z - world_point.z) ** 2
            # )

            return error_xy

        except Exception as e:
            print(f"重投影误差计算失败: {e}")
            return float('inf')

    def get_world_points_for_detection(self, detection):
        """根据检测结果获取所有对应的世界坐标点"""
        class_name = detection.class_name

        # 根据类别名称映射到世界坐标点
        if class_name == "XCross":
            return [self.world_points_dict['X1'], self.world_points_dict['X2']]
        elif class_name == "PenaltyPoint":
            return [self.world_points_dict['P1'], self.world_points_dict['P2']]
        elif class_name == "TCross":
            # T型标记点有多种可能
            t_points = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']
            return [self.world_points_dict[point] for point in t_points]
        elif class_name == "LCross":
            # L型标记点
            l_points = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12']
            return [self.world_points_dict[point] for point in l_points]
        else:
            # 对于其他类别，返回空列表（不用于定位）
            return []

    def calculate_min_reprojection_error(self, detection, world_points, back_projector):
        """计算检测结果与所有可能的世界坐标点之间的最小重投影误差"""
        if not world_points:
            return float('inf')

        min_error = float('inf')

        for world_point in world_points:
            error = self.calculate_reprojection_error(detection, world_point, back_projector)
            if error < min_error:
                min_error = error

        return min_error

    def update_weights(self, detections, intrinsics):
        """根据检测结果更新粒子权重"""
        total_weight = 0.0

        for i, particle in enumerate(self.particles):
            back_projector = self.create_back_projector_for_particle(particle, intrinsics)
            total_error = 0.0
            valid_detections = 0

            for detection in detections:
                # 根据检测类别获取所有对应的世界坐标点
                world_points = self.get_world_points_for_detection(detection)
                if world_points:
                    # 计算最小重投影误差
                    min_error = self.calculate_min_reprojection_error(detection, world_points, back_projector)
                    if min_error < float('inf'):
                        total_error += min_error
                        valid_detections += 1

            # 计算平均误差
            if valid_detections > 0:
                avg_error = total_error / valid_detections
                # 使用高斯权重函数，误差越小权重越大
                particle_weight = math.exp(-avg_error / 20)  # 调整分母可以改变权重分布的宽度
            else:
                particle_weight = 1e-10  # 极小权重避免除零

            # print(f"粒子 {i}: 平均误差 = {avg_error if valid_detections > 0 else 'N/A'}, 权重 = {particle_weight}")

            particle['weight'] = particle_weight
            total_weight += particle_weight

        # 归一化权重
        if total_weight > 0:
            for particle in self.particles:
                particle['weight'] /= total_weight
        else:
            # 如果所有权重都为0，重新初始化权重
            for particle in self.particles:
                particle['weight'] = 1.0 / self.num_particles

        # 更新权重数组
        self.weights = np.array([p['weight'] for p in self.particles])

        # 打印权重统计信息
        max_weight = max(p['weight'] for p in self.particles)
        min_weight = min(p['weight'] for p in self.particles)
        avg_weight = sum(p['weight'] for p in self.particles) / len(self.particles)
        print(f"权重统计: 最大={max_weight:.6f}, 最小={min_weight:.6f}, 平均={avg_weight:.6f}")

    def resample(self):
        """重采样步骤"""
        # 系统重采样
        indices = self.systematic_resample()
        new_particles = [copy.deepcopy(self.particles[i]) for i in indices]
        self.particles = new_particles

        # 重置权重
        for particle in self.particles:
            particle['weight'] = 1.0 / self.num_particles
        self.weights = np.ones(self.num_particles) / self.num_particles

    def systematic_resample(self):
        """系统重采样"""
        N = self.num_particles
        positions = (np.arange(N) + random.random()) / N
        indices = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        return indices

    def predict(self):
        """预测步骤 - 添加过程噪声"""
        for particle in self.particles:
            # 添加位置噪声
            particle['x'] += random.gauss(0, self.position_noise)
            particle['y'] += random.gauss(0, self.position_noise)
            # 添加偏航角噪声
            particle['yaw'] += random.gauss(0, self.yaw_noise)

    def get_best_estimate(self):
        """获取最佳估计 - 使用权重最大的粒子"""
        if not self.particles:
            return None

        # 找到权重最大的粒子
        best_particle = max(self.particles, key=lambda p: p['weight'])

        self.best_particle = {
            'x': best_particle['x'],
            'y': best_particle['y'],
            'yaw': best_particle['yaw'],
            'weight': best_particle['weight']
        }

        print(f"最佳粒子: x={self.best_particle['x']:.3f}, y={self.best_particle['y']:.3f}, "
              f"yaw={self.best_particle['yaw']:.3f}, weight={self.best_particle['weight']:.6f}")

        return self.best_particle

    def run_filter(self, detections, intrinsics, num_iterations=15):
        """运行粒子滤波"""
        print(f"开始粒子滤波，迭代次数: {num_iterations}")

        for iteration in range(num_iterations):
            print(f"\n--- 迭代 {iteration + 1}/{num_iterations} ---")

            # 预测步骤
            self.predict()

            # 更新步骤
            self.update_weights(detections, intrinsics)

            # 获取当前最佳估计
            best_est = self.get_best_estimate()
            print(f"当前最佳估计: x={best_est['x']:.3f}, y={best_est['y']:.3f}, yaw={best_est['yaw']:.3f}")

            # 计算有效粒子数
            effective_particles = 1.0 / np.sum(self.weights ** 2)
            print(f"有效粒子数: {effective_particles:.1f}/{self.num_particles}")

            # 重采样
            if effective_particles < self.num_particles / 2:
                print("执行重采样")
                self.resample()

        final_estimate = self.get_best_estimate()
        print(f"\n=== 最终估计结果 ===")
        print(f"相机位置: x={final_estimate['x']:.3f}, y={final_estimate['y']:.3f}")
        print(f"偏航角: yaw={final_estimate['yaw']:.3f}°")

        return final_estimate


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


def test_particle_filter_camera_estimation(image_path: str, model_path: str,
                                           confidence_threshold=0.25, nms_threshold=0.4):
    print("=== 基于粒子滤波的相机外参估计 ===")

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

        if not detections:
            print("未检测到任何可用于定位的目标")
            return

        print(f"检测到 {len(detections)} 个目标")

        # 设置相机内参
        intrinsics = CameraIntrinsics()
        intrinsics.fx = 645.060547
        intrinsics.fy = 644.257935
        intrinsics.cx = 649.562866
        intrinsics.cy = 373.498932

        # 初始化粒子滤波器
        particle_filter = ParticleFilterCameraEstimator(num_particles=50)

        # 运行粒子滤波
        camera_estimate = particle_filter.run_filter(detections, intrinsics, num_iterations=15)

        if camera_estimate:
            # 使用估计的外参创建反投影器
            extrinsic = create_extrinsic_matrix(
                camera_estimate['x'], camera_estimate['y'], particle_filter.camera_z,
                particle_filter.pitch_deg, camera_estimate['yaw'], particle_filter.roll_deg
            )

            back_projector = BackProjector(intrinsics, extrinsic)

            # 为所有检测结果估计世界坐标
            detections_with_3d = detector.estimate_world_positions(detections, back_projector)

            # 绘制检测结果
            result_image = detector.draw_detections_with_3d(image, detections_with_3d)

            # 在图像上显示估计的相机参数
            text_lines = [
                f"Camera: x={camera_estimate['x']:.2f}, y={camera_estimate['y']:.2f}",
                f"Yaw: {camera_estimate['yaw']:.1f}deg",
                f"Particles: {particle_filter.num_particles}"
            ]

            for i, text in enumerate(text_lines):
                cv2.putText(result_image, text, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 显示结果
            cv2.imshow("Particle Filter Camera Estimation", result_image)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 保存结果图像
            output_path = "particle_filter_camera_estimation_result.jpg"
            cv2.imwrite(output_path, result_image)
            print(f"结果已保存到: {output_path}")

    except Exception as e:
        print(f"粒子滤波测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 请替换为实际的文件路径
    image_path = "football_image.jpg"
    model_path = "model/best.onnx"

    # 运行粒子滤波相机外参估计
    test_particle_filter_camera_estimation(image_path, model_path,
                                           confidence_threshold=0.25,
                                           nms_threshold=0.4)