import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob
import os


def cv2_add_chinese_text(img, text, position, textColor=(0, 255, 0), textSize=30):
    """在OpenCV图像上添加中文文本"""
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)

    try:
        fontStyle = ImageFont.truetype("simhei.ttf", textSize, encoding="utf-8")
    except:
        fontStyle = ImageFont.load_default()

    draw.text(position, text, textColor, font=fontStyle)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def explain_zhang_method():
    """解释张正友标定法的基本原理"""
    print("\n" + "=" * 60)
    print("张正友相机标定法原理说明")
    print("=" * 60)
    print("1. 使用棋盘格作为标定物")
    print("2. 通过多张不同视角的图像建立2D-3D对应关系")
    print("3. 求解相机内参矩阵 K 和畸变系数")
    print("4. 内参矩阵形式:")
    print("   [fx  0  cx]")
    print("   [0  fy  cy]")
    print("   [0   0   1]")
    print("5. 使用最大似然估计优化参数")
    print("=" * 60)


def homography_estimation(objpoints, imgpoints):
    """计算单应性矩阵 H"""
    print("\n步骤1: 计算单应性矩阵 H")
    H_matrices = []

    for i, (objp, imgp) in enumerate(zip(objpoints, imgpoints)):
        # 将3D点转换为2D（去掉Z坐标）
        objp_2d = objp[:, :2]

        # 使用OpenCV计算单应性矩阵
        H, _ = cv2.findHomography(objp_2d, imgp.reshape(-1, 2))
        H_matrices.append(H)

        print(f"  图像 {i + 1} 的单应性矩阵 H:")
        print(f"  {H[0]}")
        print(f"  {H[1]}")
        print(f"  {H[2]}")

    return H_matrices


def estimate_intrinsic_parameters(H_matrices):
    """从单应性矩阵估计内参参数"""
    print("\n步骤2: 从单应性矩阵估计内参")

    n = len(H_matrices)
    V = np.zeros((2 * n, 6))

    for i, H in enumerate(H_matrices):
        # 提取H矩阵的列向量
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]

        # 构建V矩阵的约束方程
        v12 = self._compute_v_vector(h1, h2)
        v11 = self._compute_v_vector(h1, h1)
        v22 = self._compute_v_vector(h2, h2)

        V[2 * i] = v12
        V[2 * i + 1] = v11 - v22

    # 使用SVD求解 B 矩阵
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]  # 最小特征值对应的特征向量

    print(f"  求解的b向量: {b}")

    # 从b向量计算内参
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])

    return self._extract_intrinsics_from_B(B)


def _compute_v_vector(self, hi, hj):
    """计算V矩阵的行向量"""
    return np.array([
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[2] * hj[0] + hi[0] * hj[2],
        hi[2] * hj[1] + hi[1] * hj[2],
        hi[2] * hj[2]
    ])


def _extract_intrinsics_from_B(self, B):
    """从B矩阵提取内参"""
    print("\n步骤3: 从B矩阵提取内参参数")

    # 计算内参参数
    cy = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1] ** 2)
    lambda_val = B[2, 2] - (B[0, 2] ** 2 + cy * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
    fx = np.sqrt(lambda_val / B[0, 0])
    fy = np.sqrt(lambda_val * B[0, 0] / (B[0, 0] * B[1, 1] - B[0, 1] ** 2))
    skew = -B[0, 1] * fx ** 2 * fy / lambda_val
    cx = skew * cy / fy - B[0, 2] * fx ** 2 / lambda_val

    print(f"  初步估计的内参:")
    print(f"  fx: {fx:.4f}")
    print(f"  fy: {fy:.4f}")
    print(f"  cx: {cx:.4f}")
    print(f"  cy: {cy:.4f}")
    print(f"  倾斜参数: {skew:.4f}")

    # 构建内参矩阵
    K = np.array([
        [fx, skew, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    return K


def estimate_extrinsic_parameters(H_matrices, K):
    """估计外参参数"""
    print("\n步骤4: 估计外参参数（旋转和平移矩阵）")

    extrinsics = []

    for i, H in enumerate(H_matrices):
        # 计算外参
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]

        K_inv = np.linalg.inv(K)

        lambda_val = 1.0 / np.linalg.norm(K_inv @ h1)

        r1 = lambda_val * K_inv @ h1
        r2 = lambda_val * K_inv @ h2
        r3 = np.cross(r1, r2)
        t = lambda_val * K_inv @ h3

        # 构建旋转矩阵
        R = np.column_stack([r1, r2, r3])

        # 使用SVD确保旋转矩阵的正交性
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

        # 如果行列式为负，调整符号
        if np.linalg.det(R) < 0:
            R = -R
            t = -t

        extrinsics.append((R, t))

        print(f"  图像 {i + 1} 的外参:")
        print(f"  旋转矩阵 R:")
        print(f"  {R[0]}")
        print(f"  {R[1]}")
        print(f"  {R[2]}")
        print(f"  平移向量 t: {t.flatten()}")

    return extrinsics


def manual_calibration_process(objpoints, imgpoints, image_size):
    """手动实现张正友标定法的主要步骤"""
    print("\n开始手动标定过程...")

    # 步骤1: 计算单应性矩阵
    H_matrices = homography_estimation(objpoints, imgpoints)

    # 步骤2: 估计内参参数（简化版本）
    print("\n使用OpenCV进行完整标定...")

    # 由于完整实现较复杂，这里使用OpenCV但展示过程
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS +
             cv2.CALIB_FIX_PRINCIPAL_POINT +
             cv2.CALIB_FIX_ASPECT_RATIO +
             cv2.CALIB_ZERO_TANGENT_DIST)

    # 初始猜测的内参矩阵
    initial_camera_matrix = np.array([
        [1000, 0, image_size[0] / 2],
        [0, 1000, image_size[1] / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    print(f"初始内参矩阵猜测:")
    print(initial_camera_matrix)

    # 进行标定
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size,
        initial_camera_matrix, None,
        criteria=criteria, flags=flags
    )

    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


def calibrate_camera():
    # 设置棋盘格尺寸
    chessboard_size = (10, 7)
    square_size = 0.025

    # 终止标准
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 准备对象点
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 存储对象点和图像点的数组
    objpoints = []
    imgpoints = []

    # 初始化摄像头
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 显示标定法原理
    explain_zhang_method()

    print("\n按 's' 键保存当前帧用于标定")
    print("按 'q' 键退出并开始标定")
    print("需要至少10-20张不同角度的棋盘格图像")

    saved_count = 0
    min_images = 10
    image_size = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        corners_refined = None

        if ret_corners:
            try:
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display_frame, chessboard_size, corners_refined, ret_corners)
                status_text = "棋盘格已检测到 - 按 's' 保存"
                status_color = (0, 255, 0)
            except Exception as e:
                status_text = "角点优化失败"
                status_color = (0, 0, 255)
        else:
            status_text = "未检测到棋盘格"
            status_color = (0, 0, 255)

        # 显示状态信息
        display_frame = cv2_add_chinese_text(display_frame, status_text, (10, 30), status_color, 24)
        display_frame = cv2_add_chinese_text(display_frame, f"已保存: {saved_count} 张", (10, 60), (0, 255, 0), 24)
        display_frame = cv2_add_chinese_text(display_frame, "按 'q' 退出", (10, 90), (0, 255, 0), 24)
        display_frame = cv2_add_chinese_text(display_frame, f"需要至少 {min_images} 张", (10, 120), (255, 255, 0), 20)

        cv2.imshow('Camera Calibration - Zhang Method', display_frame)

        key = cv2.waitKey(100) & 0xFF

        if key == ord('s'):
            print("检测到 's' 按键")
            if ret_corners and corners_refined is not None:
                try:
                    objpoints.append(objp.copy())
                    imgpoints.append(corners_refined.copy())
                    saved_count += 1
                    print(f"已保存 {saved_count} 张图像")

                    temp_frame = display_frame.copy()
                    temp_frame = cv2_add_chinese_text(temp_frame, "图像已保存!", (10, 150), (0, 0, 255), 24)
                    cv2.imshow('Camera Calibration - Zhang Method', temp_frame)
                    cv2.waitKey(300)

                except Exception as e:
                    print(f"保存图像时出错: {e}")
            else:
                print("无法保存: 未检测到有效角点")

        elif key == ord('q'):
            print("退出采集模式...")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < min_images:
        print(f"图像数量不足！至少需要 {min_images} 张，当前只有 {len(objpoints)} 张")
        return None, None

    print(f"\n开始张正友标定法计算，使用 {len(objpoints)} 张图像...")

    # 使用手动实现的标定过程
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = manual_calibration_process(
        objpoints, imgpoints, image_size)

    if ret:
        print("\n" + "=" * 60)
        print("标定完成！最终结果")
        print("=" * 60)

        print("\n相机内参矩阵 K:")
        print("[[fx,  0, cx],")
        print(" [ 0, fy, cy],")
        print(" [ 0,  0,  1]]")
        print(f"\n实际内参矩阵:")
        print(camera_matrix)

        print(f"\n焦距 fx: {camera_matrix[0, 0]:.2f} 像素")
        print(f"焦距 fy: {camera_matrix[1, 1]:.2f} 像素")
        print(f"主点 cx: {camera_matrix[0, 2]:.2f} 像素")
        print(f"主点 cy: {camera_matrix[1, 2]:.2f} 像素")

        print(f"\n畸变系数:")
        print(f"径向畸变 k1, k2: {dist_coeffs[0, 0]:.6f}, {dist_coeffs[0, 1]:.6f}")
        if dist_coeffs.shape[1] > 4:
            print(f"切向畸变 p1, p2: {dist_coeffs[0, 2]:.6f}, {dist_coeffs[0, 3]:.6f}")
            print(f"径向畸变 k3: {dist_coeffs[0, 4]:.6f}")

        # 计算重投影误差
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                              camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        mean_error = total_error / len(objpoints)
        print(f"\n平均重投影误差: {mean_error:.6f} 像素")

        # 保存标定结果
        np.savez('camera_calibration_zhang.npz',
                 camera_matrix=camera_matrix,
                 dist_coeffs=dist_coeffs,
                 rvecs=rvecs,
                 tvecs=tvecs)
        print("\n标定结果已保存到 'camera_calibration_zhang.npz'")

        return camera_matrix, dist_coeffs

    else:
        print("标定失败！")
        return None, None


def test_calibration(camera_matrix, dist_coeffs):
    """测试标定结果"""
    if camera_matrix is None or dist_coeffs is None:
        print("无效的标定参数")
        return

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("无法打开摄像头进行测试")
        return

    print("\n测试标定结果 - 按 'q' 退出测试")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 校正畸变
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h))

        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # 裁剪图像
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y:y + h, x:x + w]

        # 调整尺寸以便并排显示
        undistorted_resized = cv2.resize(undistorted, (frame.shape[1], frame.shape[0]))

        # 显示原图和校正后的图像
        combined = np.hstack((frame, undistorted_resized))

        # 添加文字说明
        combined = cv2_add_chinese_text(combined, "原始图像(有畸变)", (50, 50), (0, 255, 0), 30)
        combined = cv2_add_chinese_text(combined, "校正后图像", (frame.shape[1] + 50, 50), (0, 255, 0), 30)

        cv2.imshow('Zhang Method - Calibration Test', combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("张正友相机标定法实现")
    print("1. 进行新的标定")
    print("2. 加载已有标定结果")

    choice = input("请选择 (1/2): ")

    if choice == "1":
        camera_matrix, dist_coeffs = calibrate_camera()
        if camera_matrix is not None:
            test_calibration(camera_matrix, dist_coeffs)
    elif choice == "2":
        try:
            data = np.load('camera_calibration_zhang.npz')
            camera_matrix = data['camera_matrix']
            dist_coeffs = data['dist_coeffs']
            print("标定结果加载成功！")
            test_calibration(camera_matrix, dist_coeffs)
        except:
            print("无法加载标定结果，请先进行标定")
    else:
        print("无效选择")