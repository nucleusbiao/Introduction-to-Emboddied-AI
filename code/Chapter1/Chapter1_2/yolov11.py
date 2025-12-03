import os
import subprocess
import sys
import time
from pathlib import Path
import yaml

def create_dataset_yaml():
    """创建dataset.yaml文件内容"""
    yaml_content = """# YOLOv11数据集配置

#path: /home/qd/SPLObjDetectDatasetV2/yolo_dataset

#train: images/train
#val: images/val

train: /home/qd/SPLObjDetectDatasetV2/yolo_dataset/images/train
val: /home/qd/SPLObjDetectDatasetV2/yolo_dataset/images/val
test: /home/qd/SPLObjDetectDatasetV2/test

# 类别数量
nc: 4

# 类别名称（根据数据中的类别ID）
names: 
  0: ball
  1: robot
  2: goal_post
  3: penalty_spot
"""
    
    # 写入dataset.yaml文件
    with open("dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    print("✅ dataset.yaml 文件已创建")

def run_command(command, description, timeout=None):
    """运行命令行命令并处理输出"""
    print(f"\n{'='*60}")
    print(f"🚀 开始: {description}")
    print(f"📝 命令: {command}")
    print(f"{'='*60}")
    
    try:
        # 实时输出命令执行过程
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        start_time = time.time()
        
        # 实时打印输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
            
            # 检查超时
            if timeout and (time.time() - start_time) > timeout:
                process.terminate()
                print(f"⏰ 命令执行超时 (超过 {timeout} 秒)")
                return False
        
        process.wait()
        
        if process.returncode == 0:
            print(f"✅ {description} 成功完成!")
            return True
        else:
            print(f"❌ {description} 失败! 返回码: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ 执行命令时出错: {e}")
        return False

def check_environment():
    """检查环境是否准备就绪"""
    print("🔍 检查训练环境...")
    
    # 创建dataset.yaml文件
    create_dataset_yaml()
    
    # 检查dataset.yaml是否存在
    if not os.path.exists("dataset.yaml"):
        print("❌ 错误: 未找到 dataset.yaml 文件")
        return False
    
    # 检查YOLO是否可用
    try:
        result = subprocess.run(
            "yolo --version", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        if result.returncode != 0:
            print("❌ YOLO 命令不可用，请确保Ultralytics YOLO已安装")
            return False
        print(f"✅ YOLO 版本可用")
    except:
        print("❌ 无法执行YOLO命令")
        return False
    
    print("✅ 环境检查通过")
    return True

def find_latest_model():
    """查找最新训练的模型"""
    print("\n🔍 查找最新训练的模型权重...")
    weights_dir = Path("runs/detect")
    
    if not weights_dir.exists():
        print("❌ 未找到训练输出目录")
        return None
    
    # 查找所有训练目录并按创建时间排序
    train_dirs = list(weights_dir.glob("train*"))
    if not train_dirs:
        print("❌ 未找到训练目录")
        return None
    
    # 按修改时间排序，找到最新的
    train_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_train = train_dirs[0]
    best_model_path = latest_train / "weights" / "best.pt"
    
    if best_model_path.exists():
        print(f"✅ 找到最佳模型: {best_model_path}")
        return str(best_model_path)
    else:
        print(f"❌ 未找到最佳模型文件: {best_model_path}")
        return None

def train_model():
    """执行模型训练"""
    print("\n🎯 开始模型训练流程...")
    
    # 训练命令 - 使用更详细的参数
    train_command = (
        "yolo train "
        "model=yolo11n.pt "
        "data=dataset.yaml "
        "epochs=50 "
        "imgsz=640 "
        "batch=16 "
        "device=cpu "
        "workers=4 "
        "patience=10 "
        "save=True "
        "exist_ok=True"
    )
    
    # 训练可能需要较长时间，设置长超时或None
    success = run_command(train_command, "YOLOv11模型训练", timeout=None)
    
    if success:
        print("\n✅ 模型训练完成!")
        return True
    else:
        print("\n❌ 模型训练失败!")
        return False

def test_model(model_path):
    """执行模型测试"""
    print("\n🧪 开始模型测试流程...")
    
    test_command = (
        f"yolo val "
        f"model={model_path} "
        f"data=dataset.yaml "
        f"split=test "
        f"verbose=True"
    )
    
    success = run_command(test_command, "模型性能测试", timeout=3600)  # 1小时超时
    
    if success:
        print("\n✅ 模型测试完成!")
        return True
    else:
        print("\n❌ 模型测试失败!")
        return False

def main():
    """主执行函数"""
    print("=" * 70)
    print("🤖 YOLOv11 自动训练测试管道")
    print("=" * 70)
    
    # 环境检查
    if not check_environment():
        sys.exit(1)
    
    # 执行训练
    if not train_model():
        print("训练失败，退出程序")
        sys.exit(1)
    
    # 查找训练好的模型
    model_path = find_latest_model()
    if not model_path:
        # 尝试使用手动路径作为备选
        manual_path = "~/SPLObjDetectDatasetV2/runs/detect/train2/weights/best.pt"
        expanded_path = os.path.expanduser(manual_path)
        if os.path.exists(expanded_path):
            print(f"✅ 使用手动路径找到模型: {expanded_path}")
            model_path = expanded_path
        else:
            print("❌ 无法找到可用的模型文件")
            sys.exit(1)
    
    # 执行测试
    test_model(model_path)
    
    print("\n" + "=" * 70)
    print("🎉 所有流程执行完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()
