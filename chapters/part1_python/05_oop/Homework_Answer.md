# 第 6 课：Python 面向对象编程基础 — 课后题参考答案

## 练习 6-1：图像预处理器类

```python
class ImagePreprocessor:
    """图像预处理器——管理图像预处理的配置和状态"""

    def __init__(self, target_size=(224, 224), normalize=True):
        self.target_size = target_size
        self.normalize = normalize
        self.original_size = None
        self.processed_count = 0

    def resize(self, width, height):
        """记录原始尺寸并计算缩放比例"""
        self.original_size = (width, height)
        scale_w = self.target_size[0] / width
        scale_h = self.target_size[1] / height
        print(f"缩放: {width}x{height} → {self.target_size[0]}x{self.target_size[1]}")
        print(f"缩放比例: w={scale_w:.3f}, h={scale_h:.3f}")
        self.processed_count += 1

    def get_config(self):
        """返回当前预处理配置"""
        return {
            "target_size": self.target_size,
            "normalize": self.normalize,
            "processed_count": self.processed_count,
            "last_original_size": self.original_size
        }


# 测试
preprocessor_cv = ImagePreprocessor(target_size=(640, 640), normalize=True)
preprocessor_cls = ImagePreprocessor(target_size=(224, 224), normalize=False)

preprocessor_cv.resize(1920, 1080)
preprocessor_cls.resize(800, 600)

print("CV 预处理器配置:", preprocessor_cv.get_config())
print("分类预处理器配置:", preprocessor_cls.get_config())
```

---

## 练习 6-2：检测框类

```python
class BoundingBox:
    """边界框——表示目标检测中的一个检测结果"""

    def __init__(self, x1, y1, x2, y2, label, confidence):
        # 确保坐标顺序正确
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.label = label
        self.confidence = confidence

    def width(self):
        return self.x2 - self.x1

    def height(self):
        return self.y2 - self.y1

    def area(self):
        return self.width() * self.height()

    def is_square(self, tolerance=0.1):
        """判断是否接近正方形"""
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return False
        ratio = max(w, h) / min(w, h)
        return ratio <= 1 + tolerance

    def __str__(self):
        return (f"{self.label}: ({self.x1},{self.y1})-({self.x2},{self.y2}) "
                f"[{self.area()}px²] conf={self.confidence:.2f}")


# 测试
box1 = BoundingBox(100, 200, 300, 500, "person", 0.95)
box2 = BoundingBox(50, 100, 250, 300, "car", 0.88)
box3 = BoundingBox(10, 10, 110, 110, "cat", 0.72)

print(box1)                      # person: (100,200)-(300,500) [60000px²] conf=0.95
print(f"面积: {box1.area()}")    # 60000
print(f"是否正方形: {box1.is_square()}")   # False (200x300)
print(f"是否正方形: {box3.is_square()}")   # True (100x100)
```

---

## 练习 6-3：继承练习——数据增强器

```python
class ImagePreprocessor:
    """图像预处理器（练习 6-1 的实现）"""
    def __init__(self, target_size=(224, 224), normalize=True):
        self.target_size = target_size
        self.normalize = normalize
        self.processed_count = 0

    def resize(self, width, height):
        self.processed_count += 1
        print(f"缩放: {width}x{height} → {self.target_size[0]}x{self.target_size[1]}")

    def get_config(self):
        return {
            "target_size": self.target_size,
            "normalize": self.normalize,
            "processed_count": self.processed_count
        }


class ImageAugmentor(ImagePreprocessor):
    """图像数据增强器——继承自 ImagePreprocessor"""

    def __init__(self, target_size=(224, 224), normalize=True,
                 rotation_range=30, flip_probability=0.5):
        # ✅ 调用父类的 __init__
        super().__init__(target_size, normalize)
        # 子类特有的属性
        self.rotation_range = rotation_range
        self.flip_probability = flip_probability

    def get_augmentation_params(self):
        """返回数据增强的参数配置"""
        return {
            "rotation_range": f"±{self.rotation_range}°",
            "flip_probability": f"{self.flip_probability:.0%}",
        }

    def get_config(self):               # ← 重写父类方法
        """在父类配置基础上增加增强参数"""
        config = super().get_config()   # 先拿父类的配置
        # 添加数据增强特有的配置
        config["augmentation"] = self.get_augmentation_params()
        return config


# 测试
augmentor = ImageAugmentor(
    target_size=(416, 416),
    normalize=True,
    rotation_range=45,
    flip_probability=0.7
)

augmentor.resize(1920, 1080)
print(augmentor.get_config())
# 输出包含:
#   target_size, normalize, processed_count (来自父类)
#   augmentation: {rotation_range, flip_probability} (子类新增)
```

---

## 练习 6-4：视频文件管理器

```python
class VideoFile:
    """视频文件类"""

    def __init__(self, filename, duration, resolution, file_size_mb):
        self.filename = filename
        self.duration = duration          # 秒
        self.resolution = resolution      # (width, height)
        self.file_size_mb = file_size_mb

    def get_info(self):
        """返回格式化的文件信息"""
        w, h = self.resolution
        return (f"{self.filename} | {w}x{h} | "
                f"{self.duration}s | {self.file_size_mb}MB")

    def is_4k(self):
        """判断是否为 4K 分辨率"""
        return self.resolution[0] >= 3840


class VideoManager:
    """视频文件管理器"""

    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.videos = []

    def add_video(self, video):
        """添加视频文件"""
        self.videos.append(video)
        print(f"已添加: {video.filename}")

    def remove_video(self, filename):
        """按文件名删除视频"""
        for i, v in enumerate(self.videos):
            if v.filename == filename:
                removed = self.videos.pop(i)
                print(f"已删除: {removed.filename}")
                return
        print(f"未找到文件: {filename}")

    def get_total_size(self):
        """返回所有视频的总大小"""
        return sum(v.file_size_mb for v in self.videos)

    def filter_by_resolution(self, min_width):
        """返回分辨率不低于指定宽度的视频"""
        return [v for v in self.videos if v.resolution[0] >= min_width]

    def list_all(self):
        """列出所有视频文件信息"""
        if not self.videos:
            print("暂无视频文件")
            return
        print(f"存储路径: {self.storage_path}")
        print(f"共 {len(self.videos)} 个视频文件 ({self.get_total_size()}MB):")
        for v in self.videos:
            print(f"  {v.get_info()}")


# 测试
manager = VideoManager("/home/jetson/videos/")

manager.add_video(VideoFile("cam0_20240215.mp4", 300, (1920, 1080), 512))
manager.add_video(VideoFile("cam1_20240215.mp4", 180, (3840, 2160), 1200))
manager.add_video(VideoFile("cam2_20240215.mp4", 600, (1280, 720), 800))

manager.list_all()

print("\n4K 视频:")
for v in manager.filter_by_resolution(3840):
    print(f"  {v.get_info()}")

manager.remove_video("cam2_20240215.mp4")
print(f"\n删除后总大小: {manager.get_total_size()}MB")
```

---

## 练习 6-5：代码重构

```python
class GPUMonitor:
    """GPU 温度监控器"""

    def __init__(self, gpu_name, alert_threshold=80, warning_threshold=65):
        self.gpu_name = gpu_name
        self.alert_threshold = alert_threshold
        self.warning_threshold = warning_threshold
        self._temperature_history = []

    def record(self, temp):
        """记录一次温度读数"""
        self._temperature_history.append(temp)

        if temp >= self.alert_threshold:
            print(f"🚨 [{self.gpu_name}] 告警！温度 {temp}°C 超过 {self.alert_threshold}°C！")
        elif temp >= self.warning_threshold:
            print(f"⚠️ [{self.gpu_name}] 注意：温度 {temp}°C 超过 {self.warning_threshold}°C")
        else:
            print(f"✅ [{self.gpu_name}] 温度正常: {temp}°C")

    def get_average(self):
        """返回平均温度"""
        if not self._temperature_history:
            return 0.0
        return sum(self._temperature_history) / len(self._temperature_history)

    def get_max(self):
        """返回最高温度"""
        if not self._temperature_history:
            return 0.0
        return max(self._temperature_history)

    def reset(self):
        """清空历史记录"""
        count = len(self._temperature_history)
        self._temperature_history.clear()
        print(f"[{self.gpu_name}] 温度历史已清空 (共 {count} 条记录)")

    def get_status(self):
        """获取完整状态报告"""
        return {
            "gpu": self.gpu_name,
            "readings": len(self._temperature_history),
            "average": round(self.get_average(), 1),
            "max": self.get_max(),
            "alert_threshold": self.alert_threshold,
            "warning_threshold": self.warning_threshold,
        }


# 测试：创建多个互不干扰的监控器
gpu0_monitor = GPUMonitor("GPU-0", alert_threshold=85, warning_threshold=70)
gpu1_monitor = GPUMonitor("GPU-1", alert_threshold=75, warning_threshold=60)

# 模拟温度读数
import random
for _ in range(5):
    gpu0_monitor.record(random.uniform(50, 90))
    gpu1_monitor.record(random.uniform(50, 80))

print("\nGPU-0 状态:", gpu0_monitor.get_status())
print("GPU-1 状态:", gpu1_monitor.get_status())

# 验证独立性
gpu0_monitor.reset()
print("\n重置后 GPU-0 状态:", gpu0_monitor.get_status())
print("GPU-1 状态不受影响:", gpu1_monitor.get_status())
```
