# 金属板表面划痕实时检测系统

基于 **Allied Vision 相机 + Vimba SDK** 的完整机器视觉检测系统，支持相机标定、实时划痕检测与标记。

---

## 系统架构

```
金属板划痕检测系统/
├── run.py                  # 启动入口
├── main_app.py             # GUI 主程序 (tkinter)
├── camera_calibration.py   # Halcon 7×7 圆点阵标定模块
├── scratch_detector.py     # 划痕检测核心算法
├── camera_interface.py     # Vimba SDK 相机接口封装
├── visualizer.py           # 实时可视化与标记
├── config.py               # 全局配置参数
├── requirements.txt        # Python 依赖
├── calib_images/           # 标定图像存储
├── config/                 # 标定参数 JSON 存储
└── output/                 # 截图/结果输出
```

## 硬件要求

| 组件 | 型号/规格 |
|------|----------|
| 相机 | Allied Vision Mako / Goldeye / Alvium 系列 (GigE / USB3) |
| SDK | Vimba SDK (从 Allied Vision 官网下载安装) |
| 标定板 | Halcon 7×7 圆点阵，点间距 10 mm (或已知间距) |
| 光源 | 同轴光或环形光，确保标定板和工件划痕对比度 |
| 系统 | Windows 10/11, Python 3.9+ |

## 安装

### 1. 安装 Vimba SDK

从 [Allied Vision 官网](https://www.alliedvision.com/en/products/software/) 下载并安装 Vimba SDK。

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

> **注意**: `vmbpy` 需要 Vimba SDK 已正确安装且环境变量 `GENICAM_GENTL64_PATH` 指向 Vimba 的 GenTL 目录。

如果无法安装 vmbpy，系统会自动切换到模拟相机模式用于测试。

## 工作流程

### 完整检测流程

```
相机连接 → 标定 → 实时采集 → 预处理 → 划痕检测 → 标记显示 → 结果输出
```

### 1. 相机标定 (使用前必须完成)

| 步骤 | 操作 |
|------|------|
| 1 | 放置 Halcon 7x7 圆点阵标定板在相机视野内 |
| 2 | 点击"连接相机" → "开始预览" |
| 3 | 调整光源和曝光使圆点清晰可见 |
| 4 | 改变标定板角度/位置, 点击"采集标定图像" (≥10张) |
| 5 | 点击"执行标定", 检查 RMS 误差 < 0.5px |
| 6 | 点击"保存标定参数" |

**标定板摆放建议**:
- 覆盖图像的四角和中心
- 角度变化 15°~45°
- 至少 10 张，越多越好 (最多 30 张)

### 2. 实时划痕检测

1. 切换到"检测参数"选项卡，调节参数
2. 切换到"相机控制"选项卡，设置曝光/增益
3. 点击"▶ 开始检测"

### 检测算法

系统使用 **多方法融合** 策略检测划痕：

| 方法 | 适用场景 | 原理 |
|------|---------|------|
| **Top-Hat 形态学** | 细划痕、低对比度 | 顶帽变换突出比背景亮/暗的细小结构 |
| **Canny + 霍夫变换** | 长划痕、边缘清晰 | 边缘检测+概率霍夫线检测 |
| **Gabor 滤波器组** | 方向性纹理划痕 | 多方向 Gabor 滤波，取最大响应 |

后处理包含连通域分析、几何特征计算和伪缺陷过滤。

## 使用方式

```bash
# 启动完整 GUI
python run.py

# 运行算法自测 (无需相机)
python run.py --test

# 仅测试标定模块
python run.py --calib-only
```

## 界面说明

| 区域 | 功能 |
|------|------|
| **实时预览** | 三视图: 原始+标记 / 缺陷掩膜 / 边缘检测 |
| **相机标定** | 标定图像采集、执行标定、保存/加载 |
| **检测参数** | 调节预处理、检测算法和判定阈值 |
| **相机控制** | 曝光、增益、开始/停止检测、截图 |
| **检测详情** | 每条划痕的长度/宽度/角度/置信度 |

## 划痕判定标准

系统根据以下几何特征区分划痕与噪声:

- **最小长度**: 默认 25 px (可调)
- **最小面积**: 默认 100 px² (可调)
- **最小长宽比**: 默认 3.0 (划痕细长，点缺陷近圆形)
- **最大宽度**: 默认 50 px (可调)

## 输出

- `config/calibration_params.json` - 相机标定参数
- `output/screenshot_*.png` - 检测截图
- `output/test_*.png` - 自测结果图像

## 常见问题

**Q: 相机无法连接?**
A: 检查 Vimba SDK 是否正确安装，运行 Vimba Viewer 确认相机可被识别。

**Q: vmbpy 安装失败?**
A: 确保 Vimba SDK 已安装，且 Python 是 64 位版本。也可以使用模拟模式测试算法。

**Q: 标定 RMS 误差过大?**
A: 确保标定板平整、光照均匀、圆点清晰，增加标定图像数量和角度变化。

**Q: 划痕检测率低?**
A: 调节"检测参数"选项卡中的阈值，或改善光源条件。

## 参考

- [Allied Vision Vimba SDK](https://www.alliedvision.com/en/products/software/)
- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Halcon Dot Pattern Calibration](https://www.mvtec.com/products/halcon)

