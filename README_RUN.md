# 云游戏帧率模拟器 - 运行指南

## 快速开始

### 1. 单文件模拟
```bash
python simulator_v2.py <trace_file.csv>
```

### 2. 批量处理目录
```bash
python simulator_v2.py <trace_directory>

# 指定线程数
python simulator_v2.py <trace_directory> 32
```

## 核心参数配置

在代码开头修改这些全局变量：

### 显示模式 (DISPLAY_MODE)
```python
DISPLAY_MODE = "simpleCtrl"  # 推荐：智能控制
# DISPLAY_MODE = "naiveVsync"  # 基线：简单vsync
# DISPLAY_MODE = "optimal"     # 上界：最优策略（需要未来信息）
```

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `naiveVsync` | 固定缓冲区，无优化 | 基线对比 |
| `simpleCtrl` | 动态缓冲区 + 主动丢帧 | **推荐使用** |
| `optimal` | 回溯优化（理论上界） | 性能上限评估 |

### 缓冲区设置
```python
MAX_BUF_SIZE = 2              # 最大缓冲帧数 (1-4)
DROP_FRAME_MODE = "lifo"      # 丢帧模式: lifo(后进先出) / fifo(先进先出)
```

### 周期性丢帧 (ENABLE_PERIO_DROP)
```python
ENABLE_PERIO_DROP = 2         # 0=关闭, 1=宽松, 2=严格
BONUS_FPS_NO_THR = 30         # 触发阈值：连续N帧后主动降延迟
```
- `0`: 禁用周期性丢帧
- `1`: 时间+帧数双条件
- `2`: **推荐** - 严格连续帧条件

### 快速丢帧 (ENABLE_QUICK_DROP)
```python
ENABLE_QUICK_DROP = 0         # 0=关闭, 1-2=基于抖动预测
JITTER_HISTORY_LTH = 600      # 历史窗口长度（帧数）
```
- `0`: **推荐** - 禁用（需要ML模型）
- `1`: 简单概率预测
- `2`: 增强版预测器

### 渲染时间预测 (RENDER_TIME_PREIDCTER)
```python
RENDER_TIME_PREIDCTER = "ewma"  # 推荐
# RENDER_TIME_PREIDCTER = "fixed"   # 固定值1ms
# RENDER_TIME_PREIDCTER = "oracle"  # 使用真实值（仅测试）
```

### 其他参数
```python
FRAME_INTERVAL = 16.666667    # 帧间隔 (60fps = 16.67ms)
ANCHOR_FRAME_EXTRAPOLATOR_MODE = 4  # PTS预测模式 (0-4)
PRINT_LOG = True              # 输出详细日志
PRINT_DEBUG_LOG = False       # 调试日志（超详细）
```

## 输出结果

### 结果文件
- **单文件**: `<trace>_<mode>_quickdrop<X>_periodrop<X>_..._sim.csv`
- **批量**: `result-<mode>-periodrop<X>_quickdrop<X>_....csv`

### 关键指标
- `origin_fps`: 原始帧率
- `optimized_fps`: 优化后帧率
- `origin_render_queue`: 原始渲染队列延迟 (ms)
- `optimized_render_queue`: 优化后延迟 (ms)
- `*_jitter_induced_queue`: 各类抖动引起的队列次数

## 典型配置示例

### 配置1: 激进降延迟
```python
DISPLAY_MODE = "simpleCtrl"
ENABLE_PERIO_DROP = 2
BONUS_FPS_NO_THR = 20         # 20帧后就优化
MAX_BUF_SIZE = 2
```

### 配置2: 保守稳定
```python
DISPLAY_MODE = "simpleCtrl"
ENABLE_PERIO_DROP = 1
BONUS_FPS_NO_THR = 45         # 45帧后才优化
MAX_BUF_SIZE = 3
```

### 配置3: 基线对比
```python
DISPLAY_MODE = "naiveVsync"
ENABLE_PERIO_DROP = 0
MAX_BUF_SIZE = 2
```

### 配置4: 理论上界
```python
DISPLAY_MODE = "optimal"
ENABLE_PERIO_DROP = 2
RENDER_TIME_PREIDCTER = "oracle"
```

## 多参数批量评估

修改 `MULTI_PARAMS` 列表：
```python
MULTI_PARAMS = [
    # [模式, buf_size, 渲染预测, 周期丢帧, 快速丢帧, bonus阈值, 丢帧模式]
    ["simpleCtrl", 2, "ewma", 2, 0, 30, "lifo"],
    ["optimal", 2, "ewma", 2, 0, 30, "lifo"],
    ["naiveVsync", 2, "ewma", 1, 0, 30, "lifo"],
]
```

## 性能优化

- **单文件测试**: 直接运行，快速查看结果
- **批量评估**: 自动使用多线程（默认32线程）
- **调试模式**: `PRINT_DEBUG_LOG = True` 输出逐帧详情

## 注意事项

1. **Trace格式**: 需要包含完整的帧timing信息（65列）
2. **帧率要求**: 默认只处理60fps trace，其他会自动分类
3. **依赖项**: 确保安装 numpy, load_data, trace_e2e_jitter_analyze

## 示例命令

```bash
# 单个trace文件
python simulator_v2.py ./log/session_info_11.177.33.17_2024-06-13/trace_001.csv

# 批量处理整个目录
python simulator_v2.py ./src/model/action/

# 使用16线程批量处理
python simulator_v2.py ./src/model/action/ 16
```
