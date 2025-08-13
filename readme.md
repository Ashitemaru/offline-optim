# Tencent Trace README

## 代码说明

- `data` 需要自备，为腾讯提供的云游戏传输 trace log
- `image` 用于存储各种结果图
- `log` 用于存储各个算法的简要 log，每个 log 的结构都是一行一个 trace，各项分别是 trace 名、FPS、平均延迟、撕裂率、有效帧比例
- `model` 曾用于存储枚举表示的 RL policy，现一般不用
- `src` 为源代码
    - `algorithm` 存储各种算法
        - `greedy.py` 内有完全贪心和基于 baseline 的完全贪心两种算法
        - `mpc.py` 内为最初的 MPC RL 算法，现在完全废弃
        - `policy_iteration.py` 为 PI RL 算法，由于效率和收敛速度问题，现在完全废弃
        - `value_iterator.py` 为 VI RL 算法，现行离线最优算法
    - `ref` 为参考代码
        - `caller.py` 为之前调用 baseline 时使用的外围代码，现在废弃
        - `load_data.py` 为原先读取数据的代码
        - `naive.py` 为离线最优算法的第一版代码
        - `trace_simulate.py` 为 baseline 的原版代码
    - `util` 为各种工具代码
        - `compare.py` 用于比较两个算法的 log，绘制各种 CDF
        - `data_loader.py` 为读取各种 trace/log 的代码
        - `estimator.py` 为部分 estimator class
        - `log_process.py` 曾用于处理细粒度 trace log
    - `enhanced_baseline.py` 为现行 baseline 的增强版代码
    - `simulator.py` 为主代码，基于原版 baseline（主要为了保证 RL 所求解的环境与 baseline 一致），增加了一些外围功能，并且将离线最优、贪心等算法集成进去

## 代码运行

baseline 算法直接运行：

```shell
cd src && python enhanced_baseline.py
```

最终结果会存储在 `log/baseline-enhanced.log` 下。

其他算法均通过：

```shell
cd src && python simulator.py
```

运行，运行前需要手动调整调用 `trace_simulator` 函数时的 `mode` 参数，以及 log 存储的路径。