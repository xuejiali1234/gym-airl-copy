# SUMO 合流 PoC

这套代码用于验证当前裸模型 `U220_D230_epoch290` 能否在 SUMO 中控制匝道车，而主线背景车继续使用 SUMO 内置模型。

## 目录

- `build_merge_network.py`
  - 生成简化合流路网与基础配置
- `sumo_obs_adapter.py`
  - 把 SUMO 车辆状态转成当前模型可吃的 18 维观测
- `run_sumo_merge_policy.py`
  - 启动 TraCI、加载模型、控制匝道车、记录轨迹与接管结果
  - 支持匝道车辆按顺序交替出现 `RL / SUMO / RL / SUMO`
- `net/`
  - 路网 XML 与生成后的 `.net.xml`
- `routes/`
  - 路由模板
- `cfg/`
  - SUMO 配置模板
- `outputs/`
  - 运行输出

## 默认行为

- 默认会连续生成多辆匝道车，并按顺序交替为 `RL / SUMO`
- 主线车始终使用 SUMO 默认行为
- RL 匝道车进入目标车道并连续保持 10 个步长后，切回 SUMO 默认控制

## 运行

先生成路网：

```powershell
python .\sumo测试\build_merge_network.py
```

再运行 headless PoC：

```powershell
python .\sumo测试\run_sumo_merge_policy.py
```

更难的对比版示例：

```powershell
python .\sumo测试\run_sumo_merge_policy.py --steps 900 --mainline-flow-rate 2200 --ramp-vehicle-count 6 --ramp-headway-seconds 7.0
```

如果想看 GUI：

```powershell
python .\sumo测试\run_sumo_merge_policy.py --gui
```

## 主要输出

- `summary.json`
- `step_log.csv`
- `vehicle_trajectory.csv`
- `vehicle_summary.csv`
- `controller_compare.csv`
- `tripinfo.xml`
- `fcd.xml`

其中 `summary.json` 会记录：

- RL / SUMO 匝道车数量
- 分组对比结果
- 是否完成合流
- 是否完成 SUMO 接管
- 是否发生碰撞
- 最终停在哪个 edge/lane
