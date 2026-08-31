# B2-Z1 低层策略 Play 视频录制手册（EGL 无头录制）

本文说明如何在无 Xorg 的 GPU 服务器上用 `play_aliengo_z1_video.py` 录制
low-level 策略回放视频（H.264 MP4），包括参数含义、录制命令、验证方法、
常见问题，以及本仓库（`visual_wholebody`，分支 `b2_z1_test`）与
`visual_wholebody_origin` 仓库的差异。

## 1. 脚本与本仓库差异（重要）

本仓库的录制脚本：

```text
low-level/legged_gym/scripts/play_aliengo_z1_video.py
```

它基于 `play.py`，只替换录像函数，不修改 checkpoint、URDF、奖励、
策略输入或训练代码。主要能力：

1. 自动把本 checkout 的 Isaac Gym、RSL-RL、low-level 包插入 `sys.path` 最前，
   避免误用另一份 checkout 的 editable install。
2. 自动写 `/tmp/nvidia_icd_egl.json` 并设置 `VK_ICD_FILENAMES`、
   `XDG_RUNTIME_DIR`（默认 EGL ICD；保留 graphics device 但不创建 viewer）。
3. 每帧先 `gym.fetch_results()` 同步 GPU PhysX 与图形场景，再
   `step_graphics()` + `render_all_camera_sensors()` 离屏渲染。
4. 相机用世界坐标跟随机身（`root_states[..., :3]`），避免 terrain
   env_origin 导致拍空。
5. 用 Pillow 把 VBC 调试标记投影叠加到 720x480 camera RGB 帧上。

### 1.1 与 `visual_wholebody_origin` 的差异（当前仓库已核实的代码事实）

| 项目 | 本仓库 `visual_wholebody` | origin 仓库 README |
| --- | --- | --- |
| 脚本文件 | 同一个文件名，位置相同 | 相同 |
| 额外 CLI 参数 `--play_seconds`、`--video_tag`、`--zero_commands` | ❌ 不存在（`helpers.get_args()` 无此参数） | ✅ 有 |
| 回放长度 | 固定 `env.max_episode_length` 步（`episode_length_s=10`，约 500 步/10 s） | 可用 `--play_seconds` 控制 |
| 录像模式 | 由 `env_cfg.env.record_video`（config）驱动 | 由 `--record_video` 驱动 |
| 叠加标记 | 蓝色/黄色/青色圆环、红点轨迹、红色碰撞框、目标姿态 RGB 轴 | 相同 |

> 结论：在本仓库不要使用 origin 文档中的 `--play_seconds` / `--video_tag`
> 参数，它们不会被解析（会报 unknown argument）。
> 视频固定 ~10 s 长度；文件名格式见第 4 节。

## 2. 录制脚本用法

### 2.1 参数说明（`get_args()`，来自 `legged_gym/utils/helpers.py`）

与录制直接相关的参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--task` | `b2z1` | 任务名，决定环境配置；b2z1 系列用 `b2_z1_...` 前缀，见第 3 节 |
| `--exptid` | 无（必填） | checkpoint 所在 run 目录名（`model_<checkpoint>.pt` 所在目录名） |
| `--proj_name` | `b2z1-low` | run 文件夹上层（`logs/<proj_name>/<exptid>/`） |
| `--checkpoint` | `-1` | 加载的 checkpoint 号，`-1` 表示最后一个 |
| `--record_video` | `False` | 录制视频开关（本仓库由它触发 camera sensor 录像） |
| `--flat_terrain` | `False` | 平地地形（不加则用 trimesh rough terrain 6x3） |
| `--stochastic` | `False` | 随机策略动作（默认确定性推理） |
| `--stand_by` | `False` | 待机模式（不进 goal 更新） |
| `--sim_device` | `cuda:0` | `physics` 设备 |
| `--rl_device` | `cuda:0` | 推理设备 |
| `--pipeline` | `gpu` | Tensor API pipeline |
| `--graphics_device_id` | `0` | 图形设备（EGL 离屏渲染用） |
| `--headless` | `True` | ⚠️ 不要追加普通 `--headless`，脚本内部用 sentinel 覆盖 |

其余参数（`--resume`、`--load_run`、`--num_envs`、`--debug` 等）主要用于训练/调试，
录制时不需要。

### 2.2 典型的完整录制命令

```bash
cd /root/rivermind-data/visual_wholebody

export LD_LIBRARY_PATH="/opt/conda/envs/b1z1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions

/opt/conda/envs/b1z1/bin/python -u \
  low-level/legged_gym/scripts/play_aliengo_z1_video.py \
  --task b2_z1_reachable_workspace_motion \
  --exptid b2_z1_reachable_workspace_motion_20260828_105622 \
  --proj_name b2z1-low \
  --checkpoint 10500 \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --pipeline gpu \
  --record_video
```

> 建议激活 `b1z1` conda 环境后直接用 `python`，或用绝对路径
> `/opt/conda/envs/b1z1/bin/python`。脚本会在开头
> 打印 `Importing module 'gym_38' (.../visual_wholebody/third_party/isaacgym/...)`，
> 路径必须包含 `visual_wholebody`（不是 origin）。

### 2.3 回放长度

本仓库录像长度固定为 `env.max_episode_length`：

```text
episode_length_s = 10        # low-level/legged_gym/envs/manip_loco/b1z1_config.py
dt = 0.02                    # 500 Hz / 50 步每秒 ⇒ 约 500 步 / 10 秒
```

`play_aliengo_z1_video.py` 的 `render_record()` 只在偶数步
（`global_steps % 2 == 0`）写帧，因此 25 fps，共约 250 帧。
若想录更长视频，需要改 `episode_length_s` 或 duration 相关代码（当前不支持 CLI 参数）。

## 3. 任务名（b2z1 系列，注册表 `legged_gym/envs/__init__.py`）

| `--task` 值 | 对应 config |
| --- | --- |
| `b2_z1_reachable_workspace` | `B2Z1ReachableWorkspaceCfg` |
| `b2_z1_reachable_workspace_motion` | `B2Z1ReachableWorkspaceMotionCfg` |
| `b2_z1_reachable_workspace_motion_plus` | `B2Z1ReachableWorkspaceMotionPlusCfg` |
| `b2_z1_reachable_balanced` | `B2Z1ReachableBalancedCfg` |
| `b2_z1_bounded_actions` / `b2_z1_aggressive_locomotion` / `b2z1` | 其他基础任务 |

对应 `experiment_name`（config 里配置的 run 名，`train.py` 训练时的
wandb run 也遵循它）：

```text
B2Z1ReachableWorkspaceCfgPPO          → experiment_name = 'b2_z1_reachable_workspace'
B2Z1ReachableWorkspaceMotionCfgPPO    → experiment_name = 'b2_z1_reachable_workspace_motion'
B2Z1ReachableWorkspaceMotionPlusCfgPPO → experiment_name = 'b2_z1_reachable_workspace_motion_plus'
```

> 注意：训练 run 目录名（`--exptid`）带时间戳（如
> `b2_z1_reachable_workspace_motion_20260828_105622`），
> 而 `experiment_name` 是任务逻辑名。录制时 `--exptid` 必须填时间戳目录名，
> `--task` 必须与训练所用任务匹配，否则 checkpoint 结构/obs 维度对不上。

## 4. 路径规则与输出

checkpoint 输入：

```text
low-level/logs/<proj_name>/<exptid>/model_<checkpoint>.pt
```

视频输出目录：

```text
low-level/logs/videos/<exptid>/
```

文件名格式（本仓库 `play.py` 逻辑）：

```text
<exptid>-<environment_index>-<checkpoint>.mp4
```

示例（已生成）：

```text
low-level/logs/videos/b2_z1_reachable_workspace_motion_20260828_105622/b2_z1_reachable_workspace_motion_20260828_105622-0-10500.mp4
low-level/logs/videos/b2_z1_reachable_workspace_motion_plus_20260828_105850/b2_z1_reachable_workspace_motion_plus_20260828_105850-0-11500.mp4
```

## 5. 两个已解决的关键问题（代码层面）

1. **手臂/腿折叠在机身上**：GPU pipeline 下物理张量已更新但图形场景未同步。
   脚本每帧先执行 `gym.fetch_results(env.sim, True)` 再 `step_graphics()`。
2. **同步后画面为空**：terrain 环境机器人世界坐标含 `env_origin`，旧脚本误减
   env_origin 后传给 `set_camera_location()`；脚本直接用
   `root_states[..., :3]` 世界坐标作为相机跟随基准（相机位于
   `root + (-0.9, 1.6, 0.65)`，看向 `root + (0.2, 0, 0.12)`）。

## 6. 标记含义（叠加在输出帧上）

camera sensor 的 `IMAGE_COLOR` 不包含 `gymutil.draw_lines()` 的 viewer 调试层，
脚本用相同环境状态在帧上重绘（仅可视化，不影响仿真/策略）：

| 标记 | 含义 | 数据来源 |
| --- | --- | --- |
| 蓝色圆环 | 实际末端位置（current pose） | `env.ee_pos` |
| 黄色圆环 | 当前目标位置（target pose） | `env.curr_ee_goal_cart_world` |
| 红绿蓝坐标轴 | 目标姿态 | `env.ee_goal_orn_quat` |
| 青色圆环 | EE 球坐标目标空间中心 | `env._get_ee_goal_spherical_center()` |
| 红色点 | EE 起点→最终目标的插值轨迹 | `ee_start_sphere`, `ee_goal_sphere` |
| 红色边框 | 机械臂目标碰撞限制区域 | `collision_lower_limits`, `collision_upper_limits` |

## 7. 验证视频

```bash
VIDEO=/root/rivermind-data/visual_wholebody/low-level/logs/videos/b2_z1_reachable_workspace_motion_20260828_105622/b2_z1_reachable_workspace_motion_20260828_105622-0-10500.mp4

ffprobe -v error \
  -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames \
  -show_entries format=duration,size \
  -of default=noprint_wrappers=1 \
  "$VIDEO"
```

已验证输出：

```text
codec_name=h264
width=720
height=480
r_frame_rate=25/1
nb_frames=250
duration=10.000000
```

抽取中间帧检查标记：

```bash
ffmpeg -y -ss 4.5 -i "$VIDEO" -frames:v 1 /tmp/b2z1_video_mid.png
```

## 8. 常见问题

- **图形初始化失败 / GLFW/X11 错误**：确认脚本在进程启动、导入 isaacgym 前设置
  EGL ICD；查看 `/tmp/nvidia_icd_egl.json`（应为 `libEGL_nvidia.so.0`）。
- **导入了另一份 visual_wholebody**：检查启动日志
  `Importing module 'gym_38' (.../visual_wholebody/third_party/isaacgym/...)`。
- **找不到 checkpoint**：按第 4 节拼路径检查 `--proj_name`/`--exptid`/`--checkpoint`。
- **视频存在但无标记**：必须使用 `play_aliengo_z1_video.py`，而不是 `play.py`。
- **cv2 缺 libGL.so.1**：脚本用 Pillow 不依赖 OpenCV，无需安装。
- **terrain 初始化慢/显存高**：rough terrain 三角形多；先用
  `--flat_terrain` 做 smoke test。
- **unknown argument: --play_seconds**：这是 origin 仓库的参数，本仓库无此参数，
  见第 1.1 节。

## 9. 本仓库当前已生成的视频

```text
/root/rivermind-data/visual_wholebody/low-level/logs/videos/b2_z1_reachable_workspace_motion_20260828_105622/b2_z1_reachable_workspace_motion_20260828_105622-0-10500.mp4
/root/rivermind-data/visual_wholebody/low-level/logs/videos/b2_z1_reachable_workspace_motion_plus_20260828_105850/b2_z1_reachable_workspace_motion_plus_20260828_105850-0-11500.mp4
```

两个视频均验证：h264 / 720x480 / 25fps / 250 帧 / 10.0 s。