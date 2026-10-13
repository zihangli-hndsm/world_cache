# MultiScan 原生 RGB-D 投影诊断

日期：2026-10-11 UTC
结论：**A — 找到并修复了有官方格式依据的技术错误；原定 5 cm 门槛在原样本复测通过。**

## 结论摘要

失败就是 `paper/supplementary.tex` 中记录的 MultiScan scene `00069` pilot：同一数据修订、同一对扫描、同一投影脚本和相同的每扫描 72 个均匀抽样帧。不是另一个实验或新挑出的场景。扫描归档 ID、SHA256 和修订记录见 [`commands_and_environment.txt`](commands_and_environment.txt)；帧级旧/新结果见 [`frame_metrics.csv`](frame_metrics.csv)。

发现的是 RGB 内参主点索引错误。脚本已经按官方列优先格式读回完整内参矩阵，却把主点从 `K[2,0]`、`K[2,1]` 取出；这两个元素在标准相机矩阵中为零。正确位置是 `K[0,2]`、`K[1,2]`。样本中 RGB 主点约为 `(945,725)`，缩放到 256×192 深度网格应约为 `(126,97)`，旧代码却以 `(0,0)` 为主点。修正没有改变深度阈值、帧、网格、姿态、网格变换或扫描范围。

| 扫描 | 旧：帧中位绝对误差的中位数 [帧级 p05–p95] | 旧：帧中位误差 ≤5 cm | 旧：每帧 5 cm 内像素比例中位数 | 修正后：帧中位绝对误差的中位数 [帧级 p05–p95] | 修正后：帧中位误差 ≤5 cm | 修正后：每帧 5 cm 内像素比例中位数 [p05–p95] | 投影覆盖率中位数（旧→新） |
|---|---:|---:|---:|---:|---:|---:|---:|
| `scene_00069_00` | 0.2302 m [0.0624, 0.4374] | 4/72 | 9.51% | 0.00616 m [0.00292, 0.01400] | **72/72** | **94.45% [85.64%, 99.98%]** | 98.62% → 99.30% |
| `scene_00069_01` | 0.2347 m [0.1097, 0.4145] | 1/72 | 8.45% | 0.00745 m [0.00287, 0.01410] | **72/72** | **95.17% [85.89%, 99.89%]** | 98.50% → 99.38% |

这里的“每帧 5 cm 内像素比例”以该帧同时有有效原生深度和网格 raycast 命中的像素为分母；“投影覆盖率”是有效深度像素中有网格命中的比例。所有扫描均使用完整固定样本，没有因结果排除帧。中位有效深度比例为 100%。阈值敏感性只作误差描述：修正后每帧 2/5/10/20 cm 内像素比例的中位数分别为 `83.39/94.45/96.92/98.34%`（`_00`）和 `84.69/95.17/97.10/98.52%`（`_01`）；**通过判定只使用预定 5 cm 门槛**。

误差分布并非全部收敛在 5 cm：修正后，各扫描“逐帧像素绝对误差 p90”的中位数为 3.01 cm / 2.67 cm，p95 的中位数为 5.48 cm / 4.86 cm。按扫描 `_00/_01` 顺序，帧级 p90 的 95 分位为 7.27/9.60 cm，帧级 p95 的 95 分位为 20.22/37.66 cm；这保留了完整尾部，而没有只展示通过像素。逐帧 p50、p90、p95、5 cm 覆盖率、有效深度比例和投影覆盖率全部列在 CSV 中。

## 核验项与依据

| 项目 | 核验 | 结论 / 处理 |
|---|---|---|
| 本次失败与旧 pilot 是否相同 | `supplementary.tex`、pilot README 和原 JSON 都指向 `scene_00069_00/01`；本地两个压缩包的 revision、文件大小和 SHA256 与旧清单一致；新旧 72 个帧 ID 逐一扫描相同 | 是同一次失败样本与抽帧流程。旧基线来自 `data/multiscan_scene_00069/rgbd_projection_audit.json`；没有重新筛帧。 |
| 深度单位、缩放、无效值 | 两个元数据文件都声明 `depth_unit=m` 和 `float16_zlib`；解码为米。已有协议的有效像素规则 `finite & 0 < depth < 8 m` 未改 | 未发现单位/缩放错误。 |
| 分辨率和内参缩放 | RGB 为 1920×1440，深度为 256×192；两轴缩放都是 0.133333。官方 bridge 以 `diag(sx,sy,1) @ K_rgb` 得到深度内参 | 缩放做法正确；主点元素索引错误已修正为 `K[0,2]` / `K[1,2]`，焦距按原规则缩放。 |
| RGB、深度和位姿帧序号 | 每扫描的 RGB、depth、confidence、camera_info 元数据均为 60 Hz 且帧数相等：`_00` 为 2387，`_01` 为 2547；JSONL 位姿条数也与 RGB 数相等。位姿时间戳严格递增，间隔中位数约 16.66 ms，最大约 33.32 ms。深度流没有独立逐帧时间戳 | 没有发现固定帧数偏移证据；同序号对应由官方读取流程和匹配计数支持，但无法仅凭本地元数据独立验证每个深度采样的硬件时间戳。没有重做偏移搜索。 |
| 相机位姿布局和方向 | 官方格式说明 `transform` 为列优先 4×4、camera-to-world；官方 bridge 同样先转置/列优先还原，再右乘 `diag(1,-1,-1,1)`。本地旋转矩阵行列式约为 1 | 当前代码按列优先解释，使用 camera-to-world，并应用 ARKit→Open3D 轴转换；投影不应再求逆。未发现这项约定错误。 |
| 镜头畸变 | 这两条采集记录未提供畸变系数；官方重建 bridge 用针孔内参 | 没有可据以加入的畸变修正参数；未额外拟合畸变。残留误差可能仍含镜头/重建误差。 |
| `coordinate_transform` | 官方 output 文档说明它是列优先 4×4，用于把存储的已对齐网格变换回原重建坐标，与 JSONL 相机位姿同处一个世界坐标；当前脚本正是对网格顶点应用一次该变换。`reference_scan_alignment.transformation` 则是跨扫描对齐 | 当前变换顺序和方向符合官方说明；同扫描投影不应再应用跨扫描 `reference_scan_alignment`。此处没有改代码。 |
| 实例标注与扫描间对齐 | 本检查只将每个扫描自己的注释网格与自己的原生深度比较；不调用对象 ID、掩码或跨扫描对齐参与投影 | 这些标签不影响修正或命中判定。身份审计是旧 pilot 的独立后验记录，不作为本次几何拟合依据。 |

官方依据：MultiScan [采集格式文档](https://3dlg-hcvc.github.io/multiscan/read-the-docs/dataset/files/acquired.html)说明深度单位、RGB 内参列优先布局、相机到世界位姿和 ARKit/Open3D 轴转换；其 [bridge.py](https://github.com/smartscenes/multiscan/blob/a050c47a81dd9227e6f27823b07e2204b656fc2a/reconstruction/scripts/bridge.py#L1010-L1040) 展示了内参重排、缩放和相机外参求逆的实际位置；[输出/对齐格式文档](https://3dlg-hcvc.github.io/multiscan/read-the-docs/dataset/files/output.html)明确区分网格坐标变换与 reference-scan alignment。

## 变更与复测

- 修改 [`scripts/audit_multiscan_rgbd_projection.py`](../../../scripts/audit_multiscan_rgbd_projection.py)：增加共享的深度网格内参提取函数，使用正确主点位置；并保存每帧绝对误差 p90、p95 和 20 cm 诊断比例，便于审查完整尾部。原有采样和 2/5/10 cm 计算保持不变。
- 修改 [`scripts/diagnose_multiscan_depth_pose_sync.py`](../../../scripts/diagnose_multiscan_depth_pose_sync.py)：共用正确的内参函数，避免同步诊断脚本重复该索引错误。**没有再次运行其 ±offset sweep**；旧 pilot 的 sweep 使用了错误主点，所以不能作为可靠的时间同步结论。
- 新增 [`tests/test_multiscan_intrinsics.py`](../../../tests/test_multiscan_intrinsics.py)：测试列优先 K、深度分辨率缩放、主点位置和主点对应光轴射线。命令 `PYTHONPATH=scripts conda run --no-capture-output -n worldcache pytest -q tests/test_multiscan_intrinsics.py` 通过，`1 passed`。
- 修正后的完整投影命令、运行环境、耗时和输入哈希记录在 [`commands_and_environment.txt`](commands_and_environment.txt)；逐帧机器可读指标在 [`corrected_projection_audit.json`](corrected_projection_audit.json) 和 [`frame_metrics.csv`](frame_metrics.csv)。完整复测耗时 10.111 秒（CPU raycast），没有使用 GPU 或视觉模型。
- 一个计时尝试因系统不存在 `/usr/bin/time` 失败，随后用 Bash `time` 重跑；一次元数据读取探针因假定 ZIP 成员位于根目录而遇到 `KeyError`，改为按归档成员后缀读取后成功。输入与正式实验流程均未受影响；见命令记录。

## 是否继续

**原定 5 cm 几何预检通过，建议继续后续冻结评测实现/数据资格审查。**这是一个有明确证据的投影代码错误修复，并在原有扫描对和原有 144 个帧 ID 上复测；没有放宽阈值或选择易通过帧。

这**不等于**已完成“独立原生 RGB-D 房间重扫验证”：目前只有一个 MultiScan 房间（两次扫描），无法提供多个独立环境的不确定性估计。修正只解决同扫描网格—深度投影预检；后续正式检索仍需按冻结协议执行，且匹配阶段不得读掩码、物体 ID 或测试扫描间对齐。本文和补充材料未被修改，完整 ViT 检索未运行。
