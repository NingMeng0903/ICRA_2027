# 扫描力矩接管代码审计

审计日期 2026-09-10。只读检查代码和 68 个采集 H5，没有连接机器人、修改控制器或重写采集数据。以下路径以 `/media/camp/EXT_DRIVE/RealUS_playground` 为项目根目录，另行注明采集脚本位置。

## 结论

- 当前扫描接管明确使用补偿到 TCP 的 `My`，输出当前 tool 坐标的 `omega_y`。图像面应为 tool x-z 才能把它直接称为 in-plane 旋转；H5 没有声学横轴到 TCP 的外参，不能仅凭图像左右确定物理正负方向。
- 正常扫描并非把角度修正叠加到同一轴的位置环：TFF 把 tool z 和 tool omega_y 从路径前馈及位置反馈中移除，分别交给法向力控制和力矩导纳。
- 68 份采集的控制器配置与扫描覆盖参数一致。有效力矩参数是 `M=0.051, D=0.22, Coulomb=0.025 Nm, vmax=0.28 rad/s, amax=3 rad/s²`。
- 有限矩驱动下导纳本身就可能比较慢。线性滑动态时间常数 `M/D=0.232 s`，这不是传感器时延，也不等于实际闭环接触响应时间。
- H5 保存补偿 wrench、原始 sensor wrench、实际 TCP、力控 mode，但未保存命令 omega_y、tilt 接触门控、QP slack、deadband stop reason。不能把重建导纳输出称为实测命令，不能据此完全证明死区或 QP 冻结是某段黑图的原因。
- 接触内的近零 My 不能独立识别零偏，也不能保证完整声学耦合。需要空载静止且姿态相关的残余 wrench 基线、已知接触几何，或独立接触验证来区分。
- 采集配置中的 TDPA 是 observe 状态 `apply=false`，safety shield 是 `observe`，`energy_sign_verified=false`。本批数据不能作为已经实施完整无源约束的证据。

## 调用链与坐标

1. `/media/camp/EXT_DRIVE/ICRA_YM/script/scan_robot.py:316` 设置 `SCAN_FORCE_AXES`，在 `:323` 调用 `hfpc(reference="icra_path", law="tff", force=4.0, force_axes=...)`。
2. `peirastic/scan_path.py:18` 定义 `SCAN_FORCE_AXES=[0,0,1,0,1,0]`。`1` 表示该轴由力控制接管；TFF selection 则使用相反含义。
3. `peirastic/realman8dof/modes/track.py:239` 构建 `LegacyForceLaw`，再在 `:248` 读取 `TorqueTiltConfig`，`:255` 包装为 `LegacyForceWithTilt`。只允许 tool 控制坐标。
4. `peirastic/realman8dof/modes/track.py:132` 至 `:151` 将路径前馈和位置反馈乘 selection，再与力输出组合。未选择的 tool omega_y 位置反馈不能与 tilt 相互拉扯。另一些旋转轴和笛卡尔路径仍然可能影响实际姿态及接触，不能把所有实际转动都当成 My 导纳的结果。
5. `peirastic/realman8dof/force/torque_tilt.py:76` 强制 `axis=4`，`:296` 取 `wrench[4]`，`:297` 计算 `tau_error_y=desired[4]-wrench[4]`。当前扫描没有设置期望力矩，因此是 `-My` 驱动；测试也明确验证正 My 得负 omega_y。
6. 法向控制保持原有输出，`torque_tilt.py:441` 左右将 omega_y 写入输出 twist。正常 engaged 状态没有为旋转新增法向速度耦合补偿。异常冻结/累积角上限时存在沿最近接触法线重映射法向退让方向的路径，不能当作正常的等力旋转补偿。

补偿链：`rm75_control/rm75_control/control/joint_admittance_8dof/loop.py:6583` 调用 observer，在 `:6594` 进行 link7→TCP wrench 变换后输入 outer loop。`rm75_control/rm75_control/force/compensation/v2/frames.py:121` 将 raw sensor 符号、旋转、原点平移变换到 link7；`:141` 用 `tau_T^L=tau_L-r_LT×f_L` 后将力和矩都旋转到 TCP。它不是仅旋转力、不平移力矩。

补偿模型快照 `tool_binding` 标记 `active_tool_name=gripper2`、`force_sign=[-1,-1,-1,-1,-1,-1]`、`wrench_semantics=environment_on_tool`。TCP 平移为 `[0,-0.015229,0.121350] m`，旋转有完整记录。但标记不能代替机械功率符号实测验证，尤其力控制代码以 desired-minus-measured 驱动且配置明确没有验证 energy sign。

## 实际姿态与角速度的离线计算

`/media/camp/EXT_DRIVE/ICRA_YM/script/record.py:319` 明示 TCP 位姿为 controller FK world 下的活动 TCP，Euler 为小写 `xyz`。`:197` 附近把 state relay 的 pose 原样写入。

`rm75_control/rm75_control/control/admittance_common/state_relay.py:654` 至 `:665` 从关节和 rail 求 `kin.fk_pose`，作为发布 pose。`rm75_control/rm75_control/control/joint_admittance_8dof/model.py:261` 实际执行 SciPy `Rotation.from_matrix(...).as_euler(self.euler_order)`。`:257` 的注释称 intrinsic，但实际小写 `xyz` 应按 SciPy 行为读取，不按该注释解释。

因此离线使用 `R_i = Rotation.from_euler("xyz", pose_i[3:6])` 是正确的。区间 tool/body 角增量可用 `log(R_i^{-1} R_{i+1})`，除以真实采样时间差，时间放在两个样本的中点。避免把 Euler y 的差分当作 body omega_y。可对角速度绘图降噪，但必须记录滤波方法和额外时延；局部回归或对称离线平滑不能冒充实时可用反馈。

## 死区、动力学与门控

`peirastic/realman8dof/force/fce.py:151` 的 `kikuuwe_step` 采用隐式 Euler 质量阻尼和 Coulomb 软阈值：

`denom=M/dt+D`

`v_star=((M/dt)*omega_prev+tau_error)/denom`

`omega_target=sign(v_star)*max(abs(v_star)-Coulomb/denom,0)`，然后限速。

在 `torque_tilt.py:372` 再将速度增量限制到 `amax*dt`。这不是无记忆的 `abs(My)<deadband => 立即零速度`。已在转动时，进入死区后可能仍会减速一小段；`:370` 的 `torque_deadband` 字符串表示瞬时误差落在 Coulomb 范围，严格静止还应同时看 omega_y。

假设接触门控开放、未达到其他约束，恒定 My 的近似稳态速度：

| abs(My) | 近似稳态速度 |
| --- | --- |
| 0.025 Nm 及以下 | 从静止不能启动 |
| 0.030 Nm | 0.0227 rad/s，约 1.30°/s |
| 0.050 Nm | 0.1136 rad/s，约 6.51°/s |
| 0.0866 Nm 及以上 | 达到 0.28 rad/s，约 16.04°/s |

M/D=0.232 s 仅为导纳滑动态的线性时间常数。实际接触弹性、路径速度、内环执行滞后、输入变化以及约束会改变整个闭环响应。角加速度限幅从零到 vmax 最快也需要约 93 ms，此外仍有惯性阻尼渐近过程，不能只按 vmax 推断一个小矩修正需要多久。

接触门控：

- `torque_tilt.py:425` 左右在 normal law 更新之后读取 `controller.contact_present`。
- `rm75_control/rm75_control/control/admittance_common/controller.py:1400` 使用补偿 filtered Fz 和 unfiltered compensated raw Fz 更新 physical contact tracker。
- 记录的阈值为 enter 0.85 N，确认 20 ms；exit 0.7 N，确认 100 ms；hard enter 1.5 N。扫描路径启动另有 4 N/100 ms gate，数据裁剪亦有独立接触规则。不要把 4 N 图像录制 gate 与 tilt 持续运行的接触阈值混为一谈。
- `torque_tilt.py:332` 的前一拍 QP slack 超过 0.05 会冻结倾转；`loop.py:6626` 附近向 outer 提供 `inner.last_slack_norm`。
- 累积 tilt 角上限为 150°，每次接触上升沿归零。该角是命令 omega_y 的积分，不是独立观测的真实探头角。
- `twist_align_deg=0`，世界竖直夹角冻结关闭；`cop_stall_s=0`，CoP 停滞锁存关闭。不要把本批损失解释成默认 CoP stall。

CoP 计算 `torque_tilt.py:30` 假设接触合力作用于 tool z=0 面，`x=-My/Fz, y=Mx/Fz`。4 N 时 0.025 Nm 对应横向 6.25 mm。这个量可帮助解释为何保持较宽矩死区时允许可观的偏心受力，但不证明已脱离，也不证明完整压力分布。若合力作用平面与 TCP 有 z 偏移，切向力也贡献 My，不能直接忽略；软组织多接触区/摩擦/线缆牵拉更不能由单个 CoP 唯一反演。

## 滤波与补偿零偏

`rm75_control/rm75_control/control/admittance_common/observer.py:84` 按 poll_hz 构建 causal Butterworth；采集 controller.yaml 的 `causal_fc_hz=45, causal_order=1`，并不是另一个 `fc_hz=6` 离线配置字段代表的 6 Hz 控制滤波。

`observer.py:230` 至 `:258` 从带符号的 raw wrench 减去 phi 重力/偏置预测，再进行因果滤波。本批 `use_dynamic_kinematics=false, use_inertia=false, dynamic_kinematics_mode=off`，惯性补偿不应用。由加速度、模型误差、线缆引起的残差都可能保留在 compensated wrench，不能一概归因于接触力矩。

`observer.py:124` 的 `update_leftover` 只在空载静止时慢估计残差，接触时冻结，注释和实现均明确不自动减去；本批 H5 没有保存 leftover 向量。力矩控制中不存在根据接触中的平均 My 在线学习零点的逻辑。

68 个 controller.yaml 的 SHA-256 前 12 位相同：`2123086ed536`；force.yaml 相同：`e14199951a84`。解析为 YAML 后与当前项目文件相同，扫描 `force_overrides` 也全部相同。

phi 文件快照有四种：

| 人员 | 份数 | phi 文件 SHA-256 前 12 位 | phi_recommended 质量 |
| --- | --- | --- | --- |
| yameng_1 | 12 | c4cf4975bb63 | 0.574736 kg |
| yameng_2、dianye | 24 | 5dc72bc4c225 | 0.561681 kg |
| chenwei、zhongyao | 20 | b48f9245bde4 | 0.581856 kg |
| zhongkai | 12 | 654c3d57176f | 0.586889 kg |

所以 chenwei 与 zhongyao 的差异不能直接归因于配置文件或文件快照中的模型版本差异。另一方面文件快照只是 record.py:329 至 :337 读取磁盘的文本，不是 observer 运行时模型 revision。`observer.py:152` 的 reload 为显式维护操作，非自动每拍重载；H5 未记录实际加载 revision，因此不把文件快照说成已经独立验证的运行时参数。

## 诊断记录缺口与证据边界

`/media/camp/EXT_DRIVE/ICRA_YM/script/record.py:222` 仅向 force 组写入 compensated wrench、contact_force、mode_valid、mode、force_control_active 和时标。`force_control_active` 是 TRACK_HYBRID 状态有效性的近似标签，不是 physical contact、tilt engaged 或真实命令输出标签。

完整 tilt 诊断实际已存在 outer.telemetry，`peirastic/realman8dof/modes/track.py:153` 至 `:166`。可选 CSV 在 `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py:3978` 定义字段，`:4809` 读取命令/死区/门控，`:5448` 写入。包括 `tilt_tau_y_nm, tilt_tau_error_y_nm, tilt_omega_y_rad_s, tilt_theta_rad, tilt_engaged, tilt_frozen, tilt_capped, tilt_stalled, tilt_stop_reason, tilt_cop_x_m, tilt_cop_r_m, tilt_on_tube, tilt_deadband_nm`，还可记录 QP 与实际速度诊断。

`peirastic/realman8dof/daemon.py:53` 的 `resolve_log_csv(None)` 默认关闭，显式 `--log-csv` 才开启。搜索默认 `rm75_control/apps/logs/peirastic` 并包含被 gitignore 的文件，没有匹配 2026-09-09 本批扫描的 CSV。最新文件 `run_20260908_034408.csv` 运行 45.38 s，不覆盖本批时间；其它新近日志也只持续十几到几十秒。该搜索不能排除用户在未给出的其它自定义路径保存了日志。

现有数据可检验：

- 黑暗/边缘回声丢失是否与持续偏心 My、正常 Fz、实际 omega_y 同时出现。
- My 穿越 0.025 Nm 前后实际角速度是否开始/减弱；这只是与死区假设一致的证据。
- 同样局部 My 下正常和失耦图像能否并存，从而证明 My 不是充分的声学质量指标。
- 高角速度/扭矩瞬态是否领先或滞后图像事件，使用校准后的时间并报告约 30 Hz 图像采样与有效时延模型的不确定性。

现有数据不能单独证明：

- 命令产生得晚，还是已发命令但内环跟不上。
- 某段是死区、QP slack freeze、执行器限制还是其它门控导致。
- 接触内 My 的非零来自标定零漂、真实压力偏心、切向力的杠杆矩或线缆/动态残差中的哪一项。
- 图像黑一定是失去机械接触；组织各向异性、骨后声影与切面变化需要图像空间形态和人工核验区分。
- 压力分布的唯一形状，以及已满足完整六维端口的无源性。

## 与后续 QP 和无源设计有关的当前事实

`peirastic/configs/force.yaml:132` 的 TDPA enabled=true 但 apply=false；`:151` safety_shield mode=observe；`:155` energy_sign_verified=false；`:106` force_barrier enabled=false。可以将这些视为研究/诊断基础，不能把尚未应用的观察量写成执行器上的安全保证。

正常 tilt 对 force law 输出的组合只增加 omega_y，没有显式加入旋转引起的 Fz 变化补偿或联立接触模型。未来若增加旋转/法向联合 QP，应保持基线 normal law 与 torque law 作为 nominal，记录 final executed twist，并把完整 wrench·twist 的符号、时标、执行误差及参考主动注能一并纳入能量账本。恒定总 Fz 本身不意味着旋转无机械功率；从 My/Fz 算出的压力中心也不能随意作为消除真实接触力矩的偏置。
