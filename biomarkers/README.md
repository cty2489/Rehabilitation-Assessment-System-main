# 02_biomarkers — 临床可解释生物标志物提取

从 `BJH/` 三模态信号（EEG / EMG / IMU）为单个病例提取一组**有生理意义、可命名**的
生物标志物，并随病例临床标签（FMA-UE / BI / hand_tone(MAS) / hand_function(Brunnstrom)）一并输出。

> ⚠️ **说明**：真实受试者仅 **5 名（S1–S5）**，其余 **S6–S15 为模拟数据**。本工具产出的是
> **可解释的信号量**，不做统计验证、更非临床诊断。

## 适用范围

- 支持全部受试者 **S1–S15**，由 `analysis/common/manifest.py` 扫描 `BJH/EEG_new/` 枚举：
  - **真实 S1–S5**：EEG 为 `.bdf`（27 个 trial），走 `load_eeg_bdf` 预处理链。
  - **模拟 S6–S15**：EEG 为 `.csv`（54 个 trial），走 `load_eeg_csv_bdf_equiv`——
    与 BDF **同链但不 z-score**（bandpass 1.5–50 + notch 50 + 平均参考 + 重采样 500Hz），
    保证脑电生物标志物与真实受试者可比。
  - EMG/IMU 对所有人均为 `.csv`（4 肌 Delsys 格式）。
- **临床标签统一取自** `patient_rehab_suggestions_15subjects.json` 的 `labels`（覆盖 S1–S15）。
  注：S1 的 BI 在此文件为 65（旧 `bjh_labels.json` 为 60）。
- 每个 trial = 一个主动抓握类动作的连续记录（task 1 抓握 / 2 勾状抓握 / 4 捏笔 / 5 拇指内收 /
  6 握圆筒 / 7 握球；S1 仅 4 试次）。所有动作均为主动完成。

## 生物标志物（26 项）

### EEG（皮层）
| 名称 | 含义 |
|---|---|
| `pathological_asymmetry_index` (PAI) | 受损 vs 健侧运动皮层 μ/β 功率不对称 |
| `corticomuscular_coherence_beta` (CMC) | 受损半球–患手主动肌 β 带相干 |
| `prefrontal_theta_beta_ratio` | 前额叶 (Fp1/Fp2/Fz) θ(4–8)/β(13–30) 功率比 |
| `interhemispheric_motor_coherence` | 健-患侧运动皮层 (C3 簇↔C4 簇) β 带相干 |
| `movement_mu_power_change` | 运动相关 μ(8–12) 功率变化（高活动 vs 低活动窗，去同步为负） |
| `movement_beta_power_change` | 运动相关 β(13–30) 功率变化（高活动 vs 低活动窗，反弹为正） |

### EMG（外周 / 肌肉）
| 名称 | 含义 |
|---|---|
| `resting_emg_level` | 患手屈肌静息肌电 RMS（绝对幅值，肌张力代理） |
| `wrist_co_contraction_index` (CCI-腕) | 腕屈肌 (FCR)/腕伸肌 (ECU) 共收缩包络重叠 |
| `finger_co_contraction_index` (CCI-指) | 指浅屈肌 (FDS)/指伸肌共收缩包络重叠 |
| `emg_activation_rms` | 全段自主激活幅度 RMS |
| `fcr_iemg` | 桡侧腕屈肌 (FCR) 积分肌电 IEMG（∫\|EMG\|dt，V·s） |
| `fds_iemg` | 指浅屈肌 (FDS, 取自掌长肌电极位) 积分肌电 IEMG |
| `ecu_iemg` | 尺侧腕伸肌 (ECU) 积分肌电 IEMG |
| `extensor_digitorum_iemg` | 指伸肌 (Extensor Digitorum) 积分肌电 IEMG |
| `flexor_extensor_iemg_ratio` | 屈伸肌 IEMG 比 = Σ屈肌 / Σ伸肌 |
| `emg_burst_duration` | 肌电爆发平均持续时间（FCR 包络阈值检测，秒） |
| `fcr_mdf` | 桡侧腕屈肌 (FCR) 中位频率 MDF（Hz，疲劳代理） |
| `fds_mdf` | 指浅屈肌 (FDS) 中位频率 MDF（Hz，疲劳代理） |
| `ecu_mdf` | 尺侧腕伸肌 (ECU) 中位频率 MDF（Hz，疲劳代理） |
| `extensor_digitorum_mdf` | 指伸肌中位频率 MDF（Hz，疲劳代理） |

### IMU（运动学）
| 名称 | 含义 |
|---|---|
| `movement_smoothness_sparc` | 谱弧长运动平滑度 |
| `range_of_motion_proxy` | 4 传感器陀螺角速度范围均值 (p98-p2) |
| `tremor_index_3_6hz` | 3–6 Hz 加速度相对功率 |
| `wrist_flexion_peak_velocity` | 腕屈方向峰值角速度：ECU 处传感器主轴去偏置后负向 \|p5\|（deg/s） |
| `wrist_extension_peak_velocity` | 腕伸方向峰值角速度：ECU 处传感器主轴去偏置后正向 p95（deg/s） |
| `finger_extension_peak_velocity` | 伸指峰值角速度：指伸肌处传感器陀螺幅值 p95 |

公式与依据见 `biomarkers.py` 注释与 `paper/Method.md` 第 III-B 节。
**患侧映射**：受损半球 = 患手对侧（患手 R → 左皮层 C3 簇）。代码读取 `affected_side` 字段，双向通用。

## 用法

```bash
# 单病例（真实或模拟，默认跨该受试者所有 trial 聚合）：打印中文报告，写 JSON
python analysis/02_biomarkers/extract_biomarkers.py --subject S1
python analysis/02_biomarkers/extract_biomarkers.py --subject S6   # 模拟受试者

# 单个 trial
python analysis/02_biomarkers/extract_biomarkers.py --subject S1 --trial 2_1

# 额外输出每个 trial 的原始值
python analysis/02_biomarkers/extract_biomarkers.py --subject S1 --all-trials

# 生成跨受试者汇总 CSV（S1–S15 共 15 行，可直接 Excel 打开）
python analysis/02_biomarkers/extract_biomarkers.py --cohort
```

## 输出（默认 `analysis/02_biomarkers/out/`）

- `json/S1.json` — 机器可读：每个标志物含 `value` 与 `n_valid`（有效 trial 数），外加 labels / demographics / cohort。
- `reports/S1.txt` — 中文人类可读报告（同时打印到屏幕）：生物标志物 | 值 | 有效 trial 数。
- `csv/cohort_biomarkers.csv` — 每受试者一行，标签 + 各标志物 `__value` 列。

## 参考范围（文献依据）

`out/biomarker_reference_ranges.json` 为 18 项标志物各给一条参考范围，分层优先
**Brunnstrom**，无分层依据时给**健康成人常模**。由 `build_reference_ranges.py` 生成
（程序化遍历 `BIOMARKER_NAMES`，保证全覆盖、键名一致，并自校验 `source↔references`）：

```bash
python analysis/02_biomarkers/build_reference_ranges.py
```

- 每项含 `reference_type`：`healthy_norm`（有健康常模数值，如 SPARC、共收缩指数）/
  `directional_trend`（仅方向性，如 CMC、前额叶 θ/β、PAI、震颤）/ `none`（设备/协议特异量，
  文献无标准范围——多数 EMG 绝对量、IEMG、IMU 陀螺量、运动相关功率变化）。
- **凡有真实文献依据的项**，`source` 列出引用 id，对应顶层 `references` 的完整出处（含 URL/PMCID）。
- ⚠️ 文献无逐 Brunnstrom 期阈值时 `by_brunnstrom=null`，用 `expected_direction_with_recovery`
  表达随康复升期的预期方向。本模块若干量纲（原始 EMG 电压、加速度 SPARC、IMU deg/s）与
  文献测量方式不同，文献值不可在绝对尺度直接套用，仅供方向参考。**绝非临床诊断阈值。**

## 注意事项（数据局限，客观说明）

- **必须用 raw 信号**：`load_trial_raw` 保留 EMG/IMU 绝对幅值（肌张力/震颤/IEMG 所需）；
  `load_trial` 的鲁棒 z-score 会抹掉它。
- **EMG 4 通道与肌肉口径**：右/患侧为 FCR / 指浅屈肌 (FDS) / ECU / 指伸肌 4 通道；其中
  **指浅屈肌 (FDS) 信号取自掌长肌 (Palmaris Longus) 电极位**，按临床约定统一命名/使用为 FDS。
  据此腕屈/伸 (FCR vs ECU) 与指屈/伸 (FDS vs 指伸肌) 的共收缩指数分别计算，四块肌肉各给出 IEMG 与中位频率 MDF。
- **运动相关 μ/β 功率变化**：基于 EMG 包络划分高/低活动窗（非事件触发标记）；EEG 与 EMG 时钟独立、
  未跨模态精同步，故其与 CMC、半球间相干的*绝对值*仅供队列内排名参考。
- **腕屈/伸方向角速度 / 伸指速度**：基于陀螺仪角速度（非关节角度测量），所有动作均为主动完成；
  腕屈/伸方向取 ECU 处传感器方差最大轴、高通去安装方向偏置后的带符号角速度正/负向峰值。
- **中位频率 MDF**：各肌 Welch 谱 20–450 Hz 限带累计 50% 功率的频率，反映肌肉疲劳趋势；受电极位置/皮下脂肪影响，绝对值仅队列内做方向比较。
- **mne 缺失时**降级：EMG/IMU 标志物仍可计算，EEG 相关项为 NaN（聚合时 `n_valid` 计为 0）。
- **真实 vs 模拟不可在绝对尺度上混比**：模拟受试者 S6–S15 的 IMU 陀螺幅值比真实数据小约 2–3 个
  数量级（模拟器生成所致），因此 `range_of_motion_proxy` / `wrist_flexion_peak_velocity` /
  `wrist_extension_peak_velocity` / `finger_extension_peak_velocity` 等 IMU 绝对量在真实与模拟之间
  **不可直接比较**，宜各自组内比较。EEG（经同链处理）与 EMG IEMG 量级则大体可比；
  震颤、CCI、MDF 等相对/频率量跨组可作方向比较。
