"""Prompt templates for the rehab-text LLM.

Both training (data_builder.py) and inference (generate.py) build chat
messages here so that the prompt format stays in one place.
"""
from __future__ import annotations

from typing import Dict, List


REPORT_TEMPLATE = (
    "患者S{sid}，{gender}性，{age}岁，{disease}，病程{days_post}天，{side_zh}侧偏瘫。"
    "当前FMA手{FMA_UE}分，提示{fma_interp}；"
    "BI评分{BI}分，{bi_interp}。"
    "手部肌张力{hand_tone}级，手分期为Brunnstrom {hand_function}期，"
    "表明{brunn_interp}，可完成{brunn_action}，但{brunn_limit}。"
    "建议{rehab_action}，结合{rehab_modality}，提升手部{rehab_goal}。"
)


SYSTEM_PROMPT = (
    "你是康复医学辅助助手。根据患者的人口学信息和四项临床评估指标"
    "（FMA手部分数、Barthel指数、手部肌张力、Brunnstrom手分期），"
    "输出一段中文康复评估与建议。\n"
    "必须严格使用以下句法骨架，仅替换 {} 内的槽位，不增删句子、不改变标点：\n"
    f"{REPORT_TEMPLATE}\n"
    "硬性要求：必须保留输入数值原样（FMA、BI、肌张力、Brunnstrom 期数）；"
    "病程统一以「天」为单位；用语正式、贴合临床康复师风格。"
)


_SIDE_ZH = {"L": "左", "R": "右", "左": "左", "右": "右"}


def build_user_message(
    subject_id: str | int,
    demographics: Dict[str, object],
    labels: Dict[str, object],
) -> str:
    """Render the user turn as a fixed-field clinical brief."""
    side_raw = str(demographics.get("affected_side", "R"))
    side_zh = _SIDE_ZH.get(side_raw, side_raw)
    return (
        f"患者编号: S{subject_id}\n"
        f"性别: {demographics.get('gender', '')}    "
        f"年龄: {int(demographics.get('age', 0))}岁\n"
        f"诊断: {demographics.get('disease', '')}    "
        f"病程: {int(demographics.get('days_post', 0))}天    "
        f"偏瘫侧: {side_zh}\n"
        f"FMA手部分数: {int(labels['FMA_UE'])}/20\n"
        f"Barthel指数(BI): {int(labels['BI'])}/100\n"
        f"手部肌张力分级: {labels['hand_tone']}\n"
        f"Brunnstrom手分期: {int(labels['hand_function'])}\n"
        f"请生成康复评估与建议。"
    )


def build_chat_messages(
    subject_id: str | int,
    demographics: Dict[str, object],
    labels: Dict[str, object],
    rehab_text: str | None = None,
) -> List[Dict[str, str]]:
    """Build the chat-format message list.

    If `rehab_text` is given, the assistant turn is appended for SFT.
    Leave `rehab_text=None` at inference time.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(subject_id, demographics, labels)},
    ]
    if rehab_text is not None:
        messages.append({"role": "assistant", "content": rehab_text})
    return messages


# --------------------------------------------------------------------------- #
# Structured clinical-reasoning prompt (full multi-section report).            #
#                                                                             #
# Unlike the legacy one-paragraph skeleton above, this asks the model to act   #
# as a rehab physician over ALL numbers (4 indicators + every digital          #
# biomarker + its per-stage reference range + the 26-gesture library) and      #
# return a STRUCTURED JSON of clinical text only — interpretations, treatment  #
# advice, subtype, strategy, gesture plan + dosing, weekly plan, warnings.     #
# The caller back-fills this text into a fixed numeric skeleton, so the model  #
# never owns any measured value (anti-tampering). Works WITHOUT fine-tuning    #
# (constrained structured reasoning); fine-tuned weights simply improve it.    #
# --------------------------------------------------------------------------- #
import json as _json


CLINICAL_SYSTEM_PROMPT = (
    "你是一名资深康复医学医师。下面给你一名脑卒中偏瘫患者的多模态评估数值：四项临床"
    "评估指标、26 项数字生物标志物（含其文献参考范围）。\n"
    "重要：26 项生物标志物的参考范围分三类——①少数项（如运动平滑度 SPARC）有【健康常模"
    "区间】；②部分项（如皮层-肌肉相干、半球间相干、半球不对称指数、前额叶 θ/β、震颤指数）"
    "文献仅给【恢复方向（↑/↓）】而无绝对阈值；③多数项（原始 EMG 电压/RMS、四块肌 IEMG、"
    "腕/指共收缩指数 CCI、四块肌中位频率 MDF、IMU 陀螺角速度、运动相关功率变化）为本研究"
    "设备/协议特异量，文献【无标准参考范围】，且部分量真实与模拟数据相差数量级，其绝对值"
    "仅供方向/队列内排名参考。\n"
    "请你像医师一样【读数→判断→开方】，针对【这名患者的真实数值】给出：每个生物标志物的"
    "解读与治疗建议、各模态亚型界定、综合亚型界定、治疗策略要点、预警与特殊建议、下次评估"
    "时间。\n"
    "\n"
    "★ 防套模板（最高优先级）：本提示中所有 `<…>` 占位符、写作步骤与括号内说明【仅为格式"
    "示意】，严禁原样照抄或复述；你输出的每一句话都必须依据上方该患者的真实数值推导，换一"
    "名数值不同的患者其输出必须明显不同。严禁产出与具体数值无关的通用模板话术。\n"
    "★ 分期接地：分期判断必须严格等于输入中给定的 Brunnstrom 手分期（payload 的 stage / "
    "stage_roman）。overall_subtype 与 group_subtypes 各模态亚型的分期前缀【必须】写成"
    "「{stage_roman}期-…」，不得臆造其它分期；若客观分期较高（如已达较晚期），临床解读与"
    "亚型也必须与之一致，不得描述成早期重症。\n"
    "\n"
    "硬性约束：\n"
    "1) 严禁修改任何给定的数值与参考范围，只产出文字判断与方案；\n"
    "2) 治疗策略必须与「总体分期 + 各生物标志物相对其参考范围或恢复方向」一致"
    "（屈/伸肌 IEMG 比偏大=屈肌主导→加大伸指比例；腕/指共收缩指数偏高→避免快速抓放并先"
    "牵伸；中枢驱动不足（皮层-肌肉/半球间相干低、半球不对称指数大）→强化运动想象与镜像反馈）；\n"
    "3) 必须严格输出符合给定 JSON Schema 的【单个 JSON 对象】，不要输出多余文字、不要"
    "使用代码块标记。\n"
    "\n"
    "写作要求（务必逐项满足）：\n"
    "A) marker_text：逐个生物标志物给【各不相同】的 interpretation 与 treatment_advice，"
    "二者都要与该标志物当前值相对其参考范围/恢复方向强绑定；【严禁】所有标志物套用同一"
    "句话术。对于①有健康常模的项，按高于/处于/低于常模解读；对于②③【无标准参考范围】的"
    "项，按以下【写作步骤】组织（步骤本身不要照抄成文字）：先点明该指标当前是偏高/偏低或"
    "处于何方向 → 声明「文献无标准参考范围（设备特异量）」→ 给出康复期望方向（↑或↓）→ "
    "结合相关肌群/相邻模态做组合解释 → 承认其不确定性 → 给出后续随访/训练建议；不得对无阈"
    "值的项硬下「超标/正常」结论。\n"
    "B) overall_subtype：必须是含五要素的一句话，按此骨架填空（占位符替换为基于本患者数值"
    "的判断，勿照抄）：「<stage_roman>期-<优势运动模式>伴<中枢驱动特征>亚型，<协同分离程度>，"
    "<关节活动度状态>」。\n"
    "C) group_subtypes：emg/eeg（imu 可选）各给一句亚型界定，分期前缀同样写「<stage_roman>"
    "期-…」，内容须与对应模态的标志物表现一致。\n"
    "D) treatment_strategy：每条要点必须覆盖六维度——①策略名称 ②具体方法（健侧/患侧/设备"
    "如何配合）③训练剂量（时间/频次/占比/辅助力度）④反馈标准（可量化阈值及奖励方式）"
    "⑤调整原则（需减少/替换/避免的训练）⑥安全注意（单次时长/疲劳/分次安排），写成一句"
    "富信息描述，且与本患者的标志物表现对应。\n"
    "E) next_assessment 固定为：「7天后执行下一次居家评估。」\n"
)


# Gesture-section instructions, appended to the system prompt ONLY when the
# clinical team's 26-gesture library is ready (gestures.library_ready()). While
# not ready we neither ask for nor accept a gesture plan, so an un-fine-tuned
# base model can't invent gesture names.
_CLINICAL_GESTURE_INSTRUCTIONS = (
    "F) gesture_plan：只能从给定的「候选手势库」中选取（不得自创手势名），至少 6 个；每个"
    "给出 purpose/force/reps，且选取理由须与本患者分期与标志物一致；不要输出任何「不推荐"
    "手势」字段。\n"
    "G) weekly_plan：必须覆盖周一至周日共 7 天，每天一条；每个训练日的训练内容只能取自上面"
    "gesture_plan 推荐的手势（周日可安排休息或自由训练）；时长合理（训练日约 20–30 分钟）。\n"
)

_CLINICAL_NO_GESTURE_NOTE = (
    "注：本次未提供康复手势库，【不要】输出 gesture_plan 与 weekly_plan 字段（手势组合与"
    "周训练计划将待手势库补充后再生成）。\n"
)


def build_clinical_reasoning_messages(context: Dict[str, object]) -> List[Dict[str, str]]:
    """Build chat messages for the structured clinical-reasoning report.

    ``context`` is the deterministic report context from
    ``backend/report_builder.build_context`` (patient, predictions, the biomarker
    groups with literature ref ranges) plus the JSON schema hint under key
    ``schema_hint``. To prevent an un-fine-tuned base model from parroting a fixed
    template, there is NO few-shot example — only system instructions + the real
    patient payload (``[system, user]``). The model must reply with a single JSON
    object matching the schema.

    Gesture fields are requested only when ``context["gesture_ready"]`` is truthy
    (the clinical team's 26-gesture library is available); otherwise the gesture
    library is omitted and the model is told not to emit gesture_plan/weekly_plan.
    """
    schema_hint = str(context.get("schema_hint", ""))
    gesture_ready = bool(context.get("gesture_ready"))

    system = CLINICAL_SYSTEM_PROMPT + (
        _CLINICAL_GESTURE_INSTRUCTIONS if gesture_ready else _CLINICAL_NO_GESTURE_NOTE
    )
    # Make stage_roman explicit so the model uses the exact 分期 prefix.
    system = system.replace("{stage_roman}", str(context.get("stage_roman", "")))

    # Only ask the model to interpret markers that were actually measured. Device
    # bundles can't compute every marker; unavailable ones are dropped from the
    # payload (and back-filled with a "数据不足" note during rendering) so the
    # model never fabricates a reading for a missing value.
    biomarkers_in = context.get("biomarkers") or {}
    n_dropped = 0
    if isinstance(biomarkers_in, dict) and biomarkers_in.get("groups"):
        filtered_groups = []
        for g in biomarkers_in["groups"]:
            kept = [m for m in g.get("markers", []) if m.get("available", True)]
            n_dropped += len(g.get("markers", [])) - len(kept)
            if kept:
                filtered_groups.append({**g, "markers": kept})
        biomarkers_payload = {**biomarkers_in, "groups": filtered_groups}
    else:
        biomarkers_payload = biomarkers_in

    payload = {
        "patient": context.get("patient"),
        "predictions": context.get("predictions"),
        "stage": context.get("stage"),
        "stage_roman": context.get("stage_roman"),
        "biomarkers": biomarkers_payload,
    }
    if gesture_ready:
        payload["gesture_library"] = context.get("gesture_library")

    dropped_note = (
        f"\n注：本次仅纳入可计算的生物标志物，已自动剔除 {n_dropped} 项数据不足/该采集格式"
        "暂不支持的标志物；marker_text 只需覆盖上方 biomarkers 中实际列出的标志物，"
        "缺失项无需解读。\n"
        if n_dropped else ""
    )

    user = (
        "【患者与评估数值（JSON）】\n"
        # Compact (no indent/whitespace) to save input tokens — the 26
        # biomarkers payload is the bulk of the prompt.
        + _json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + f"\n\n本患者 Brunnstrom 手分期为第 {context.get('stage_roman', '')} 期，"
        + "所有亚型界定的分期前缀必须与之一致。\n"
        + dropped_note
        + "\n【必须返回的 JSON Schema】\n"
        + schema_hint
        + "\n\n请只返回一个符合上述 Schema 的 JSON 对象，且全部文字基于上方该患者的真实数值。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


__all__ = [
    "REPORT_TEMPLATE",
    "SYSTEM_PROMPT",
    "build_user_message",
    "build_chat_messages",
    "CLINICAL_SYSTEM_PROMPT",
    "build_clinical_reasoning_messages",
]
