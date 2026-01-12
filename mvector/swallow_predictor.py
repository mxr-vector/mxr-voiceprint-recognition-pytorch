# ============================================================
# 改进版二合一吞音检测系统（生产级方案2）
# 归一化 + 分段非线性 + 可选 ADMM 优化
# ============================================================
import torch
import torchaudio.functional as F
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pypinyin import pinyin, Style
from mvector.utils.audio_utils import load_audio_segment
from yeaudio.audio import AudioSegment

# ============================================================
# 全局配置（已添加详细注释）
# ============================================================
TOKEN_LANGUAGE = [
    "chinese",
    "japanese",
    "korean",
    "zh-cn",
    "ja-jp",
    "ko-kr",
    "zh_cn",
    "ja_jp",
    "ko_kr",
]

PHONEME_LANGUAGE_DICT = {
    "english": "en-us",
    "en-us": "en-us",
}


# 采样率：所有音频在处理前会被重采样到该采样率（Hz）
SAMPLE_RATE = 16_000

# 每帧时长（毫秒），决定特征帧的 hop 长度和 CTC 帧率映射
FRAME_MS = 20.0

# CTC 模型中的 blank token ID（通常为 0），用于分割静音/填充
BLANK_ID = 0

# 合并非常短段的阈值（毫秒），小于该时长的段会合并到邻近段
MERGE_SHORT_MS = 10

# ------- 权重（各特征在最终 S2 评分中的线性权重，随后会归一化） -------
"""
S2 惩罚项
duration: 惩罚音素过短的情况，吞音或吐字不清通常表现为音素被压缩，时长短于正常音素
bank: 惩罚静音片段，通常为静音片段被误判为音素
posterior: 惩罚模型对音素的低置信度
energy: 惩罚模型对音素的能量过低
entropy: 惩罚音素频谱不集中（模糊、不确定）
voicing: 惩罚低发声率音素
zcr: 惩罚频率变化过快的音素
spectral: 惩罚频谱偏高或偏低的音素
"""
# 关键特征加权
W_DURATION = 4.0
W_BLANK = 3.0
W_POSTERIOR = 2.0
W_ENERGY = 2.5
# 可选其他特征小权重
W_ENTROPY = 1.0
W_VOICING = 1.0
W_ZCR = 0.2
W_SPECTRAL = 0.5


# ------ 阈值（用于判断各个特征是否触发“原因”提示） ------
# 时长短的判定阈值（归一化后判断）
THRESH_DURATION = 0.7
# 空白比例阈值（大于认为空白过多）
THRESH_BLANK = 0.1
# 后验概率阈值（低于则认为模型置信度不足）
THRESH_POSTERIOR = 0.5
# 能量归一化阈值（高于代表能量过低）
THRESH_ENERGY = 0.4
# 谱熵阈值（高于认为模糊发音）
THRESH_ENTROPY = 0.2
# 发声率阈值（低于认为低发声率）
THRESH_VOICING = 0.3
# 基频突变判定阈值（大于认为基频剧烈变化）
THRESH_PITCH_DROP = 0.5
# 过零率阈值
THRESH_ZCR = 0.4

# ---------- 权重归一化（内部逻辑保持不变，但添加注释） ----------
# 将上面定义的权重按总和归一化，确保它们之和为 1，避免尺度问题
total = sum(
    [
        W_DURATION,
        W_BLANK,
        W_POSTERIOR,
        W_ENERGY,
        W_ENTROPY,
        W_VOICING,
        W_ZCR,
        W_SPECTRAL,
    ]
)
W_DURATION /= total
W_BLANK /= total
W_POSTERIOR /= total
W_ENERGY /= total
W_ENTROPY /= total
W_VOICING /= total
W_ZCR /= total
W_SPECTRAL /= total


# ===============================
# 核心检测类
# ===============================
class SwallowPredictor:
    """
    吞音检测器主类，基于 wav2vec2 的 CTC 输出结合多模态特征（能量、谱、基频等）计算每个音素的疑似吞音评分。
    参数：
      - use_admm: bool，是否启用可选的 ADMM 优化流程（当前示例中保留接口，未实现特殊流程）
    主要方法：
      - forward(wav_path): 读取音频并得到模型 logits
      - analyze(wav_path, reference_text=None): 对音频进行完整分析并返回句级与音素级结果
      - ctc_segments(logits): 将连续帧按最大后验 token 切分为段
      - score_s2(...): 计算改进后的 S2（基于多模态特征的吞音风险分数）
      - aggregate(phonemes): 聚合音素分数为句级分数
    """

    def __init__(
        self,
        language="zh-cn",
        token_model_path="jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        phoneme_model_path="facebook/wav2vec2-large-960h-lv60-self",
        risk_threshold=0.5,
        severe_threshold=0.7,
        use_gpu=True,
        use_admm=False,
    ):
        """
        初始化检测器，加载 processor 与模型到 DEVICE 并设置为推理模式。
        参数：
          - use_admm: 是否在后续评分中启用 ADMM 优化（布尔）
        注意：
          - 模型加载较慢且占显存，请在初始化时做好资源管理。
        """
        language = language.lower()
        model_path = ""
        if language in TOKEN_LANGUAGE:
            model_path = token_model_path
        elif PHONEME_LANGUAGE_DICT.get(language) is not None:
            model_path = phoneme_model_path
            language = PHONEME_LANGUAGE_DICT[language]
        else:
            raise ValueError(f"不支持的语言: {language}")

        # 推理设备，优先 GPU，否则使用 CPU
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        # 全句级风险阈值（用于最终句子级别风险分层）
        self.risk_threshold = risk_threshold
        self.severe_threshold = severe_threshold
        # 加载 processor 与模型
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.language = language
        # TODO 后续扩展s2方案无参考文本的 admm惩罚项优化
        self.use_admm = use_admm

    # ---------- forward ----------
    def forward(self, audio_segment: AudioSegment) -> tuple:
        """
        前向推理接口：从文件加载音频并通过模型得到 logits（不进行 softmax）。
        参数：
          - audio_segment: audio 文件路径（被 librosa 加载并重采样到 SAMPLE_RATE）
        返回：
          - wav: numpy 一维数组（原始音频波形）
          - logits: torch.Tensor，shape (T, C) 的模型输出 logits（未归一化）
        注意：
          - 返回的 logits 仍在 DEVICE（cuda/CPU），调用方在分析时会在 CPU 上计算进一步指标。
        """
        # 转成 np.float32，保证后续处理兼容
        wav = audio_segment.samples.astype(np.float32)
        inputs = self.processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits
        return wav, logits[0]

    # ---------- CTC segmentation ----------
    def ctc_segments(self, logits: torch.Tensor):
        """
        基于 logits 的逐帧最大后验 token 划分连续段，并合并过短段。
        参数：
          - logits: torch.Tensor 或 numpy array，模型在每帧的 logits
        返回：
          - segments: 列表，每项为 (pid, start_frame, end_frame, probs_segment)
            - pid: token id（int）
            - start_frame, end_frame: 帧索引（半开区间 end）
            - probs_segment: numpy array，段内每帧 softmax 概率
        说明：
          - 合并时会将小于 MERGE_SHORT_MS 的段与前一段合并，减少短时抖动。
        """
        log_probs = torch.log_softmax(logits, dim=-1).cpu()
        probs = log_probs.exp().numpy()
        ids = np.argmax(probs, axis=-1)
        segments, start, prev = [], 0, ids[0]
        for i, pid in enumerate(ids):
            if pid != prev:
                segments.append((prev, start, i, probs[start:i]))
                prev, start = pid, i
        segments.append((prev, start, len(ids), probs[start:]))

        # 合并过短段
        merged = []
        for pid, start, end, pseg in segments:
            duration_ms = (end - start) * FRAME_MS
            if merged and duration_ms < MERGE_SHORT_MS:
                last_pid, last_start, last_end, last_pseg = merged.pop()
                merged.append(
                    (
                        last_pid,
                        last_start,
                        end,
                        np.concatenate([last_pseg, pseg], axis=0),
                    )
                )
            else:
                merged.append((pid, start, end, pseg))
        return merged

    # ---------- 改进 S2 评分 ----------
    def score_s2(
        self,
        pid: int,
        start: float,
        end: float,
        probs_segment: np.ndarray,
        rms: np.ndarray,
        rms_mean: float,
        avg_duration_ms: float,
        zcr=None,
        centroid=None,
        bandwidth=None,
        f0=None,
    ):
        """
        计算改进的 S2（分段吞音风险评分），结合时长、后验、空白率、能量、谱熵、发声率、ZCR、谱质心等特征。
        参数：
          - pid, start, end: 段信息（token id, 帧起, 帧止）
          - probs_segment: 段内每帧模型概率分布（numpy array，shape (L, C)）
          - rms: 每帧能量（numpy array）
          - rms_mean: 全句平均能量
          - avg_duration_ms: 非空白音素平均时长（毫秒）
          - zcr, centroid, bandwidth: 对应帧级谱/时域特征（可选）
          - f0: 基频序列（帧级，带 NaN 表示非音高帧）
        返回：
          - S2: 归一化后的分数（0-1，值越大表示越可能吞音）
          - reasons: 原因列表（字符串），对应触发的特征异常项
        设计说明：
          - 先做线性加权得到中间量，再通过分段非线性映射（exponential）放大高风险区域。
          - 各归一化分量使用 clamp 保证稳定性。
        """
        duration_ms = (end - start) * FRAME_MS
        posterior = (
            float(probs_segment[:, pid].mean()) if pid < probs_segment.shape[1] else 0.0
        )
        blank_ratio = float((probs_segment[:, BLANK_ID] > 0.6).mean())
        energy = (
            float(rms[start:end].mean())
            if start < len(rms) and end <= len(rms)
            else 0.0
        )

        # 归一化特征
        short_threshold = max(FRAME_MS, avg_duration_ms * 0.5)
        D_short = clamp((short_threshold - duration_ms) / short_threshold)
        posterior_norm = 1 - posterior
        energy_norm = clamp((0.95 * rms_mean - energy) / (0.8 * rms_mean + 1e-8))
        try:
            p = np.clip(probs_segment, 1e-12, 1.0)
            entropy_frames = -np.sum(p * np.log(p), axis=1)
            posterior_entropy = float(np.mean(entropy_frames))
            max_ent = np.log(probs_segment.shape[1])
            H_norm = clamp(posterior_entropy / (max_ent + 1e-8))
        except Exception:
            H_norm = 0.0

        # 发声率和基频
        voicing_ratio, pitch_drop = 0.0, 0.0
        if f0 is not None and start < len(f0):
            seg_f0 = f0[start:end]
            voiced = np.isfinite(seg_f0) & (seg_f0 > 0)
            voicing_ratio = float(np.mean(voiced)) if len(seg_f0) > 0 else 0.0
            if voicing_ratio > 0:
                f0_vals = seg_f0[voiced]
                if len(f0_vals) > 1:
                    pitch_drop = clamp(
                        max(
                            0.0,
                            (np.max(f0_vals) - np.min(f0_vals))
                            / (np.max(f0_vals) + 1e-8),
                        )
                    )
        voicing_norm = 1 - voicing_ratio

        # ZCR & 谱质心
        zcr_mean = float(np.mean(zcr[start:end])) if zcr is not None else 0.0
        zcr_norm = clamp(zcr_mean)
        centroid_mean = (
            float(np.mean(centroid[start:end])) if centroid is not None else 0.0
        )
        spectral_norm = clamp(centroid_mean / (SAMPLE_RATE / 2))

        # ---------- 分段非线性组合 ----------
        S2_linear = (
            W_DURATION * D_short
            + W_POSTERIOR * posterior_norm
            + W_BLANK * blank_ratio
            + W_ENERGY * energy_norm
            + W_ENTROPY * H_norm
            + W_VOICING * voicing_norm
            + W_ZCR * zcr_norm
            + W_SPECTRAL * spectral_norm
        )

        # 分段非线性映射
        S2 = nonlinear_map(S2_linear)

        # ---------- 原因列表 ----------
        reasons = []
        if D_short > THRESH_DURATION:
            reasons.append("音素过短")
        if posterior < THRESH_POSTERIOR:
            reasons.append("模型后验率低")
        if blank_ratio > THRESH_BLANK:
            reasons.append("空白静音过长")
        if energy_norm > THRESH_ENERGY:
            reasons.append("能量低")
        if H_norm > THRESH_ENTROPY:
            reasons.append("模糊发音")
        if voicing_norm > THRESH_VOICING:
            reasons.append("低发声率")
        if pitch_drop > THRESH_PITCH_DROP:
            reasons.append("基频剧烈变化")
        if zcr_norm > THRESH_ZCR:
            reasons.append("过零率偏高(频率成分高)")

        return S2, reasons

    # ---------- 分析入口 ----------
    def analyze(
        self, audio_segment: AudioSegment, reference_text: str
    ) -> dict:
        """
        主分析入口：对给定音频执行完整流程并返回结构化结果。
        参数：
          - wav_data: 音频文件路径
          - reference_text: 可选的参考文本（中文），用于 S1 对齐与额外压缩/删除判定
          - is_show: 是否显示梅尔谱（调试用）
        返回：
          - dict，包含：
            - final_score: 句级分数（0-100）
            - sentence_risk_level: 风险分层字符串
            - phonemes: 音素级结果列表（每项包含 pid, token, pinyin, start, end, S2, S1, final, reasons 等）
        说明：
          - 方法内部会计算多种帧级特征：RMS、ZCR、谱质心、谱带宽、基频等，用于 score_s2。
        """
        audio_segment = load_audio_segment(audio_segment, SAMPLE_RATE)
        wav, logits = self.forward(audio_segment)
        segments = self.ctc_segments(logits)

        hop_length = int(FRAME_MS / 1000 * SAMPLE_RATE)
        rms = librosa.feature.rms(
            y=wav, frame_length=hop_length, hop_length=hop_length
        )[0]
        rms_mean = rms.mean()
        zcr = librosa.feature.zero_crossing_rate(
            y=wav, frame_length=hop_length, hop_length=hop_length
        )[0]
        n_fft = max(512, hop_length * 2)
        centroid = librosa.feature.spectral_centroid(
            y=wav, sr=SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length
        )[0]
        bandwidth = librosa.feature.spectral_bandwidth(
            y=wav, sr=SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length
        )[0]
        try:
            f0 = librosa.yin(
                y=wav,
                fmin=50,
                fmax=500,
                sr=SAMPLE_RATE,
                frame_length=2048,
                hop_length=hop_length,
            )
            f0 = np.where(np.isfinite(f0), f0, np.nan)
        except Exception:
            f0 = None

        durations = [
            (end - start) * FRAME_MS
            for pid, start, end, _ in segments
            if pid != BLANK_ID
        ]
        avg_duration_ms = np.mean(durations) if durations else FRAME_MS * 2
        phoneme_results = []

        for pid, start, end, probs_segment in segments:
            if pid == BLANK_ID:
                continue
            S2, reasons = self.score_s2(
                pid,
                start,
                end,
                probs_segment,
                rms,
                rms_mean,
                avg_duration_ms,
                zcr=zcr,
                centroid=centroid,
                bandwidth=bandwidth,
                f0=f0,
            )
            token = self.processor.tokenizer._convert_id_to_token(pid)
            py = token_to_pinyin(token)
            phoneme_results.append(
                {
                    "pid": int(pid),
                    "token": token,
                    "pinyin": py,
                    "start": start * FRAME_MS / 1000,
                    "end": end * FRAME_MS / 1000,
                    "S2": S2,
                    "S2_percent": S2 * 100,
                    "reasons": reasons,
                    "penalty_score": S2,
                    "penalty_score_percent": S2 * 100,
                    "risk_level": (
                        "高风险"
                        if S2 < self.risk_threshold
                        else "疑似吞音" if S2 < self.severe_threshold else "正常"
                    ),
                }
            )
        # ---------- 参考文本 S1（字符级，推荐方案） ----------
        if reference_text:
            targets = text2phoneme_or_token(
                reference_text, self.language, self.processor
            ).to(logits.device)

            # log_probs: (1, T, C)
            log_probs = torch.log_softmax(logits, dim=-1).unsqueeze(0)

            # CTC 强制对齐
            alignment, _ = F.forced_align(log_probs, targets, blank=BLANK_ID)

            alignment = alignment[0]  # (T,)
            target_ids = targets[0]  # (L,)

            s1_map = {}
            for pid in target_ids:
                frames = (alignment == pid).nonzero(as_tuple=True)[0]
                if len(frames) == 0:
                    s1_map[int(pid)] = (1.0, "deletion")
                else:
                    duration = frames[-1] - frames[0] + 1
                    s1_map[int(pid)] = (
                        0.7 if duration <= 2 else 0.0,
                        "compressed" if duration <= 2 else "ok",
                    )
            alpha = 0.2  # S2 权重（声学占比）
            beta = 0.8  # S1 权重（对齐占比）
            for p in phoneme_results:
                s1, r1 = s1_map.get(p["pid"], (0.0, "no_ref"))
                p["S1"] = s1
                p["S1_percent"] = s1 * 100

                p["penalty_score"] = 1 - ((1 - p["S2"]) ** alpha) * ((1 - s1) ** beta)

                if s1 > 0 and r1 not in p["reasons"]:
                    p["reasons"].append(r1)
                p["penalty_score_percent"] = p["penalty_score"] * 100

        return self.aggregate(phoneme_results)

    # ---------- 聚合句级评分 ----------
    def aggregate(self, phonemes) -> dict:
        """
        将音素级 final 分数聚合为句级评分与风险等级。
        参数：
          - phonemes: 音素结果列表，每项包含 'final' 键（0-1）
        返回：
          - dict 包含 final_score（0-100）、sentence_risk_level、phonemes（原列表）
        聚合策略：
          - 结合 90 百分位与高风险比例进行非线性聚合，兼顾极端高分项与整体分布。
        """
        scores = np.array([p["penalty_score"] for p in phonemes])
        max_score = scores.max()
        high_risk_ratio = (scores > self.severe_threshold).mean()
        # 改进非线性聚合：使用 percentile + 平均权重
        percentile90 = np.percentile(scores, 90)
        final_score = float(clamp(1 - 0.3 * percentile90 - 0.7 * high_risk_ratio, 0, 1))

        risk_level = (
            "高风险"
            if final_score < self.risk_threshold
            else "疑似吞音" if final_score < self.severe_threshold else "正常"
        )
        return {
            "final_score": final_score * 100,
            "sentence_risk_level": risk_level,
            "phonemes": phonemes,
        }


# ===============================
# 工具函数
# ===============================
def clamp(x, min_val=0.0, max_val=1.0) -> np.ndarray:
    """限制输出在 [0,1] 范围内"""
    return np.clip(x, min_val, max_val)


def nonlinear_map(
    S2_linear: float, alpha1=5.0, alpha2=5.0, threshold=0.5
) -> np.ndarray:
    """
    分段非线性映射优化版（严重吞音指数下降版）

    参数：
        S2_linear: 归一化线性分数（0~1）
        alpha1: 低分段斜率，轻微异常放大
        alpha2: 高分段斜率，严重异常压低
        threshold: 分段切换点，建议0.5左右
    返回：
        S2_nonlinear: 非线性映射后的分数，范围[0,1]
    """
    S2_linear = np.array(S2_linear)

    # Sigmoid平滑函数
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # 低分段映射: 轻微异常拉高
    low_map = 1 - np.exp(-alpha1 * S2_linear)

    # 高分段映射: 严重异常指数下降
    # 注意这里使用衰减: exp(-alpha2 * (x - threshold)) → 越严重越低
    high_map = np.exp(-alpha2 * (S2_linear - threshold))

    # 平滑过渡
    smooth_factor = sigmoid(12 * (S2_linear - threshold))  # 越大过渡越陡峭
    S2_nonlinear = low_map * (1 - smooth_factor) + high_map * smooth_factor

    # 限制在 [0,1]
    return clamp(S2_nonlinear)


def token_to_pinyin(token: str) -> str:
    """
    将模型输出的单个 token（通常为一个音素或字符）转换为带声调的拼音字符串。
    参数：
      - token: 模型 tokenizer 的单个 token
    返回：
      - 对应拼音（若无法转换则返回原 token）
    用途：
      - 在结果展示中给出更易理解的拼音注释。
    """
    py = pinyin(token, style=Style.TONE3, strict=False)
    return py[0][0] if py else token


def text2phoneme_or_token(
    reference_text: str, language: str, processor
) -> torch.Tensor:
    """
    参数：
      - reference_text: 可以转音素语言的参考文本
    返回：
      - 返回 shape: (1, L)，batch=1
    用途:
      - 音素按顺序返回，未匹配的 token 会被原样返回。
    """

    # 若语言支持 tokenizer，则使用 tokenizer
    if language in TOKEN_LANGUAGE:
        # 使用 tokenizer 对参考文本进行编码（processor 用于音频）
        return processor.tokenizer(
            reference_text, return_tensors="pt", add_special_tokens=False
        ).input_ids
    # 否则语言支持phoneme，则使用 phonemizer
    language = PHONEME_LANGUAGE_DICT[language]
    if language is None:
        raise ValueError(f"不支持的语言：{language}")

    from phonemizer import phonemize

    phonemes = phonemize(
        reference_text,
        backend="espeak",
        language=language,
        strip=True,
        with_stress=False,
    )
    # 转成 token ids
    ids = processor.tokenizer.convert_tokens_to_ids(phonemes)

    # 如果返回单个 int，包成 list
    if isinstance(ids, int):
        ids = [ids]

    # 防止空 tensor
    if len(ids) == 0:
        ids = [BLANK_ID]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # shape (1, L)

# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    detector = SwallowPredictor(language="chinese", use_admm=False)
    # audio_path = "../datasets/2.wav"
    # result = detector.analyze(audio_path)

    audio_path = "datasets/a_1.wav"
    reference_text = "我要定从高碑店东站到北京西的火车票"
    result = detector.analyze(
        audio_data=audio_path, reference_text=reference_text, is_show_mel=False
    )

    # detector = SwallowPredictor(language="english", use_admm=False)
    # audio_path = "../datasets/en_100538.wav"
    # reference_text = "when you start to eat like this something is the matter"
    # result = detector.analyze(audio_path, reference_text=reference_text)
    print(result)
