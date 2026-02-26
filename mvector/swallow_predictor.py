# ============================================================
# 企业级吞音检测系统
# 模块化架构：Config → FeatureExtractor → Scorer → Orchestrator
# ============================================================
import logging
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import librosa
import numpy as np
import torch
import torchaudio.functional as F
from pypinyin import pinyin, Style
from qwen_asr import Qwen3ForcedAligner
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers.utils import is_flash_attn_2_available

from mvector.utils.audio_utils import load_audio_segment
from yeaudio.audio import AudioSegment

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================
# (语言常量交由模型内部处理，统一为端到端方式)

SAMPLE_RATE = 16_000
FRAME_MS = 10.0
BLANK_ID = 0
MERGE_SHORT_MS = 10


# ============================================================
# SwallowConfig — 统一配置中心
# ============================================================
@dataclass
class SwallowConfig:
    """
    所有权重、阈值集中管理。
    权重在初始化后自动归一化（总和=1）。
    """
    # ── 权重 ──
    w_duration: float = 3.0
    w_blank: float = 3.0
    w_posterior: float = 6.0
    w_energy: float = 2.0
    w_entropy: float = 4.0
    w_voicing: float = 1.0
    w_zcr: float = 0.2
    w_spectral: float = 0.5
    w_bandwidth: float = 0.3

    # ── 阈值 ──
    thresh_duration: float = 0.7
    thresh_blank: float = 0.1
    thresh_posterior: float = 0.5
    thresh_energy: float = 0.4
    thresh_entropy: float = 0.2
    thresh_voicing: float = 0.3
    thresh_pitch_drop: float = 0.5
    thresh_zcr: float = 0.4
    thresh_bandwidth: float = 0.65

    # ── 风险阈值 ──
    risk_threshold: float = 0.45
    severe_threshold: float = 0.65

    # ── S1/S2 融合权重 ──
    alpha_s2: float = 0.2   # S2 声学权重
    beta_s1: float = 0.8    # S1 对齐权重

    # ── 归一化后权重（自动计算） ──
    _norm_weights: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        raw = {
            "duration": self.w_duration, "blank": self.w_blank,
            "posterior": self.w_posterior, "energy": self.w_energy,
            "entropy": self.w_entropy, "voicing": self.w_voicing,
            "zcr": self.w_zcr, "spectral": self.w_spectral,
            "bandwidth": self.w_bandwidth,
        }
        total = sum(raw.values())
        self._norm_weights = {k: v / total for k, v in raw.items()}

    @property
    def weights(self) -> dict:
        return self._norm_weights

    def risk_level(self, score: float) -> str:
        if score < self.risk_threshold:
            return "高风险"
        if score < self.severe_threshold:
            return "疑似吞音"
        return "正常"


# ============================================================
# GlobalFeatures — 全局声学特征容器
# ============================================================
class GlobalFeatures(NamedTuple):
    rms: np.ndarray
    rms_mean: float
    zcr: np.ndarray
    centroid: np.ndarray
    bandwidth: np.ndarray
    f0: Optional[np.ndarray]


# ============================================================
# S2Score — S2 评分结果容器
# ============================================================
class S2Score(NamedTuple):
    score: float        # 质量分 0～1，越高越好
    reasons: list       # 触发原因列表
    metrics: dict       # 各维度指标详情


# ============================================================
# AcousticFeatureExtractor — 全局音频特征一次性提取
# ============================================================
class AcousticFeatureExtractor:
    """无状态特征提取器，一次性计算音频的全局声学特征。"""

    @staticmethod
    def extract(wav: np.ndarray) -> GlobalFeatures:
        hop = int(FRAME_MS / 1000 * SAMPLE_RATE)
        rms = librosa.feature.rms(y=wav, frame_length=hop, hop_length=hop)[0]
        zcr = librosa.feature.zero_crossing_rate(y=wav, frame_length=hop, hop_length=hop)[0]
        n_fft = max(512, hop * 2)
        centroid = librosa.feature.spectral_centroid(y=wav, sr=SAMPLE_RATE, n_fft=n_fft, hop_length=hop)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=wav, sr=SAMPLE_RATE, n_fft=n_fft, hop_length=hop)[0]
        try:
            f0 = librosa.yin(y=wav, fmin=50, fmax=500, sr=SAMPLE_RATE, frame_length=2048, hop_length=hop)
            f0 = np.where(np.isfinite(f0), f0, np.nan)
        except Exception:
            f0 = None
        return GlobalFeatures(rms=rms, rms_mean=float(rms.mean()), zcr=zcr,
                              centroid=centroid, bandwidth=bandwidth, f0=f0)


# ============================================================
# S2AcousticScorer — 声学特征多维评分
# ============================================================
class S2AcousticScorer:
    """
    基于帧级概率 + 声学特征计算 S2 惩罚/质量分。
    S2 = 1 − weighted_penalty，越高越好。
    """

    def __init__(self, cfg: SwallowConfig):
        self.cfg = cfg
        self.w = cfg.weights

    def score(
        self,
        pid: int,
        start: int,
        end: int,
        probs_segment: np.ndarray,
        wav: np.ndarray,
        durations_ms: list,
        gf: GlobalFeatures,
        frame_ms: float = FRAME_MS,
    ) -> S2Score:
        avg_dur = np.mean(durations_ms) if durations_ms else frame_ms * 2
        duration_ms = (end - start) * frame_ms

        # ── 后验概率 ──
        posterior = float(probs_segment[:, pid].mean()) if pid < probs_segment.shape[1] else 0.0
        blank_ratio = float((probs_segment[:, BLANK_ID] > 0.6).mean())

        # ── 能量 ──
        energy = float(gf.rms[start:end].mean()) if start < len(gf.rms) and end <= len(gf.rms) else 0.0

        # ── 归一化特征 ──
        short_thresh = max(FRAME_MS, avg_dur * 0.5)
        D_short = _clamp((short_thresh - duration_ms) / short_thresh)
        posterior_norm = 1 - posterior
        energy_norm = _clamp((0.95 * gf.rms_mean - energy) / (0.8 * gf.rms_mean + 1e-8))

        # ── 谱熵 ──
        try:
            p = np.clip(probs_segment, 1e-12, 1.0)
            entropy_frames = -np.sum(p * np.log(p), axis=1)
            max_ent = np.log(probs_segment.shape[1])
            H_norm = _clamp(float(np.mean(entropy_frames)) / (max_ent + 1e-8))
        except Exception:
            H_norm = 0.0

        # ── 发声率 / 基频突变 ──
        voicing_ratio, pitch_drop = 0.0, 0.0
        if gf.f0 is not None and start < len(gf.f0):
            seg_f0 = gf.f0[start:end]
            voiced = np.isfinite(seg_f0) & (seg_f0 > 0)
            voicing_ratio = float(np.mean(voiced)) if len(seg_f0) > 0 else 0.0
            if voicing_ratio > 0:
                f0_vals = seg_f0[voiced]
                if len(f0_vals) > 1:
                    pitch_drop = _clamp(max(0.0, (np.max(f0_vals) - np.min(f0_vals)) / (np.max(f0_vals) + 1e-8)))
        voicing_norm = 1 - voicing_ratio

        # ── ZCR / 谱质心 / 谱带宽 ──
        zcr_norm = _clamp(float(np.mean(gf.zcr[start:end]))) if gf.zcr is not None else 0.0
        centroid_mean = float(np.mean(gf.centroid[start:end])) if gf.centroid is not None else 0.0
        spectral_norm = _clamp(centroid_mean / (SAMPLE_RATE / 2))
        bw_mean = float(np.mean(gf.bandwidth[start:end])) if gf.bandwidth is not None else 0.0
        bandwidth_norm = _clamp(1.0 - bw_mean / (SAMPLE_RATE / 2))

        # ── 线性加权 ──
        w = self.w
        S2_linear = (
            w["duration"] * D_short
            + w["posterior"] * posterior_norm
            + w["blank"] * blank_ratio
            + w["energy"] * energy_norm
            + w["entropy"] * H_norm
            + w["voicing"] * voicing_norm
            + w["zcr"] * zcr_norm
            + w["spectral"] * spectral_norm
            + w["bandwidth"] * bandwidth_norm
        )
        S2 = float(_clamp(1 - S2_linear))

        metrics = {
            "duration": float(w["duration"] * D_short),
            "blank": float(w["blank"] * blank_ratio),
            "posterior": float(w["posterior"] * posterior_norm),
            "energy": float(w["energy"] * energy_norm),
            "entropy": float(w["entropy"] * H_norm),
            "voicing": float(w["voicing"] * voicing_norm),
            "zcr": float(w["zcr"] * zcr_norm),
            "spectral": float(w["spectral"] * spectral_norm),
            "bandwidth": float(w["bandwidth"] * bandwidth_norm),
            "linear_score": float(S2_linear),
            "final_score": float(S2),
        }

        # ── 原因列表 ──
        c = self.cfg
        reasons = []
        if D_short > c.thresh_duration:
            reasons.append("音素过短")
        if posterior < c.thresh_posterior:
            reasons.append("模型后验率低")
        if blank_ratio > c.thresh_blank:
            reasons.append("空白静音过长")
        if energy_norm > c.thresh_energy:
            reasons.append("能量低")
        if H_norm > c.thresh_entropy:
            reasons.append("模糊发音")
        if voicing_norm > c.thresh_voicing:
            reasons.append("低发声率")
        if pitch_drop > c.thresh_pitch_drop:
            reasons.append("基频剧烈变化")
        if zcr_norm > c.thresh_zcr:
            reasons.append("过零率偏高(频率成分高)")
        if bandwidth_norm > c.thresh_bandwidth:
            reasons.append("谱带宽过窄(发音不清晰)")

        return S2Score(score=S2, reasons=reasons, metrics=metrics)


# ============================================================
# S1AlignmentScorer — 强制对齐评分
# ============================================================
class S1AlignmentScorer:
    """
    有参考文本时的强制对齐评分。
    优先 Qwen3 ForcedAligner，失败时回退 CTC forced_align。
    """

    def __init__(
        self,
        forced_aligner_model_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self._model_path = forced_aligner_model_path
        self._device = device
        self._dtype = dtype
        self._aligner: Optional[Qwen3ForcedAligner] = None

    # ── Qwen3 懒加载 ──
    def _get_aligner(self) -> Qwen3ForcedAligner:
        if self._aligner is not None:
            return self._aligner
        attn_impl = _select_attention_impl(self._dtype)
        self._aligner = Qwen3ForcedAligner.from_pretrained(
            self._model_path,
            dtype=self._dtype,
            device_map=self._device,
            attn_implementation=attn_impl,
        )
        logger.info("Qwen3 ForcedAligner loaded: %s", self._model_path)
        return self._aligner

    # ── Qwen3 对齐 ──
    def align_qwen3(
        self, wav: np.ndarray, reference_text: str, language: str
    ) -> Optional[list[dict]]:
        """
        返回 [{text, start, end}, ...] 或 None（失败时）。
        """
        try:
            aligner = self._get_aligner()
            results = aligner.align(audio=(wav, SAMPLE_RATE), text=reference_text, language=language)
        except Exception as e:
            logger.warning("Qwen3 align failed: %s", e)
            return None
        return _normalize_qwen_units(results)

    # ── CTC 强制对齐回退 ──
    @staticmethod
    def align_ctc_s1(
        phoneme_results: list,
        logits: torch.Tensor,
        reference_text: str,
        language: str,
        processor: Wav2Vec2Processor,
        cfg: SwallowConfig,
    ) -> bool:
        """对已有 phoneme_results 补充 CTC S1 评分。返回是否成功。"""
        try:
            targets = _text2phoneme_or_token(reference_text, language, processor).to(logits.device)
            log_probs = torch.log_softmax(logits, dim=-1).unsqueeze(0)
            alignment, _ = F.forced_align(log_probs, targets, blank=BLANK_ID)
            alignment = alignment[0]
            target_ids = targets[0]

            s1_map = {}
            for pid in target_ids:
                frames = (alignment == pid).nonzero(as_tuple=True)[0]
                if len(frames) == 0:
                    s1_map[int(pid)] = (1.0, "音素缺失")
                else:
                    dur = frames[-1] - frames[0] + 1
                    s1_map[int(pid)] = (0.7 if dur <= 2 else 0.0, "音素模糊" if dur <= 2 else "ok")

            for p in phoneme_results:
                s1, r1 = s1_map.get(p["pid"], (0.0, "no_ref"))
                p["S1"] = s1
                p["S1_percent"] = s1 * 100
                p["penalty_score"] = float(_clamp(cfg.alpha_s2 * p["S2"] + cfg.beta_s1 * (1 - s1)))
                if s1 > 0 and r1 not in p["reasons"]:
                    p["reasons"].append(r1)
                p["penalty_score_percent"] = p["penalty_score"] * 100
                p["risk_level"] = cfg.risk_level(p["penalty_score"])
            return True
        except Exception as e:
            logger.warning("CTC S1 alignment failed: %s", e)
            return False


# ============================================================
# SwallowPredictor — 编排器
# ============================================================
class SwallowPredictor:
    """
    吞音检测编排器。
    - 有参考文本：Qwen3 强制对齐 → S1 + S2 综合评分
    - 无参考文本：CTC 分段 → 仅 S2 声学评分，token 为空串
    """

    def __init__(
        self,
        language="zh-cn",
        acoustic_model_path="jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        forced_aligner_model_path="Qwen/Qwen3-ForcedAligner-0.6B",
        use_forced_aligner=True,
        risk_threshold=0.45,
        severe_threshold=0.65,
        use_gpu=True,
        use_admm=False,
    ):
        self.language = language.lower()

        # ── 设备 ──
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.dtype = torch.bfloat16
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32

        # ── 配置 ──
        self.cfg = SwallowConfig(risk_threshold=risk_threshold, severe_threshold=severe_threshold)

        # ── wav2vec2 声学提取模型 ──
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        self.processor = Wav2Vec2Processor.from_pretrained(acoustic_model_path)
        attn_impl = _select_attention_impl(self.dtype)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            acoustic_model_path,
            attn_implementation=attn_impl if torch.cuda.is_available() else None,
        ).to(self.device)
        self.model.eval()

        # ── 子模块 ──
        self._extractor = AcousticFeatureExtractor()
        self._s2_scorer = S2AcousticScorer(self.cfg)
        self._s1_scorer = (
            S1AlignmentScorer(forced_aligner_model_path, self.device, self.dtype)
            if use_forced_aligner
            else None
        )

    # ── Forward ──
    def forward(self, audio_segment: AudioSegment) -> tuple:
        wav = audio_segment.samples.astype(np.float32)
        inputs = self.processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(inputs.input_values.to(self.device)).logits
        return wav, logits[0]

    # ── CTC 分段 ──
    def _ctc_segments(self, logits: torch.Tensor) -> list:
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
        for pid, s, e, pseg in segments:
            dur_ms = (e - s) * FRAME_MS
            if merged and dur_ms < MERGE_SHORT_MS:
                lp, ls, le, lpseg = merged.pop()
                merged.append((lp, ls, e, np.concatenate([lpseg, pseg], axis=0)))
            else:
                merged.append((pid, s, e, pseg))
        return merged

    # ── 构造统一的 phoneme 结果字典 ──
    def _make_phoneme(
        self,
        pid: int,
        token: str,
        start_sec: float,
        end_sec: float,
        s2: S2Score,
        s1: float = 0.0,
        s1_reason: str = "ok",
    ) -> dict:
        cfg = self.cfg
        penalty = float(_clamp(cfg.alpha_s2 * s2.score + cfg.beta_s1 * (1 - s1)))
        reasons = list(s2.reasons)
        if s1 > 0 and s1_reason not in reasons:
            reasons.append(s1_reason)
        py = _token_to_pinyin(token) if token else ""
        return {
            "pid": int(pid),
            "token": token,
            "pinyin": py,
            "start": float(start_sec),
            "end": float(end_sec),
            "S2": float(s2.score),
            "S2_percent": float(s2.score * 100),
            "S1": float(s1),
            "S1_percent": float(s1 * 100),
            "metrics": s2.metrics,
            "reasons": reasons,
            "penalty_score": float(penalty),
            "penalty_score_percent": float(penalty * 100),
            "risk_level": cfg.risk_level(penalty),
        }

    # ── 有参考文本：Qwen3 对齐路径 ──
    def _build_from_qwen3(self, wav: np.ndarray, logits: torch.Tensor, reference_text: str) -> Optional[list]:
        if self._s1_scorer is None:
            return None

        units = self._s1_scorer.align_qwen3(wav, reference_text, self.language)
        if not units:
            return None

        gf = self._extractor.extract(wav)
        probs = torch.log_softmax(logits, dim=-1).cpu().exp().numpy()
        total_frames = probs.shape[0]
        real_frame_ms = (len(wav) / SAMPLE_RATE * 1000.0) / total_frames

        durations_ms = [(u["end"] - u["start"]) * 1000.0 for u in units]
        durations_sec = [u["end"] - u["start"] for u in units]
        base_dur = max(0.03, float(np.median(durations_sec))) if durations_sec else 0.08

        results = []
        for u in units:
            fs = max(0, min(int(u["start"] * 1000.0 / real_frame_ms), total_frames - 1))
            fe = max(fs + 1, min(int(u["end"] * 1000.0 / real_frame_ms), total_frames))
            pseg = probs[fs:fe]
            if pseg.shape[0] == 0:
                pseg = probs[fs:fs + 1]

            mean_p = pseg.mean(axis=0)
            mean_p[BLANK_ID] = 0.0
            pid = int(np.argmax(mean_p))

            s2 = self._s2_scorer.score(pid, fs, fe, pseg, wav, durations_ms, gf)

            # S1：基于对齐时长比例
            unit_dur = max(1e-4, u["end"] - u["start"])
            ratio = unit_dur / (base_dur + 1e-8)
            s1 = float(_clamp((1.0 - ratio) * 0.7, 0.0, 0.7)) if ratio < 1.0 else 0.0
            s1_reason = "音素模糊" if s1 > 0 else "ok"

            results.append(self._make_phoneme(pid, u["text"], u["start"], u["end"], s2, s1, s1_reason))

        return results

    # ── 无参考文本 / 回退：CTC 分段路径 ──
    def _build_from_ctc(self, segments: list, wav: np.ndarray, logits: torch.Tensor, reference_text: str) -> list:
        total_frames = logits.shape[0]
        real_frame_ms = (len(wav) / SAMPLE_RATE * 1000.0) / total_frames
        gf = self._extractor.extract(wav)

        durations_ms = [
            (end - start) * real_frame_ms
            for pid, start, end, _ in segments if pid != BLANK_ID
        ]

        results = []
        for pid, start, end, pseg in segments:
            # Mumble detection：高能量的 Blank 段
            is_mumble = False
            if pid == BLANK_ID:
                dur_ms = (end - start) * FRAME_MS
                seg_energy = float(gf.rms[start:end].mean()) if start < len(gf.rms) else 0.0
                if dur_ms > 50 and seg_energy > 0.5 * gf.rms_mean:
                    is_mumble = True
                else:
                    continue

            s2 = self._s2_scorer.score(pid, start, end, pseg, wav, durations_ms, gf)

            if is_mumble:
                s2 = S2Score(
                    score=min(s2.score, 0.4),
                    reasons=s2.reasons + (["模糊发音(Mumble)"] if "模糊发音(Mumble)" not in s2.reasons else []),
                    metrics=s2.metrics,
                )

            # 无参考文本：token 为空串，不给前端转写
            if reference_text:
                token = "[MUMBLE]" if is_mumble else self.processor.tokenizer._convert_id_to_token(pid)
            else:
                token = ""

            start_sec = start * real_frame_ms / 1000
            end_sec = end * real_frame_ms / 1000
            results.append(self._make_phoneme(pid, token, start_sec, end_sec, s2))

        # 有参考文本但 Qwen3 失败的回退：补充 CTC S1
        if reference_text and self._s1_scorer is not None:
            S1AlignmentScorer.align_ctc_s1(results, logits, reference_text, self.language, self.processor, self.cfg)

        return results

    # ── 主分析入口 ──
    def analyze(self, audio_segment: AudioSegment, reference_text: str = None) -> dict:
        """
        对给定音频执行完整吞音检测流程。
        - 有参考文本：优先 Qwen3 强制对齐构建结果
        - 无参考文本：CTC 分段 + 仅 S2 评分，token 为空串
        """
        audio_segment = load_audio_segment(audio_segment, SAMPLE_RATE)
        wav, logits = self.forward(audio_segment)

        phoneme_results = None

        # ── 有参考文本：优先 Qwen3 ──
        if reference_text:
            logger.info("Path: Qwen3 Forced Aligner (lang=%s)", self.language)
            phoneme_results = self._build_from_qwen3(wav, logits, reference_text)

        # ── 回退 / 无参考文本：CTC ──
        if phoneme_results is None:
            if reference_text:
                logger.warning("Qwen3 failed or unavailable, falling back to CTC segmentation")
            else:
                logger.info("No reference text, using CTC segmentation (S2 only)")
            segments = self._ctc_segments(logits)
            phoneme_results = self._build_from_ctc(segments, wav, logits, reference_text)

        return self._aggregate(phoneme_results)

    # ── 聚合句级评分 ──
    def _aggregate(self, phonemes: list) -> dict:
        scores = np.array([p["penalty_score"] for p in phonemes])
        mean_score = float(np.mean(scores))
        p10 = float(np.percentile(scores, 10))
        final = float(_clamp(0.4 * p10 + 0.6 * mean_score, 0, 1))
        return {
            "final_score": final * 100,
            "sentence_risk_level": self.cfg.risk_level(final),
            "phonemes": phonemes,
        }


# ============================================================
# 工具函数
# ============================================================
def _clamp(x, min_val=0.0, max_val=1.0):
    return np.clip(x, min_val, max_val)


def _token_to_pinyin(token: str) -> str:
    py = pinyin(token, style=Style.TONE3, strict=False)
    return py[0][0] if py else token


def _normalize_qwen_units(results) -> Optional[list[dict]]:
    """将 Qwen3 对齐器原始输出规范化为 [{text, start, end}, ...]。"""
    if not isinstance(results, (list, tuple)) or len(results) == 0:
        return None
    units = results[0]
    normalized = []
    for u in units:
        if u is None:
            continue
        if isinstance(u, dict):
            text = u.get("text")
            st = u.get("start_time", u.get("start"))
            et = u.get("end_time", u.get("end"))
        else:
            text = getattr(u, "text", None)
            st = getattr(u, "start_time", None)
            et = getattr(u, "end_time", None)
        if text is None:
            continue
        try:
            st, et = float(st), float(et)
        except (TypeError, ValueError):
            continue
        if et < st:
            continue
        if et == st:
            et = st + 0.01
        normalized.append({"text": text, "start": st, "end": et})
    return normalized if normalized else None


def _text2phoneme_or_token(reference_text: str, language: str, processor) -> torch.Tensor:
    """将参考文本转为 token ID tensor，shape (1, L)。"""
    # 统一使用 Processor 的 Tokenizer 进行回退对齐编码，移除厚重的 phonemizer 依赖
    return processor.tokenizer(reference_text, return_tensors="pt", add_special_tokens=False).input_ids


def _select_attention_impl(dtype: torch.dtype) -> str:
    if not torch.cuda.is_available():
        return "eager"
    if dtype not in (torch.float16, torch.bfloat16):
        return "sdpa"
    if not is_flash_attn_2_available():
        return "sdpa"
    try:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    except Exception:
        return "sdpa"
    return "sdpa"


# ============================================================
# 向后兼容：保留旧的公共函数名
# ============================================================
clamp = _clamp
token_to_pinyin = _token_to_pinyin
text2phoneme_or_token = _text2phoneme_or_token
select_attention_impl = _select_attention_impl


# ============================================================
# 示例运行
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector = SwallowPredictor(language="chinese", use_admm=False)
    audio_path = "datasets/a_1.wav"
    reference_text = "我要订从高碑店东站到北京西的火车票"
    result = detector.analyze(
        audio_segment=audio_path, reference_text=reference_text,
    )
    print(result)
