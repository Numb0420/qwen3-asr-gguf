# coding=utf-8
import os
import time
import re
import codecs
import dataclasses
import threading
import numpy as np
import multiprocessing as mp
from pathlib import Path
from collections import deque
from typing import Optional, List, Callable

from .chunk_cache import (
    cache_embd,
    cache_entry,
    is_full_window,
    raw_window_text,
    set_cache_embd,
    set_cache_raw_text,
)
from .schema import MsgType, StreamingMessage, DecodeResult, ASREngineConfig, TranscribeResult, ForcedAlignItem, ForcedAlignResult
from .utils import normalize_language_name, validate_language
from .encoder import QwenAudioEncoder
from .repetition import collapse_repetitions, max_new_tokens_for_audio, suffix_loop_pattern, trim_loop_tail
from . import llama

@dataclasses.dataclass
class ASRS_Segment:
    """管理分片记忆及其物理时间坐标"""
    idx: int
    audio_start: float
    audio_end: float
    text: str = ""
    items: List[ForcedAlignItem] = None   

class QwenASREngine:
    """Qwen3-ASR 流式转录引擎 (GGUF 后端) - 统一辅助进程架构"""
    def __init__(self, config: ASREngineConfig):
        self.config = config
        self.verbose = config.verbose
        if self.verbose: print(f"--- [QwenASR] 初始化引擎 (Provider: {config.onnx_provider}) ---")
        
        # 路径解析
        llm_gguf = os.path.join(config.model_dir, config.llm_fn)
        frontend_path = os.path.join(config.model_dir, config.encoder_frontend_fn)
        backend_path = os.path.join(config.model_dir, config.encoder_backend_fn)

        # 1. 初始化 Encoder
        self.encoder = QwenAudioEncoder(
            frontend_path=frontend_path,
            backend_path=backend_path,
            onnx_provider=config.onnx_provider,
            dml_pad_to=config.dml_pad_to,
            verbose=self.verbose
        )

        # 2. 初始化 Aligner (可选)
        self.aligner = None
        if config.enable_aligner and config.align_config:
            from .aligner import QwenForcedAligner
            self.aligner = QwenForcedAligner(config.align_config)
        
        # 3. 加载识别 LLM
        self.model = llama.LlamaModel(llm_gguf, use_gpu=config.llm_use_gpu)
        self.embedding_table = llama.get_token_embeddings_gguf(llm_gguf)
        self.ctx = llama.LlamaContext(self.model, n_ctx=config.n_ctx, n_batch=4096, embeddings=False)

        # 缓存 Token ID
        self.ID_IM_START = self.model.token_to_id("<|im_start|>")
        self.ID_IM_END = self.model.token_to_id("<|im_end|>")
        self.ID_AUDIO_START = self.model.token_to_id("<|audio_start|>")
        self.ID_AUDIO_END = self.model.token_to_id("<|audio_end|>")
        self.ID_ASR_TEXT = self.model.token_to_id("<asr_text>")

    def shutdown(self):
        self.close()

    def close(self):
        """Release native llama.cpp / ONNX resources in a fixed order."""
        ctx = getattr(self, "ctx", None)
        if ctx is not None:
            try:
                if getattr(ctx, "ptr", None):
                    llama.llama_free(ctx.ptr)
                    ctx.ptr = None
            except Exception:
                pass
            self.ctx = None

        model = getattr(self, "model", None)
        if model is not None:
            try:
                if getattr(model, "ptr", None):
                    llama.llama_model_free(model.ptr)
                    model.ptr = None
            except Exception:
                pass
            self.model = None

        encoder = getattr(self, "encoder", None)
        if encoder is not None:
            if hasattr(encoder, "close"):
                encoder.close()
            self.encoder = None

        self.embedding_table = None
        aligner = getattr(self, "aligner", None)
        if aligner is not None:
            if hasattr(aligner, "close"):
                try:
                    aligner.close()
                except Exception:
                    pass
            self.aligner = None
        if self.verbose:
            print("--- [QwenASR] 引擎已关闭 ---")

    def _build_prompt_embd(self, audio_embd: np.ndarray, prefix_text: str, context: Optional[str], language: Optional[str]):
        """构造用于 LLM 输入的 Embedding 序列 (区块化打包模式)"""
        def tk(t): return self.model.tokenize(t)

        # 1. 区块 A: 音频之前的所有内容 (System + User Header)
        prefix_str = f"system\n{context or 'You are a helpful assistant.'}"
        prefix_tokens = [self.ID_IM_START] + tk(prefix_str) + [self.ID_IM_END] + \
                        [self.ID_IM_START] + tk("user\n") + [self.ID_AUDIO_START]
        
        # 2. 区块 B: 音频之后的所有内容 (Instruction + Assistant Header + History)
        suffix_head = f"assistant\n"
        if language: suffix_head += f"language {language}"
        
        suffix_tokens = [self.ID_AUDIO_END] + [self.ID_IM_END] + \
                        [self.ID_IM_START] + tk(suffix_head) + [self.ID_ASR_TEXT]
        prefix_ids = tk(prefix_text) if prefix_text else []
        max_n = int(getattr(self.ctx, "n_batch", 2048) or 2048)
        n_fixed = len(prefix_tokens) + audio_embd.shape[0] + len(suffix_tokens)
        room = max_n - n_fixed - 8
        if room < 0:
            raise RuntimeError(
                f"ASR prompt+audio tokens {n_fixed} exceed n_batch={max_n}; shorten the chunk"
            )
        if len(prefix_ids) > room:
            prefix_ids = prefix_ids[-room:]
        suffix_tokens = suffix_tokens + prefix_ids

        # 3. 统计并拼接
        n_pre, n_aud, n_suf = len(prefix_tokens), audio_embd.shape[0], len(suffix_tokens)
        total_embd = np.zeros((n_pre + n_aud + n_suf, self.model.n_embd), dtype=np.float32)
        
        total_embd[:n_pre] = self.embedding_table[prefix_tokens]
        total_embd[n_pre : n_pre + n_aud] = audio_embd
        total_embd[n_pre + n_aud:] = self.embedding_table[suffix_tokens]
        
        return total_embd

    def _decode(
        self, 
        full_embd: np.ndarray, 
        prefix_text: str, 
        rollback_num: int,
        is_last_chunk: bool = False, 
        temperature: float = 0.4, 
        streaming: bool = False,
        max_new_tokens: int = 512,
        abort_event: "threading.Event | None" = None,
    ) -> DecodeResult:
        """底层方法：执行单次 LLM 生成循环（物理推理）"""
        result = DecodeResult()
        total_len = full_embd.shape[0]
        n_ctx = int(getattr(self.ctx, "n_ctx", 2048) or 2048)
        if total_len >= n_ctx:
            raise RuntimeError(
                f"ASR prefill {total_len} tokens >= n_ctx={n_ctx}; shorten the chunk"
            )
        if not np.isfinite(full_embd).all():
            raise RuntimeError("ASR encoder embedding has NaN/Inf; skip this window")
        pos_base = np.arange(0, total_len, dtype=np.int32)
        pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(total_len, dtype=np.int32)])
        batch = llama.LlamaBatch(max(total_len * 4, 8192), self.model.n_embd, 1)
        batch.set_embd(full_embd, pos=pos_arr)
        
        # 1. Prefill
        self.ctx.clear_kv_cache()
        t_pre_start = time.time()
        self.ctx.decode(batch)
        prefill_time = time.time() - t_pre_start
        
        # 2. Generation Loop（使用新采样器和随机种子）
        t_gen_start = time.time()
        n_gen_tokens = 0
        display_queue = deque()
        stable_tokens = []
        stable_text_acc = ""
        text_decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        # Leave one slot so decode_token cannot walk past n_ctx (GGML_ASSERT i01 < ne01).
        token_budget = max(1, min(int(max_new_tokens), n_ctx - total_len - 1))
        n_vocab = int(llama.llama_vocab_n_tokens(self.model.vocab))
        
        # 每次解码使用新的随机种子
        seed = int(np.random.randint(0, 2**31 - 1))
        sampler = llama.LlamaSampler(temperature=temperature, seed=seed)
        last_sampled_token = sampler.sample(self.ctx.ptr)
        for _ in range(token_budget):
            # final 抢占：同 utt 的 final 入队时 set abort_event，partial 尽快退出
            if abort_event is not None and abort_event.is_set():
                result.is_aborted = True
                result.abort_reason = "final_preempt"
                result.text = ""
                return result
            if last_sampled_token in [self.model.eos_token, self.ID_IM_END]:
                break
            # Invalid ids abort the whole process inside ggml GET_ROWS.
            if last_sampled_token < 0 or last_sampled_token >= n_vocab:
                break
            
            if self.ctx.decode_token(last_sampled_token) != 0:
                    break
            
            display_queue.append(last_sampled_token)
            if len(display_queue) > rollback_num:
                ready_token = display_queue.popleft()
                stable_tokens.append(ready_token)
                piece = text_decoder.decode(self.model.token_to_bytes(ready_token))
                if piece:
                    if streaming: print(re.sub(r'([，。？！：,\.])', r'\1\n', piece), end='', flush=True)
                    stable_text_acc += piece
                    loop_pat = suffix_loop_pattern(stable_text_acc)
                    if loop_pat is not None:
                        result.is_aborted = True
                        result.abort_reason = "loop"
                        stable_text_acc = trim_loop_tail(stable_text_acc, loop_pat, keep=2)
                        break
            
            last_sampled_token = sampler.sample(self.ctx.ptr)
            n_gen_tokens += 1
            
        gen_time = time.time() - t_gen_start
        del sampler  # 释放采样器资源
        del batch
            
        if is_last_chunk and not result.is_aborted:
            while display_queue:
                t = display_queue.popleft()
                stable_tokens.append(t)
                piece = text_decoder.decode(self.model.token_to_bytes(t))
                if piece:
                    if streaming: print(re.sub(r'([，。？！：,\.])', r'\1\n', piece), end="", flush=True)
                    stable_text_acc += piece
                    loop_pat = suffix_loop_pattern(stable_text_acc)
                    if loop_pat is not None:
                        result.is_aborted = True
                        result.abort_reason = "loop"
                        stable_text_acc = trim_loop_tail(stable_text_acc, loop_pat, keep=2)
                        break
            if not result.is_aborted:
                final_p = text_decoder.decode(b"", final=True)
                if final_p: 
                    if streaming: print(final_p, end='', flush=True)
                    stable_text_acc += final_p
        
        # 填充结果（内核输出标准化）
        result.text = collapse_repetitions(stable_text_acc)
        result.stable_tokens = stable_tokens
        result.t_prefill = prefill_time
        result.t_generate = gen_time
        result.n_prefill = total_len
        result.n_generate = n_gen_tokens
        if self.verbose and result.is_aborted:
            print(f"[QwenASR] decode aborted on phrase loop, kept {len(result.text)} chars")
        return result

    def _safe_decode(
        self, 
        full_embd: np.ndarray, 
        prefix_text: str, 
        rollback_num: int, 
        is_last_chunk: bool, 
        temperature: float, 
        streaming: bool = False,
        max_new_tokens: int = 512,
        abort_event: "threading.Event | None" = None,
    ) -> DecodeResult:
        """单次解码：短语循环只截断，不加温重试。"""
        return self._decode(
            full_embd,
            prefix_text,
            rollback_num,
            is_last_chunk,
            temperature,
            streaming=streaming,
            max_new_tokens=max_new_tokens,
            abort_event=abort_event,
        ) 

    def _print_stats(self, stats: dict, audio_duration: float, t_total: float):
        """打印转录过程的性能统计指标"""
        rtf = t_total / audio_duration if audio_duration > 0 else 0
        pre_speed = stats["prefill_tokens"] / stats["prefill_time"] if stats["prefill_time"] > 0 else 0
        gen_speed = stats["decode_tokens"] / stats["decode_time"] if stats["decode_time"] > 0 else 0
        
        print(f"\n\n📊 性能统计:")
        print(f"  🔹 RTF (实时率) : {rtf:.3f} (越小越快)")
        print(f"  🔹 音频时长    : {audio_duration:.2f} 秒")
        print(f"  🔹 总处理耗时  : {t_total:.2f} 秒")
        if stats.get("align_time"):
            print(f"  🔹 对齐耗时    : {stats['align_time']:.3f} 秒")
        print(f"  🔹 编码耗时    : {stats['encode_time']:.3f} 秒")
        print(f"  🔹 LLM 预填充  : {stats['prefill_time']:.3f} 秒 ({stats['prefill_tokens']} tokens, {pre_speed:.1f} tokens/s)")
        print(f"  🔹 LLM 生成    : {stats['decode_time']:.3f} 秒 ({stats['decode_tokens']} tokens, {gen_speed:.1f} tokens/s)")

    def transcribe(
        self, 
        audio_file: str, 
        language: Optional[str] = None, 
        context: Optional[str] = None, 
        start_second: float = 0.0,
        duration: float = 0.0,
        temperature: float = 0.4,
        rollback_num: int = 5
    ) -> TranscribeResult:
        """运行完整转录流水线 (从文件加载音频)"""
        from .audio import load_audio
        audio = load_audio(audio_file, start_second=start_second, duration=duration)
        
        return self.asr(
            audio=audio,
            context=context or "",
            language=language,
            chunk_size_sec=self.config.chunk_size,
            memory_chunks=self.config.memory_num,
            temperature=temperature,
            rollback_num=rollback_num,
            streaming=False,
        )

    def asr(
        self, 
        audio: np.ndarray,
        context: Optional[str],
        language: Optional[str],
        chunk_size_sec: float = 40.0,
        memory_chunks: int = 2,
        temperature: float = 0.4,
        rollback_num: int = 5,
        streaming: bool = False,
        do_align: bool = True,
        on_chunk: Optional[Callable[[int, int], None]] = None,
        prefix_text: str = "",
        abort_event: "threading.Event | None" = None,
        encoder_cache=None,
        skip_decode_before: int = 0,
        yield_event: "threading.Event | None" = None,
    ) -> TranscribeResult:
        """运行完整转录流水线 (三级流水线：i+1 预取, i 识别, i-1 对齐)"""
        # 语言归一化与校验
        if language:
            language = normalize_language_name(language)
            validate_language(language)

        sr = 16000
        configured_spc = max(1, int(round(float(chunk_size_sec) * sr)))
        samples_per_chunk = configured_spc
        total_len = len(audio)
        if total_len <= 0 or samples_per_chunk <= 0:
            return TranscribeResult(text="", alignment=None, performance={
                "prefill_time": 0.0, "decode_time": 0.0,
                "prefill_tokens": 0, "decode_tokens": 0,
                "encode_time": 0.0, "align_time": 0.0,
                "cache_hits": 0, "n_chunks": 0,
                "decode_skipped": 0, "committed_chunks": 0,
            })
        # Silence-overlap slices can be 40.4s; int(40.4*16000) is 646399, which
        # would otherwise create a 1-sample second window and empty align audio.
        # Shrinking must happen after configured_spc is saved so a short clip
        # is not treated as a cacheable full window.
        if total_len <= samples_per_chunk + int(0.05 * sr):
            samples_per_chunk = total_len
            num_chunks = 1
        else:
            num_chunks = int(np.ceil(total_len / samples_per_chunk))
        total_duration = total_len / sr
        
        # 记忆管理 (预定义所有分片的物理边界)
        all_segments: List[ASRS_Segment] = [
            ASRS_Segment(
                idx=i,
                audio_start=i * chunk_size_sec,
                audio_end=min((i + 1) * chunk_size_sec, total_duration)
            ) for i in range(num_chunks)
        ]
        asr_memory = deque(maxlen=memory_chunks) # 存储 (embd, text)
        history_text = prefix_text or ""
        total_full_text = ""
        all_aligned_items: List[ForcedAlignItem] = []
        
        # 统计指标
        stats = {
            "prefill_time": 0.0, "decode_time": 0.0,
            "prefill_tokens": 0, "decode_tokens": 0,
            "encode_time": 0.0, "align_time": 0.0,
            "cache_hits": 0,
            "n_chunks": num_chunks,
            "decode_skipped": 0,
            "committed_chunks": 0,
            "yielded": 0,
            "next_chunk": 0,
        }
        t_main_start = time.time()
        skip_before = max(0, int(skip_decode_before or 0))

        # --- 顺序同步处理循环 ---
        for i in range(num_chunks):
            if abort_event is not None and abort_event.is_set():
                stats["aborted"] = "final_preempt"
                break
            if yield_event is not None and yield_event.is_set() and i > 0:
                stats["yielded"] = 1
                stats["next_chunk"] = i
                break
            # 1. 编码第 i 片段
            s, e = i * samples_per_chunk, min((i + 1) * samples_per_chunk, total_len)
            chunk_data = audio[s:e]
            if chunk_data.size == 0:
                continue
            full_window = is_full_window(int(e - s), configured_spc)
            if len(chunk_data) < samples_per_chunk: 
                chunk_data = np.pad(chunk_data, (0, samples_per_chunk - len(chunk_data)))

            audio_feature = None
            enc_time = 0.0
            cached = cache_embd(cache_entry(encoder_cache, i)) if full_window else None
            if full_window and cached is not None:
                audio_feature = cached
                stats["cache_hits"] += 1
            else:
                audio_feature, enc_time = self.encoder.encode(chunk_data)
                stats["encode_time"] += enc_time
                if full_window and encoder_cache is not None and audio_feature is not None:
                    set_cache_embd(encoder_cache, i, np.copy(audio_feature))
            was_last = (i == num_chunks - 1)

            if i < skip_before and audio_feature is not None:
                kept = raw_window_text(cache_entry(encoder_cache, i))
                if kept is not None:
                    all_segments[i].text = kept
                    asr_memory.append((audio_feature, kept))
                    total_full_text += kept
                    stats["decode_skipped"] += 1
                    if on_chunk:
                        on_chunk(i + 1, num_chunks)
                    continue

            # 2. 识别第 i 片段文字
            prefix_text = "".join([m[1] for m in asr_memory]) or history_text
            combined_audio = np.concatenate([m[0] for m in asr_memory] + [audio_feature], axis=0)
            full_embd = self._build_prompt_embd(combined_audio, prefix_text, context, language)
            chunk_dur = (e - s) / float(sr)
            max_tok = max_new_tokens_for_audio(chunk_dur)

            res = self._safe_decode(
                full_embd,
                prefix_text,
                rollback_num,
                was_last,
                temperature,
                streaming=streaming,
                max_new_tokens=max_tok,
                abort_event=abort_event,
            )
            clean_text = collapse_repetitions(res.text)
            res.text = clean_text

            # final 抢占：立即结束整段 asr，不再处理后续 chunk
            if res.abort_reason == "final_preempt":
                break

            # 更新记忆与统计
            all_segments[i].text = clean_text
            asr_memory.append((audio_feature, clean_text))
            if encoder_cache is not None:
                set_cache_raw_text(encoder_cache, i, clean_text)
            
            total_full_text += clean_text
            stats["prefill_tokens"] += res.n_prefill; stats["prefill_time"] += res.t_prefill
            stats["decode_tokens"] += res.n_generate; stats["decode_time"] += res.t_generate

            # 3. 对齐第 i 片段 (同步) — 先清洗再送 Aligner
            if do_align and self.aligner and clean_text.strip():
                t_align_start = time.time()
                # 计算偏移（同步版本逻辑简化：直接使用片起点，不考虑前片动态边界）
                offset_sec = all_segments[i].audio_start
                s_smpl, e_smpl = int(round(offset_sec * sr)), int(round(all_segments[i].audio_end * sr))
                s_smpl = max(0, min(s_smpl, total_len))
                e_smpl = max(s_smpl, min(e_smpl, total_len))
                audio_slice = audio[s_smpl:e_smpl]
                if audio_slice.size == 0:
                    continue
                
                align_res = self.aligner.align(
                    audio_slice, 
                    clean_text, 
                    language=language, 
                    offset_sec=float(offset_sec)
                )
                all_segments[i].items = align_res.items
                all_aligned_items.extend(align_res.items)
                stats["align_time"] += (time.time() - t_align_start)

            if on_chunk:
                on_chunk(i + 1, num_chunks)

        # 4. 结果整理
        all_aligned_items.sort(key=lambda x: x.start_time)
        t_total = time.time() - t_main_start
        if self.verbose: self._print_stats(stats, total_duration, t_total)
            
        return TranscribeResult(
            text=total_full_text,
            alignment=ForcedAlignResult(items=all_aligned_items) if all_aligned_items else None,
            performance=stats
        )
