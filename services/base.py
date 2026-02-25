"""
services/base.py — 异步服务公共基础设施
────────────────────────────────────────
提供：
  run_sync(fn, *args, **kwargs)
      将同步阻塞调用提交到默认线程池，避免阻塞 asyncio 事件循环。

  AsyncServiceBase
      - _get_async_lock(name)      按名称延迟创建并缓存 asyncio.Lock
      - _sync_lazy_init(check, init)
                                   线程安全双重检查懒加载（同步版，适合启动钩子）
      - _async_lazy_init(lock_name, check, init)
                                   线程安全双重检查懒加载（asyncio 版，不阻塞事件循环）
"""

from __future__ import annotations

import asyncio
import functools
import threading
from typing import Any, Callable


# ── 工具函数 ──────────────────────────────────────────────────────────────────

async def run_sync(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    将同步阻塞调用提交到默认线程池执行，返回结果。

    用法：
        result = await run_sync(predictor.predict, audio_segment)
        result = await run_sync(predictor.diarize, audio, speaker_num=2)
    """
    loop = asyncio.get_event_loop()
    if kwargs:
        fn = functools.partial(fn, **kwargs)
    return await loop.run_in_executor(None, fn, *args)


# ── 基类 ──────────────────────────────────────────────────────────────────────

class AsyncServiceBase:
    """
    异步服务基类。

    子类获得：
      - 统一的同步线程锁（_sync_init_lock）
      - 按名称管理的 asyncio.Lock 集合（_get_async_lock）
      - 同步 / 异步 双重检查懒加载辅助方法
    """

    def __init__(self) -> None:
        # 用于同步懒加载（如 FastAPI lifespan startup 中调用）
        self._sync_init_lock = threading.Lock()
        # 按名称缓存 asyncio.Lock，延迟到首次事件循环访问时创建
        self._async_locks: dict[str, asyncio.Lock] = {}

    # ── asyncio.Lock 管理 ────────────────────────────────────────────────────

    def _get_async_lock(self, name: str) -> asyncio.Lock:
        """
        按名称延迟创建并缓存 asyncio.Lock。

        :param name: 锁名称，如 "init"、"db_write"、"rw"
        :return: 对应的 asyncio.Lock 实例
        """
        if name not in self._async_locks:
            self._async_locks[name] = asyncio.Lock()
        return self._async_locks[name]

    # ── 懒加载辅助 ───────────────────────────────────────────────────────────

    def _sync_lazy_init(
        self,
        check_fn: Callable[[], bool],
        init_fn: Callable[[], None],
    ) -> None:
        """
        线程安全双重检查懒加载（同步版）。

        :param check_fn: 返回 True 表示已初始化，无需再执行 init_fn
        :param init_fn:  实际初始化逻辑，仅在未初始化时调用一次
        """
        if check_fn():
            return
        with self._sync_init_lock:
            if check_fn():
                return
            init_fn()

    async def _async_lazy_init(
        self,
        lock_name: str,
        check_fn: Callable[[], bool],
        init_fn: Callable[[], None],
    ) -> None:
        """
        线程安全双重检查懒加载（asyncio 版）。

        init_fn 会在线程池中执行（不阻塞事件循环），且保证只执行一次。

        :param lock_name: 使用 _get_async_lock(lock_name) 作为初始化锁
        :param check_fn:  返回 True 表示已初始化，无需再执行 init_fn
        :param init_fn:   实际初始化逻辑（同步阻塞，在线程池中执行）
        """
        if check_fn():
            return
        async with self._get_async_lock(lock_name):
            if check_fn():
                return
            await run_sync(init_fn)
