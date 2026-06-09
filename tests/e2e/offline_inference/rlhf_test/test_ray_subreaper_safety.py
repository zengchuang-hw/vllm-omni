# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for Ray actor subreaper safety with parallel stage initialization.

Verifies that worker processes spawned during stage initialization are not
killed by Ray's actor subreaper when the spawning thread exits.

Root cause (PR #3533 / #3641): ``prctl(PR_SET_PDEATHSIG, SIGTERM)`` fires on
Linux when the **spawning thread** dies, not just the parent process. When
diffusion workers were submitted to a scoped ``ThreadPoolExecutor``, the
executor thread would die at the end of the ``with`` block, causing the
workers to be ``SIGTERM``'d (exit code 143).

The fix: diffusion replicas are launched **inline on the orchestrator thread**
(a long-lived daemon thread) so their ``PDEATHSIG`` never fires. LLM replicas
are submitted to an **engine-level** ``ThreadPoolExecutor`` whose threads also
outlive the init phase.

Two test levels:

1. ``test_minimal_scoped_spawn_reaper_safety`` — bare subprocess spawn from a
   scoped executor thread; reproduces the bug by verifying the child dies with
   exit code 143 when the spawning thread exits.
2. ``test_full_engine_reaper_safety`` — starts ``AsyncOmniEngine`` with a
   multi-stage pipeline; verifies all worker processes remain alive after
   engine initialization completes.
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import logging
import multiprocessing
import os
import platform
import signal
import time
from typing import Any

import pytest

from vllm.utils.system_utils import get_mp_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimal (process-level) test helpers
# ---------------------------------------------------------------------------

_SIGNAL_EXIT_BASE = 128
_CHILD_SLEEP_S = 30


def _signal_exit_code(signum: int) -> int:
    """Conventional process exit code for signal-driven exits."""
    return _SIGNAL_EXIT_BASE + signum


def _set_death_signal(signum: int) -> None:
    """Mirrors :func:`vllm_omni.engine.stage_init_utils.set_death_signal`.

    Installs ``PR_SET_PDEATHSIG`` so the kernel delivers *signum* when the
    spawning thread dies.
    """
    if platform.system() != "Linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        # PR_SET_PDEATHSIG = 1
        libc.prctl(1, signum)
    except Exception:  # noqa: BLE001
        # libc.so.6 may not be available (musl, alpine) — best-effort.
        pass


def _long_running_child(sleep_s: int = _CHILD_SLEEP_S) -> None:
    """Simple child process that installs ``PDEATHSIG(SIGTERM)`` then sleeps.

    Mirrors the behaviour of :class:`StageEngineCoreProc` which calls
    ``set_death_signal(signal.SIGTERM)`` at startup.
    """
    _set_death_signal(signal.SIGTERM)
    try:
        time.sleep(sleep_s)
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# Minimal test
# ---------------------------------------------------------------------------

@pytest.mark.hardware
class TestRaySubreaperMinimal:
    """Process-level tests that reproduce the PDEATHSIG / thread-death bug.

    These tests do **not** require a GPU, model weights, or the full engine.
    """

    def test_scoped_executor_thread_death_triggers_pdeathsig(self):
        """Child spawned from scoped executor thread dies with exit code 143.

        The scoped executor thread is destroyed when the ``with`` block exits.
        On Linux, this triggers ``prctl(PR_SET_PDEATHSIG, SIGTERM)`` in the
        child, resulting in exit code 143 (128 + 15).
        """
        if platform.system() != "Linux":
            pytest.skip("PDEATHSIG behaviour is Linux-specific")

        ctx = get_mp_context()
        shared: list[multiprocessing.Process] = []

        def _spawner() -> None:
            """Spawn a child from **this** executor thread.

            The child's clone-parent is the executor thread, so when this
            thread exits at the end of the ``with`` block the child receives
            ``SIGTERM`` via ``PDEATHSIG``.
            """
            proc = ctx.Process(
                target=_long_running_child,
                kwargs={"sleep_s": _CHILD_SLEEP_S},
            )
            proc.start()
            shared.append(proc)

        # Spawn inside a **scoped** ThreadPoolExecutor.
        # When the 'with' block exits the executor's threads are destroyed.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="scoped-init",
        ) as executor:
            executor.submit(_spawner).result()

        # Executor thread has exited — PDEATHSIG should have been delivered.
        assert shared, "Child process was not spawned"
        proc = shared[0]

        # Wait for the signal to be delivered (kernel may need a few ticks).
        deadline = time.monotonic() + 10
        while proc.is_alive() and time.monotonic() < deadline:
            time.sleep(0.1)

        assert not proc.is_alive(), (
            f"Child (pid={proc.pid}) should have been killed by PDEATHSIG "
            f"after the spawning thread exited, but it is still alive"
        )
        assert proc.exitcode == _signal_exit_code(signal.SIGTERM), (
            f"Expected exit code {_signal_exit_code(signal.SIGTERM)} (128 + SIGTERM), "
            f"got {proc.exitcode}"
        )

    def test_scoped_executor_single_thread_no_pdeathsig(self):
        """Child spawned from *main* thread survives after scoped executor.

        When the child is spawned from the main thread (not the executor
        thread), the main thread lives forever, so PDEATHSIG never fires.
        This is the baseline that the fix relies on.
        """
        if platform.system() != "Linux":
            pytest.skip("PDEATHSIG behaviour is Linux-specific")

        ctx = get_mp_context()

        # Spawn from the main thread (which is long-lived).
        proc = ctx.Process(
            target=_long_running_child,
            kwargs={"sleep_s": _CHILD_SLEEP_S},
        )
        proc.start()

        try:
            # Create + destroy a scoped executor in the meantime — this should
            # not affect the child because the child's clone-parent is the
            # main thread, not the executor thread.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="scoped-init",
            ) as executor:
                future = executor.submit(time.sleep, 0.5)
                future.result()

            # Child should still be alive.
            time.sleep(1.0)
            assert proc.is_alive(), (
                f"Child (pid={proc.pid}) died unexpectedly despite being "
                f"spawned from the main thread. Exit code: {proc.exitcode}"
            )
        finally:
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)

    def test_long_lived_executor_thread_survival(self):
        """Child spawned from long-lived executor thread survives.

        This validates the fix: the engine-level executor keeps threads alive
        for the engine lifetime, so PDEATHSIG never fires and the child
        processes survive.
        """
        if platform.system() != "Linux":
            pytest.skip("PDEATHSIG behaviour is Linux-specific")

        ctx = get_mp_context()
        proc: multiprocessing.Process | None = None

        # Create a long-lived executor — threads survive the full test.
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="engine-init",
        )

        def _spawner() -> multiprocessing.Process:
            proc = ctx.Process(
                target=_long_running_child,
                kwargs={"sleep_s": _CHILD_SLEEP_S},
            )
            proc.start()
            return proc

        try:
            future = executor.submit(_spawner)
            proc = future.result()

            # Child should survive well past the scoped-executor scenario.
            time.sleep(5)
            assert proc.is_alive(), (
                f"Child (pid={proc.pid}) died unexpectedly. "
                f"Exit code: {proc.exitcode}. The engine-level executor "
                f"should keep threads alive."
            )
        finally:
            if proc is not None and proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5)
            executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Full engine test
# ---------------------------------------------------------------------------

# Deploy YAML inline (minimal multi-stage config matching HunyuanImage-3
# AR+DiT pipeline). Uses the authoritative pipeline name and only the fields
# the engine actually requires for startup.
_HUNYUAN_IMAGE3_DEPLOY_YAML = """
pipeline: hunyuan_image_3_moe
async_chunk: false
trust_remote_code: true

connectors:
  shared_memory_connector:
    name: SharedMemoryConnector

stages:
  - stage_id: 0
    final_output: true
    final_output_type: text
    max_num_seqs: 1
    gpu_memory_utilization: 0.9
    enforce_eager: true
    devices: "4,5"
    tensor_parallel_size: 2
    default_sampling_params:
      temperature: 0.0
      max_tokens: 8192

  - stage_id: 1
    max_num_seqs: 1
    enforce_eager: true
    devices: "6,7"
    distributed_executor_backend: mp
    parallel_config:
      tensor_parallel_size: 2
      enable_expert_parallel: true
    default_sampling_params:
      num_inference_steps: 50
      guidance_scale: 0

edges:
  - from: 0
    to: 1
"""


@pytest.mark.hardware
class TestRaySubreaperFullEngine:
    """Full-engine tests that verify worker process survival after startup.

    These tests require a GPU environment with HunyuanImage-3.0 model weights
    available in the configured ``HF_HOME``.
    """

    @pytest.fixture
    def deploy_yaml_path(self, tmp_path: Any) -> str:
        """Write the inline deploy YAML to a temporary file."""
        yaml_path = tmp_path / "hunyuan_image3_test.yaml"
        yaml_path.write_text(_HUNYUAN_IMAGE3_DEPLOY_YAML)
        return str(yaml_path)

    def test_engine_startup_workers_survive(self, deploy_yaml_path: str) -> None:
        """Start AsyncOmniEngine and verify worker processes survive.

        The engine is started with a multi-stage AR+DiT pipeline. After
        initialization completes, we verify that all worker processes
        are still alive — confirming that neither PDEATHSIG nor Ray's
        actor subreaper killed them during init.
        """
        if platform.system() != "Linux":
            pytest.skip("PDEATHSIG behaviour is Linux-specific")

        # Prerequisites check
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        if not os.path.isdir(hf_home):
            pytest.skip(f"HF_HOME directory not accessible: {hf_home}")

        model_path = os.path.join(
            hf_home, "hub", "models--tencent--HunyuanImage-3.0-Instruct"
        )
        if not os.path.isdir(model_path):
            pytest.skip(
                f"Model not found: {model_path}. "
                f"Download the model first or set HF_HOME correctly."
            )

        # Check GPU availability
        try:
            import torch
            if torch.cuda.device_count() < 4:
                pytest.skip(
                    f"Test requires 4 GPUs (AR TP2 + DiT TP2), "
                    f"found {torch.cuda.device_count()}"
                )
        except ImportError:
            pytest.skip("torch not available")

        # Snapshot active children BEFORE engine init so we can tell which
        # processes belong to the engine.
        before_pids = {p.pid for p in multiprocessing.active_children()}

        from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

        engine: AsyncOmniEngine | None = None
        try:
            engine = AsyncOmniEngine(
                model="tencent/HunyuanImage-3.0-Instruct",
                stage_configs_path=deploy_yaml_path,
                single_stage_mode=False,
                stage_init_timeout=600,
            )

            # Give worker processes a moment to stabilize after init.
            time.sleep(5)

            # Find engine-owned child processes (spawned after engine init).
            engine_children = [
                p for p in multiprocessing.active_children()
                if p.pid not in before_pids
            ]

            assert engine_children, (
                "No engine worker processes found — "
                "multiprocessing.active_children() is empty"
            )

            alive = [p for p in engine_children if p.is_alive()]
            assert len(alive) == len(engine_children), (
                f"Some worker processes died during/after init: "
                f"{[(p.pid, p.exitcode) for p in engine_children if not p.is_alive()]}"
            )

            # Additional survival check: wait 10s and verify again.
            time.sleep(10)

            still_alive = [p for p in engine_children if p.is_alive()]
            assert len(still_alive) == len(engine_children), (
                f"Worker processes died within 10s of startup: "
                f"{[(p.pid, p.exitcode) for p in engine_children if not p.is_alive()]}"
            )

            logger.info(
                "✅ %s worker processes survived 10 seconds. "
                "LLM reaper-safe under Ray.",
                len(still_alive),
            )

        finally:
            if engine is not None:
                engine.shutdown()
