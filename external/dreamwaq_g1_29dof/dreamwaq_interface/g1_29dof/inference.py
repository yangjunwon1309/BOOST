"""TorchScript-only FlashSAC CENet plus lower-body actor inference."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np

from .contract import DreamWaQ29Contract


@dataclass(frozen=True)
class InferenceResult:
    v_hat: np.ndarray
    z: np.ndarray
    actor_input: np.ndarray
    policy_mean: np.ndarray
    lower_action: np.ndarray
    latency_ns: int


@dataclass(frozen=True)
class ActionInferenceResult:
    """Minimal allocation result used by the real-time policy worker."""

    lower_action: np.ndarray
    latency_ns: int


class TorchScriptDreamWaQ:
    """Load sealed deployment artifacts; never reconstruct training code."""

    def __init__(
        self,
        contract: DreamWaQ29Contract,
        *,
        device: str = "cpu",
        cpu_threads: int = 1,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for DreamWaQ TorchScript inference") from exc
        self.torch = torch
        self.contract = contract
        self.device = torch.device(str(device))
        self.clock_ns = clock_ns
        if self.device.type == "cpu":
            threads = int(cpu_threads)
            if threads <= 0:
                raise ValueError("cpu_threads must be positive")
            torch.set_num_threads(threads)
            # Match HOMIE's ONNX intra/inter-op=1 policy.  Without this,
            # TorchScript may retain a multi-threaded inter-op pool even when
            # intra-op work is limited to one thread, creating scheduler
            # bursts on the isolated policy CPU.
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                # A caller may have initialized the global Torch pool before
                # constructing this backend.  In that case keep the existing
                # pool rather than failing deployment after the process has
                # already started.
                pass
        try:
            self.policy = torch.jit.load(str(contract.policy_path), map_location=self.device).eval()
            self.cenet = torch.jit.load(str(contract.cenet_norm_path), map_location=self.device).eval()
        except Exception as exc:
            raise RuntimeError("cannot load sealed DreamWaQ TorchScript artifacts") from exc
        self._build_action_model()
        self._validate_network_contract()

    def _build_action_model(self) -> None:
        """Inline CENet and actor into one frozen graph for the control loop.

        The public ``infer`` method retains every intermediate tensor for trace
        verification.  The hardware worker needs only the bounded action, so it
        uses this equivalent graph and avoids five CPU copies plus the dispatch
        boundary between the two sealed TorchScript modules.
        """

        torch = self.torch
        contract = self.contract

        class _CombinedAction(torch.nn.Module):
            def __init__(self, cenet, policy, velocity_scale: float) -> None:
                super().__init__()
                self.cenet = cenet
                self.policy = policy
                self.velocity_scale = float(velocity_scale)

            def forward(self, current, history):
                output = self.cenet(history)
                v_hat = output[0].reshape(1, -1)
                z = output[1].reshape(1, -1)
                actor_current = current.clone()
                actor_current[:, :3] = v_hat * self.velocity_scale
                actor_input = torch.cat((actor_current, z), dim=-1)
                policy_mean = self.policy(actor_input).reshape(1, -1)
                return torch.tanh(policy_mean)

        self._action_current = torch.empty(
            (1, contract.observation_dim), dtype=torch.float32, device=self.device
        )
        self._action_history = torch.empty(
            (1, contract.history_dim), dtype=torch.float32, device=self.device
        )
        combined = _CombinedAction(
            self.cenet, self.policy, contract.velocity_obs_scale
        ).eval()
        try:
            with torch.inference_mode():
                traced = torch.jit.trace(
                    combined,
                    (self._action_current.zero_(), self._action_history.zero_()),
                    check_trace=True,
                )
                self._action_model = torch.jit.freeze(traced.eval())
        except Exception as exc:
            raise RuntimeError("cannot build fused DreamWaQ action graph") from exc

        if self.device.type == "cpu":
            self._action_current_numpy = self._action_current.numpy().reshape(-1)
            self._action_history_numpy = self._action_history.numpy().reshape(-1)
        else:
            self._action_current_numpy = None
            self._action_history_numpy = None

    @staticmethod
    def _array(values: Any, size: int, label: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32).reshape(-1)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError("%s must contain exactly %d finite values" % (label, size))
        return result

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _forward(self, policy_current: np.ndarray, obs_history: np.ndarray) -> InferenceResult:
        torch = self.torch
        current_t = torch.as_tensor(policy_current, dtype=torch.float32, device=self.device).unsqueeze(0)
        history_t = torch.as_tensor(obs_history, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._synchronize()
        started = int(self.clock_ns())
        with torch.inference_mode():
            output = self.cenet(history_t)
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError("normalized CENet export must return (v_hat, z)")
            v_hat = output[0].reshape(1, -1)
            z = output[1].reshape(1, -1)
            actor_current = current_t.clone()
            actor_current[:, :3] = v_hat * float(self.contract.velocity_obs_scale)
            actor_input = torch.cat((actor_current, z), dim=-1)
            policy_mean = self.policy(actor_input).reshape(1, -1)
            lower_action = torch.tanh(policy_mean)
        self._synchronize()
        finished = int(self.clock_ns())

        def cpu(value: Any) -> np.ndarray:
            return value.detach().float().cpu().numpy().reshape(-1).astype(np.float32, copy=True)

        result = InferenceResult(
            v_hat=cpu(v_hat),
            z=cpu(z),
            actor_input=cpu(actor_input),
            policy_mean=cpu(policy_mean),
            lower_action=cpu(lower_action),
            latency_ns=finished - started,
        )
        expected = {
            "v_hat": (result.v_hat, 3),
            "z": (result.z, self.contract.latent_dim),
            "actor_input": (result.actor_input, self.contract.actor_input_dim),
            "policy_mean": (result.policy_mean, self.contract.action_dim),
            "lower_action": (result.lower_action, self.contract.action_dim),
        }
        for name, (value, size) in expected.items():
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise RuntimeError("DreamWaQ %s output violates the sealed dimension/finite contract" % name)
        if float(np.max(np.abs(result.lower_action))) > 1.0 + 1.0e-6:
            raise RuntimeError("tanh deployment action escaped [-1, 1]")
        return result

    def _forward_action(
        self, policy_current: np.ndarray, obs_history: np.ndarray
    ) -> ActionInferenceResult:
        torch = self.torch
        if self.device.type == "cpu":
            np.copyto(self._action_current_numpy, policy_current)
            np.copyto(self._action_history_numpy, obs_history)
        else:
            self._action_current.copy_(
                torch.as_tensor(policy_current, dtype=torch.float32, device=self.device)
            )
            self._action_history.copy_(
                torch.as_tensor(obs_history, dtype=torch.float32, device=self.device)
            )
        self._synchronize()
        started = int(self.clock_ns())
        with torch.inference_mode():
            lower_action = self._action_model(
                self._action_current, self._action_history
            )
        self._synchronize()
        finished = int(self.clock_ns())
        value = (
            lower_action.detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float32, copy=True)
        )
        if value.shape != (self.contract.action_dim,) or not np.all(np.isfinite(value)):
            raise RuntimeError("DreamWaQ lower action violates the sealed dimension/finite contract")
        if float(np.max(np.abs(value))) > 1.0 + 1.0e-6:
            raise RuntimeError("tanh deployment action escaped [-1, 1]")
        return ActionInferenceResult(value, finished - started)

    def _validate_network_contract(self) -> None:
        current = np.linspace(
            -0.25, 0.25, self.contract.observation_dim, dtype=np.float32
        )
        history = np.linspace(
            0.25, -0.25, self.contract.history_dim, dtype=np.float32
        )
        result = self._forward(current, history)
        if result.actor_input.shape != (self.contract.actor_input_dim,):
            raise RuntimeError("policy actor input dimension differs from the export manifest")
        action = self._forward_action(current, history)
        if not np.allclose(
            result.lower_action, action.lower_action, rtol=1.0e-5, atol=2.0e-6
        ):
            raise RuntimeError("fused DreamWaQ action graph differs from the sealed modules")

    def warmup(self, iterations: int = 10) -> None:
        current = np.zeros(self.contract.observation_dim, dtype=np.float32)
        history = np.zeros(self.contract.history_dim, dtype=np.float32)
        for _ in range(max(1, int(iterations))):
            self._forward_action(current, history)

    def infer(self, obs_frame: Any, obs_history: Any) -> InferenceResult:
        return self._forward(
            self._array(obs_frame, self.contract.observation_dim, "FlashSAC policy current"),
            self._array(obs_history, self.contract.history_dim, "obs_history"),
        )

    def infer_action(self, obs_frame: Any, obs_history: Any) -> ActionInferenceResult:
        """Return only the action through the fused real-time graph."""

        return self._forward_action(
            self._array(obs_frame, self.contract.observation_dim, "FlashSAC policy current"),
            self._array(obs_history, self.contract.history_dim, "obs_history"),
        )


__all__ = ["ActionInferenceResult", "InferenceResult", "TorchScriptDreamWaQ"]
