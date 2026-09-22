"""PTQ helpers for MASt3R GFM backbone (Track A, desktop accuracy).

Scope (CLAUDE.md): compress GFM forward only. Classical SLAM back-end stays FP32.

Default W8 uses *fake* int8 weight-only (quantize→dequantize into the same dtype).
That is picklable under MASt3R-SLAM's multiprocessing and matches the plan's Track-A
accuracy path. Real packed kernels (torchao Int8Tensor) are optional via method=.

W8A8 replaces trunk Linear with FakeIntLinear (weight fake-quant + dynamic act fake-quant).
FP8 uses FakeFP8Linear (e4m3 cast via scale). Must be importable under spawn — keep on PYTHONPATH.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger(__name__)

# e4m3 finite max (~448); used for dynamic scaling into float8_e4m3fn
_FP8_E4M3_MAX = 448.0


def _is_head_module(fqn: str) -> bool:
    f = fqn.lower()
    return "downstream_head" in f or f.startswith("head")


def _is_linearish(module: nn.Module) -> bool:
    return isinstance(module, (nn.Linear, FakeIntLinear, FakeFP8Linear))


def filter_trunk_linear(module: nn.Module, fqn: str) -> bool:
    if not _is_linearish(module):
        return False
    if _is_head_module(fqn):
        return False
    return True


def filter_encoder_linear(module: nn.Module, fqn: str) -> bool:
    if not _is_linearish(module):
        return False
    if _is_head_module(fqn):
        return False
    f = fqn.lower()
    return "enc_blocks" in f or "patch_embed" in f or f.startswith("enc_")

def _qrange(bits: int) -> tuple[int, int, int]:
    assert bits in (4, 8)
    qmax = (1 << (bits - 1)) - 1  # 127 or 7
    qmin = -qmax - 1 if bits == 8 else -qmax  # [-128,127] or [-7,7]
    return qmin, qmax, qmax


def _fake_int_weight_only_linear(linear: nn.Linear, bits: int) -> None:
    """Per-channel symmetric fake-quant on weight; bias untouched. In-place."""
    qmin, qmax, _ = _qrange(bits)
    w = linear.weight.data
    max_abs = w.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = max_abs / float(qmax)
    q = (w / scale).round().clamp(qmin, qmax)
    linear.weight.data.copy_((q * scale).to(dtype=w.dtype))


def fake_quant_activation(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Dynamic per-token (last-dim) symmetric fake-quant."""
    qmin, qmax, _ = _qrange(bits)
    max_abs = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = max_abs / float(qmax)
    return (x / scale).round().clamp(qmin, qmax) * scale


class FakeIntLinear(nn.Module):
    """Picklable Linear stand-in: static weight fake-quant + dynamic act fake-quant."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        weight_bits: int = 8,
        act_bits: int = 8,
    ):
        super().__init__()
        self.weight_bits = int(weight_bits)
        self.act_bits = int(act_bits)
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.bias = (
            None
            if bias is None
            else nn.Parameter(bias.detach().clone(), requires_grad=False)
        )
        with torch.no_grad():
            qmin, qmax, _ = _qrange(self.weight_bits)
            max_abs = self.weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            scale = max_abs / float(qmax)
            q = (self.weight / scale).round().clamp(qmin, qmax)
            self.weight.data.copy_((q * scale).to(dtype=weight.dtype))

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, weight_bits: int = 8, act_bits: int = 8
    ) -> "FakeIntLinear":
        return cls(
            linear.weight.data,
            None if linear.bias is None else linear.bias.data,
            weight_bits=weight_bits,
            act_bits=act_bits,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = fake_quant_activation(x, bits=self.act_bits)
        return F.linear(x_q, self.weight, self.bias)


def fake_fp8_e4m3(x: torch.Tensor, dim: Optional[Union[int, tuple]] = -1) -> torch.Tensor:
    """Scaled cast through float8_e4m3fn → original dtype (Track A accuracy probe)."""
    if dim is None:
        amax = x.detach().abs().amax().clamp(min=1e-12)
        scale = amax / _FP8_E4M3_MAX
        return (x / scale).to(torch.float8_e4m3fn).to(dtype=x.dtype) * scale
    amax = x.detach().abs().amax(dim=dim, keepdim=True).clamp(min=1e-12)
    scale = amax / _FP8_E4M3_MAX
    return (x / scale).to(torch.float8_e4m3fn).to(dtype=x.dtype) * scale


class FakeFP8Linear(nn.Module):
    """Picklable Linear: static per-channel FP8-e4m3 weights + dynamic per-token FP8 acts."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        quantize_acts: bool = True,
    ):
        super().__init__()
        self.quantize_acts = bool(quantize_acts)
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.bias = (
            None
            if bias is None
            else nn.Parameter(bias.detach().clone(), requires_grad=False)
        )
        with torch.no_grad():
            self.weight.data.copy_(fake_fp8_e4m3(self.weight.data, dim=1))

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, quantize_acts: bool = True
    ) -> "FakeFP8Linear":
        return cls(
            linear.weight.data,
            None if linear.bias is None else linear.bias.data,
            quantize_acts=quantize_acts,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = fake_fp8_e4m3(x, dim=-1) if self.quantize_acts else x
        return F.linear(x_q, self.weight, self.bias)


def _scope_filter(scope: str) -> Callable[[nn.Module, str], bool]:
    if scope == "trunk":
        return filter_trunk_linear
    if scope == "encoder":
        return filter_encoder_linear
    raise ValueError(f"unknown scope={scope}")


def apply_int_weight_only_fake(
    model: nn.Module, scope: str = "trunk", bits: int = 8
) -> dict:
    filt = _scope_filter(scope)
    n = 0
    with torch.no_grad():
        for fqn, module in model.named_modules():
            if isinstance(module, nn.Linear) and filt(module, fqn):
                _fake_int_weight_only_linear(module, bits=bits)
                n += 1
    info = {
        "method": f"fake_int{bits}_weight_only_per_channel",
        "scope": scope,
        "bits": bits,
        "act_bits": 16,
        "n_linears_quantized": n,
    }
    log.info("PTQ applied: %s", info)
    return info


def apply_int_wa_fake(
    model: nn.Module,
    scope: str = "trunk",
    weight_bits: int = 8,
    act_bits: int = 8,
) -> dict:
    """Replace matching Linear with FakeIntLinear (W + A fake-quant)."""
    filt = _scope_filter(scope)
    targets = [
        fqn
        for fqn, module in model.named_modules()
        if isinstance(module, nn.Linear) and filt(module, fqn)
    ]
    for fqn in targets:
        parts = fqn.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old = getattr(parent, parts[-1])
        setattr(
            parent,
            parts[-1],
            FakeIntLinear.from_linear(old, weight_bits=weight_bits, act_bits=act_bits),
        )
    info = {
        "method": f"fake_w{weight_bits}a{act_bits}_per_channel_w_per_token_a",
        "scope": scope,
        "bits": weight_bits,
        "act_bits": act_bits,
        "n_linears_quantized": len(targets),
    }
    log.info("PTQ applied: %s", info)
    return info


def apply_int8_weight_only_fake(model: nn.Module, scope: str = "trunk") -> dict:
    return apply_int_weight_only_fake(model, scope=scope, bits=8)


def apply_fp8_fake(
    model: nn.Module, scope: str = "trunk", quantize_acts: bool = True
) -> dict:
    """Replace matching Linear with FakeFP8Linear (e4m3 W+A by default)."""
    filt = _scope_filter(scope)
    targets = [
        fqn
        for fqn, module in model.named_modules()
        if isinstance(module, nn.Linear) and filt(module, fqn)
    ]
    for fqn in targets:
        parts = fqn.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old = getattr(parent, parts[-1])
        setattr(
            parent,
            parts[-1],
            FakeFP8Linear.from_linear(old, quantize_acts=quantize_acts),
        )
    info = {
        "method": "fake_fp8_e4m3_per_channel_w_per_token_a"
        if quantize_acts
        else "fake_fp8_e4m3_weight_only_per_channel",
        "scope": scope,
        "bits": 8,
        "act_bits": 8 if quantize_acts else 16,
        "dtype": "float8_e4m3fn",
        "n_linears_quantized": len(targets),
    }
    log.info("PTQ applied: %s", info)
    return info


def apply_int8_weight_only_torchao(model: nn.Module, scope: str = "trunk") -> dict:
    """Real torchao Int8Tensor packing. Not picklable for MASt3R-SLAM mp."""
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    filt = filter_trunk_linear if scope == "trunk" else filter_encoder_linear
    quantize_(model, Int8WeightOnlyConfig(), filter_fn=filt)
    return {"method": "torchao.Int8WeightOnlyConfig", "scope": scope}


def estimate_weight_bytes(model: nn.Module) -> dict:
    param_b = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_b = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "param_bytes": param_b,
        "buffer_bytes": buf_b,
        "total_bytes": param_b + buf_b,
        "total_mb": (param_b + buf_b) / 1e6,
    }


def logical_int_mb(model: nn.Module, scope: str = "trunk", bits: int = 8) -> float:
    """Approx size if trunk Linears were stored at `bits` (rest keep current dtype)."""
    filt = _scope_filter(scope)
    bytes_per = bits / 8.0
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    saved = 0.0
    for fqn, module in model.named_modules():
        if filt(module, fqn) and hasattr(module, "weight"):
            saved += module.weight.numel() * (module.weight.element_size() - bytes_per)
    return (total - saved) / 1e6


def patch_load_mast3r(
    scope: str = "trunk",
    method: str = "fake",
    bits: int = 8,
    act_bits: Optional[int] = None,
) -> None:
    """Monkeypatch mast3r_slam.mast3r_utils.load_mast3r to apply PTQ after load."""
    import mast3r_slam.mast3r_utils as mu

    orig = mu.load_mast3r

    def load_mast3r_quant(path=None, device="cuda"):
        model = orig(path=path, device=device)
        if method in ("fake_fp8", "fp8"):
            info = apply_fp8_fake(model, scope=scope, quantize_acts=True)
            logical_bits = 8
        elif method == "fake_fp8_wo":
            info = apply_fp8_fake(model, scope=scope, quantize_acts=False)
            logical_bits = 8
        elif method == "fake":
            if act_bits is not None and act_bits <= 8:
                info = apply_int_wa_fake(
                    model, scope=scope, weight_bits=bits, act_bits=act_bits
                )
            else:
                info = apply_int_weight_only_fake(model, scope=scope, bits=bits)
            logical_bits = bits
        elif method == "torchao":
            if bits != 8:
                raise ValueError("torchao path currently wired for int8 only")
            info = apply_int8_weight_only_torchao(model, scope=scope)
            logical_bits = 8
        else:
            raise ValueError(method)
        info["logical_mb"] = logical_int_mb(model, scope=scope, bits=logical_bits)
        info["fp16_mb"] = estimate_weight_bytes(model)["total_mb"]
        model._gfm_ptq_info = info  # type: ignore[attr-defined]
        print(f"[PTQ] applied {info}")
        return model

    mu.load_mast3r = load_mast3r_quant
