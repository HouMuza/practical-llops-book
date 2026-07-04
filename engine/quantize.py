from __future__ import annotations
 
from dataclasses import dataclass
from typing import Iterable
 
import torch
from opentelemetry import trace
 
tracer = trace.get_tracer(__name__)
 
 
@dataclass(slots=True)
class QuantizedTensor:
    qweight: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor | None = None
    group_size: int | None = None
    bits: int = 8
    symmetric: bool = True
 
 
def symmetric_int8_quantize(weight: torch.Tensor, eps: float = 1e-8) -> QuantizedTensor:
    with tracer.start_as_current_span("quantize.symmetric_int8"):
        max_abs = weight.abs().amax(dim=1, keepdim=True).clamp_min(eps)
        scale = max_abs / 127.0
        q = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
        return QuantizedTensor(qweight=q, scale=scale.squeeze(1), bits=8, symmetric=True)
 
 
def dequantize_int8(qt: QuantizedTensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    if qt.bits != 8 or not qt.symmetric:
        raise ValueError("Expected symmetric INT8 QuantizedTensor")
    return (qt.qweight.float() * qt.scale[:, None].float()).to(dtype)
 
 
def groupwise_int4_quantize(weight: torch.Tensor, group_size: int = 128, eps: float = 1e-8) -> QuantizedTensor:
    """Symmetric groupwise INT4 quantisation.
 
    This stores the unpacked int4 values in int8 for readability. A production version packs
    two 4-bit values per byte and uses an int4-aware matmul kernel.
    """
    with tracer.start_as_current_span("quantize.groupwise_int4") as span:
        if weight.ndim != 2:
            raise ValueError("weight must be a 2-D matrix")
        out_dim, in_dim = weight.shape
        if in_dim % group_size != 0:
            pad = group_size - (in_dim % group_size)
            weight = torch.nn.functional.pad(weight, (0, pad))
            in_dim = weight.shape[1]
        groups = in_dim // group_size
        view = weight.view(out_dim, groups, group_size)
        max_abs = view.abs().amax(dim=2, keepdim=True).clamp_min(eps)
        scale = max_abs / 7.0
        q = torch.round(view / scale).clamp(-8, 7).to(torch.int8)
        span.set_attribute("quant.group_size", group_size)
        span.set_attribute("quant.groups", groups)
        return QuantizedTensor(
            qweight=q.view(out_dim, in_dim),
            scale=scale.squeeze(2),
            group_size=group_size,
            bits=4,
            symmetric=True,
        )
 
 
def calibrate_activation_scale(samples: Iterable[torch.Tensor], percentile: float = 0.999) -> torch.Tensor:
    """Choose an activation scale from calibration tensors."""
    with tracer.start_as_current_span("quantize.calibrate_activation_scale"):
        flat = torch.cat([x.detach().float().abs().flatten().cpu() for x in samples])
        q = torch.quantile(flat, percentile)
        return (q / 127.0).clamp_min(1e-8)
