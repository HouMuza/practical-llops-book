from __future__ import annotations
 
from dataclasses import dataclass
 
import torch
from opentelemetry import trace
from transformers import AutoModelForCausalLM, AutoTokenizer
 
from engine.sampling import sample_next_token
from engine.types import SamplingParams
 
tracer = trace.get_tracer(__name__)
 
 
@dataclass(slots=True)
class SpeculativeResult:
    accepted_token_ids: list[int]
    target_calls: int
    draft_calls: int
    acceptance_rate: float
 
 
class DraftModelSpeculator:
    """Draft-model speculative decoding.
 
    The draft proposes k tokens cheaply. The target verifies them in one forward pass. This
    class is deliberately independent of the main engine so speculative decoding can be
    switched on/off by configuration.
    """
 
    def __init__(self, draft_model_name: str, device: str = "cuda") -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(draft_model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
 
    @torch.inference_mode()
    def propose(
        self,
        input_ids: torch.Tensor,
        sampling: SamplingParams,
        k: int,
    ) -> list[int]:
        with tracer.start_as_current_span("speculative.propose") as span:
            ids = input_ids.clone()
            proposed: list[int] = []
            generator = torch.Generator(device=ids.device)
            if sampling.seed is not None:
                generator.manual_seed(sampling.seed)
            for _ in range(k):
                out = self.model(input_ids=ids, use_cache=True)
                next_id = sample_next_token(out.logits[:, -1, :], sampling, proposed, generator)
                proposed.append(next_id)
                ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device)], dim=1)
            span.set_attribute("speculative.proposed", len(proposed))
            return proposed
 
 
@torch.inference_mode()
def verify_with_target(
    target_model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    proposed_ids: list[int],
) -> list[int]:
    """Greedy verification helper.
 
    For exact speculative sampling you compare target probabilities token by token. This
    simplified verifier accepts a proposed token when it matches target argmax. It is useful
    for operational wiring and acceptance-rate monitoring.
    """
    with tracer.start_as_current_span("speculative.verify") as span:
        if not proposed_ids:
            return []
        proposed = torch.tensor([proposed_ids], device=input_ids.device, dtype=input_ids.dtype)
        full = torch.cat([input_ids, proposed], dim=1)
        out = target_model(input_ids=full, use_cache=True)
        logits = out.logits[:, input_ids.shape[1] - 1 : -1, :]
        target_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
        accepted: list[int] = []
        for draft_id, target_id in zip(proposed_ids, target_ids):
            if int(draft_id) == int(target_id):
                accepted.append(int(draft_id))
            else:
                accepted.append(int(target_id))
                break
        span.set_attribute("speculative.accepted", len(accepted))
        span.set_attribute("speculative.acceptance_rate", len(accepted) / max(1, len(proposed_ids)))
        return accepted
