from __future__ import annotations

import math
import random
from typing import Any

from ecr_grpo.policies import format_agent_prompt
from ecr_grpo.types import PolicyAction, StepRecord


class HFLoraPolicy:
    """HuggingFace causal-LM policy with an optional LoRA adapter.

    This class is intentionally compact. It provides the pieces this project needs:

    - text-action generation
    - old logprob collection for the selected action
    - clipped GRPO-style policy update over step-level advantages

    Heavy distributed rollout/training can be swapped in later without changing the
    ECR buffer and credit-refill code.
    """

    def __init__(
        self,
        *,
        action_space: list[str],
        model_id: str,
        adapter_path: str | None = None,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        device: str | None = None,
        max_new_tokens: int = 8,
        temperature: float = 0.7,
        top_p: float = 1.0,
        clip_eps: float = 0.2,
        grad_accum_steps: int = 1,
        seed: int = 0,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HFLoraPolicy requires optional dependencies. Install with "
                "`pip install -e .[hf]`."
            ) from exc

        self.torch = torch
        self.rng = random.Random(seed)
        self.action_space = list(action_space)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.clip_eps = clip_eps
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else None
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.to(self.device)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        elif use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as exc:
                raise RuntimeError("LoRA requires `peft`. Install with `pip install -e .[hf]`.") from exc
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_cfg)

        self.model.train()
        self.optimizer = None

    def act(
        self,
        observation: str,
        action_space: list[str] | None = None,
        *,
        greedy: bool = False,
    ) -> PolicyAction:
        torch = self.torch
        actions = action_space or self.action_space
        prompt = format_agent_prompt(observation, actions)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                prompt_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=not greedy,
                temperature=max(self.temperature, 1e-6),
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        raw_response_ids = generated[0, prompt_ids.shape[1] :].detach().cpu().tolist()
        raw_text = self.tokenizer.decode(raw_response_ids, skip_special_tokens=True)
        action = self._parse_action(raw_text, actions)
        response_ids = self._encode_response(action)
        with torch.no_grad():
            old_logprob = float(
                self._sequence_logprob(prompt_ids[0].detach().cpu().tolist(), response_ids).detach().cpu()
            )
        return PolicyAction(
            text=action,
            old_logprob=old_logprob,
            prompt_ids=prompt_ids[0].detach().cpu().tolist(),
            response_ids=response_ids,
        )

    def update(self, steps: list[StepRecord], lr: float) -> dict[str, float]:
        if not steps:
            return {"policy_loss": 0.0, "entropy": 0.0}
        if self.optimizer is None:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = self.torch.optim.AdamW(trainable, lr=lr)

        torch = self.torch
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_ratio = 0.0

        for idx, step in enumerate(steps, start=1):
            prompt_ids = step.prompt_ids or self._encode_prompt(step.observation, step.action_space)
            response_ids = step.response_ids or self._encode_response(step.action)
            new_logprob = self._sequence_logprob(prompt_ids, response_ids)
            old_logprob = torch.tensor(step.old_logprob, device=self.device, dtype=new_logprob.dtype)
            advantage = torch.tensor(step.advantage, device=self.device, dtype=new_logprob.dtype)
            ratio = torch.exp(new_logprob - old_logprob).clamp(0.0, 10.0)
            clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            loss = -torch.minimum(ratio * advantage, clipped * advantage)
            (loss / self.grad_accum_steps).backward()
            total_loss += float(loss.detach().cpu())
            total_ratio += float(ratio.detach().cpu())
            if idx % self.grad_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        if len(steps) % self.grad_accum_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        denom = max(1, len(steps))
        return {"policy_loss": total_loss / denom, "mean_ratio": total_ratio / denom, "entropy": 0.0}

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def _sequence_logprob(self, prompt_ids: list[int], response_ids: list[int]):
        torch = self.torch
        ids = torch.tensor([prompt_ids + response_ids], device=self.device, dtype=torch.long)
        outputs = self.model(input_ids=ids)
        logits = outputs.logits[:, :-1, :]
        targets = ids[:, 1:]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        start = max(0, len(prompt_ids) - 1)
        return token_logprobs[:, start:].sum()

    def _encode_prompt(self, observation: str, action_space: list[str]) -> list[int]:
        prompt = format_agent_prompt(observation, action_space)
        return self.tokenizer(prompt, add_special_tokens=True).input_ids

    def _encode_response(self, action: str) -> list[int]:
        suffix = self.tokenizer.eos_token or ""
        ids = self.tokenizer(action + suffix, add_special_tokens=False).input_ids
        if not ids:
            ids = [self.tokenizer.eos_token_id]
        return ids

    def _parse_action(self, text: str, actions: list[str]) -> str:
        first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
        lowered = first_line.lower()
        for action in actions:
            if lowered == action.lower():
                return action
        for action in actions:
            if action.lower() in lowered:
                return action
        return actions[0] if actions else first_line
