from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from ecr_grpo.policies import format_agent_prompt
from ecr_grpo.types import PolicyAction, StepRecord


class HFLoraPolicy:
    """HuggingFace causal-LM policy with an optional LoRA adapter.

    This class is intentionally compact. It provides the pieces this project needs:

    - text-action generation
    - candidate-action scoring or text generation
    - old logprob collection for the selected candidate-action distribution
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
        action_selection: str = "score",
        action_score_batch_size: int = 8,
        action_score_normalization: str = "mean",
        action_score_calibration: str = "pmi",
        use_chat_template: bool = True,
        update_score_mode: str = "full_distribution",
        clip_eps: float = 0.2,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
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
        self.action_selection = action_selection.lower()
        self.action_score_batch_size = max(1, action_score_batch_size)
        self.action_score_normalization = action_score_normalization.lower()
        self.action_score_calibration = action_score_calibration.lower()
        self.use_chat_template = use_chat_template
        self.update_score_mode = update_score_mode.lower()
        self.clip_eps = clip_eps
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.max_grad_norm = max_grad_norm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.action_selection in {"candidate", "candidates", "scoring"}:
            self.action_selection = "score"
        if self.action_selection not in {"score", "generate"}:
            raise ValueError("HF action_selection must be 'score' or 'generate'")
        if self.update_score_mode not in {"selected", "full_distribution"}:
            raise ValueError("HF update_score_mode must be 'selected' or 'full_distribution'")
        if self.action_score_normalization not in {"sum", "mean", "sqrt"}:
            raise ValueError("HF action_score_normalization must be 'sum', 'mean', or 'sqrt'")
        if self.action_score_calibration not in {"none", "pmi"}:
            raise ValueError("HF action_score_calibration must be 'none' or 'pmi'")
        model_path = Path(model_id)
        if model_path.is_absolute() and not model_path.exists():
            raise FileNotFoundError(
                f"HF model path does not exist: {model_id}. "
                "Set MODEL_ID or ECR_GRPO_MODEL_ID to a valid local model path, "
                "or use a HuggingFace repo id such as Qwen/Qwen2.5-1.5B-Instruct."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else None
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if dtype is not None:
            kwargs["dtype"] = dtype
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except TypeError:
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
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
        self._response_id_cache: dict[str, list[int]] = {}
        self._prior_score_cache: dict[tuple[str, ...], Any] = {}

    def act(
        self,
        observation: str,
        action_space: list[str] | None = None,
        *,
        greedy: bool = False,
    ) -> PolicyAction:
        if self.action_selection == "generate":
            return self._act_generate(observation, action_space, greedy=greedy)
        return self._act_score(observation, action_space, greedy=greedy)

    def _act_score(
        self,
        observation: str,
        action_space: list[str] | None = None,
        *,
        greedy: bool = False,
    ) -> PolicyAction:
        torch = self.torch
        actions = action_space or self.action_space
        prompt_ids = self._encode_prompt(observation, actions)

        self.model.eval()
        with torch.no_grad():
            scores = self._candidate_scores_for_observation(observation, actions)
            log_probs, entropy = self._candidate_log_distribution_from_scores(scores)
        if greedy:
            action_idx = int(torch.argmax(log_probs).detach().cpu())
        else:
            probs = torch.exp(log_probs).detach().cpu().tolist()
            action_idx = self._sample_index(probs)
        action = actions[action_idx]
        return PolicyAction(
            text=action,
            old_logprob=float(self._old_logprob_for_action(scores, log_probs, action_idx).detach().cpu()),
            prompt_ids=prompt_ids,
            response_ids=self._encode_response(action),
        )

    def _act_generate(
        self,
        observation: str,
        action_space: list[str] | None = None,
        *,
        greedy: bool = False,
    ) -> PolicyAction:
        torch = self.torch
        actions = action_space or self.action_space
        prompt = format_agent_prompt(observation, actions)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)
        gen_kwargs: dict[str, Any] = {
            "input_ids": prompt_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": not greedy,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
        }
        if not greedy:
            gen_kwargs["temperature"] = max(self.temperature, 1e-6)
            gen_kwargs["top_p"] = self.top_p

        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(**gen_kwargs)
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
        total_entropy = 0.0

        for idx, step in enumerate(steps, start=1):
            prompt_ids = step.prompt_ids or self._encode_prompt(step.observation, step.action_space)
            if self.action_selection == "score":
                actions = step.action_space or self.action_space
                if self.update_score_mode == "selected":
                    response_ids = step.response_ids or self._encode_response(step.action)
                    new_logprob = self._sequence_score_for_observation(
                        step.observation,
                        actions,
                        response_ids,
                    )
                    entropy = self._candidate_entropy_no_grad(step.observation, actions)
                else:
                    new_logprob, entropy = self._candidate_action_logprob(
                        step.observation,
                        actions,
                        step.action,
                    )
            else:
                response_ids = step.response_ids or self._encode_response(step.action)
                new_logprob = self._sequence_logprob(prompt_ids, response_ids)
                entropy = self.torch.tensor(0.0, device=self.device, dtype=new_logprob.dtype)
            old_logprob = torch.tensor(step.old_logprob, device=self.device, dtype=new_logprob.dtype)
            advantage = torch.tensor(step.advantage, device=self.device, dtype=new_logprob.dtype)
            ratio = torch.exp(new_logprob - old_logprob).clamp(0.0, 10.0)
            clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            loss = -torch.minimum(ratio * advantage, clipped * advantage)
            (loss / self.grad_accum_steps).backward()
            total_loss += float(loss.detach().cpu())
            total_ratio += float(ratio.detach().cpu())
            total_entropy += float(entropy.detach().cpu())
            if idx % self.grad_accum_steps == 0:
                self._clip_grad_norm()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._clear_runtime_caches()

        if len(steps) % self.grad_accum_steps != 0:
            self._clip_grad_norm()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._clear_runtime_caches()

        denom = max(1, len(steps))
        return {
            "policy_loss": total_loss / denom,
            "mean_ratio": total_ratio / denom,
            "entropy": total_entropy / denom,
        }

    def update_grpo(self, steps: list[StepRecord], lr: float) -> dict[str, float]:
        return self.update(steps, lr=lr)

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def _clip_grad_norm(self) -> None:
        if self.max_grad_norm <= 0:
            return
        trainable = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if trainable:
            self.torch.nn.utils.clip_grad_norm_(trainable, self.max_grad_norm)

    def _clear_runtime_caches(self) -> None:
        self._prior_score_cache.clear()

    def rank_actions(
        self,
        observation: str,
        action_space: list[str] | None = None,
        *,
        top_k: int = 5,
    ) -> list[dict[str, float | int | str]]:
        actions = action_space or self.action_space
        self.model.eval()
        with self.torch.no_grad():
            scores = self._candidate_scores_for_observation(observation, actions)
            log_probs, _ = self._candidate_log_distribution_from_scores(scores)
        ranked = sorted(
            range(len(actions)),
            key=lambda idx: float(scores[idx].detach().cpu()),
            reverse=True,
        )[: max(1, top_k)]
        return [
            {
                "rank": rank + 1,
                "action": actions[idx],
                "score": float(scores[idx].detach().cpu()),
                "logprob": float(log_probs[idx].detach().cpu()),
            }
            for rank, idx in enumerate(ranked)
        ]

    def _sequence_logprob(self, prompt_ids: list[int], response_ids: list[int]):
        return self._batch_sequence_logprobs(prompt_ids, [response_ids])[0]

    def _sequence_score(self, prompt_ids: list[int], response_ids: list[int]):
        return self._batch_sequence_scores(prompt_ids, [response_ids])[0]

    def _sequence_score_for_observation(
        self,
        observation: str,
        actions: list[str],
        response_ids: list[int],
    ):
        prompt_ids = self._encode_prompt(observation, actions)
        score = self._sequence_score(prompt_ids, response_ids)
        if self.action_score_calibration != "pmi":
            return score
        action = self._decode_cached_response(response_ids)
        if action is None:
            prior_ids = self._encode_prompt(self._reference_observation(), actions)
            with self.torch.no_grad():
                prior_score = self._sequence_score(prior_ids, response_ids).detach()
            return score - prior_score
        prior_scores = self._cached_prior_scores(actions)
        return score - prior_scores[actions.index(action)].detach()

    def _batch_sequence_logprobs(self, prompt_ids: list[int], response_ids_list: list[list[int]]):
        torch = self.torch
        sequences = [prompt_ids + response_ids for response_ids in response_ids_list]
        max_len = max(len(ids) for ids in sequences)
        ids = torch.full(
            (len(sequences), max_len),
            fill_value=self.pad_token_id,
            device=self.device,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(ids, device=self.device)
        response_mask = torch.zeros((len(sequences), max_len - 1), device=self.device, dtype=torch.float32)
        start = max(0, len(prompt_ids) - 1)
        for row, (sequence_ids, response_ids) in enumerate(zip(sequences, response_ids_list)):
            seq_len = len(sequence_ids)
            ids[row, :seq_len] = torch.tensor(sequence_ids, device=self.device, dtype=torch.long)
            attention_mask[row, :seq_len] = 1
            response_mask[row, start : start + len(response_ids)] = 1.0

        outputs = self.model(input_ids=ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        targets = ids[:, 1:]
        target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        log_norm = torch.logsumexp(logits, dim=-1)
        token_logprobs = target_logits - log_norm
        return (token_logprobs * response_mask).sum(dim=-1)

    def _batch_sequence_scores(self, prompt_ids: list[int], response_ids_list: list[list[int]]):
        scores = self._batch_sequence_logprobs(prompt_ids, response_ids_list)
        if self.action_score_normalization == "sum":
            return scores
        lengths = self.torch.tensor(
            [max(1, len(response_ids)) for response_ids in response_ids_list],
            device=self.device,
            dtype=scores.dtype,
        )
        if self.action_score_normalization == "sqrt":
            lengths = self.torch.sqrt(lengths)
        return scores / lengths

    def _candidate_scores(self, prompt_ids: list[int], actions: list[str]):
        response_ids = [self._encode_response(action) for action in actions]
        if len(response_ids) <= self.action_score_batch_size:
            return self._batch_sequence_scores(prompt_ids, response_ids)

        chunks = []
        for start in range(0, len(response_ids), self.action_score_batch_size):
            chunk = response_ids[start : start + self.action_score_batch_size]
            chunks.append(self._batch_sequence_scores(prompt_ids, chunk))
        return self.torch.cat(chunks, dim=0)

    def _candidate_scores_for_observation(self, observation: str, actions: list[str]):
        prompt_ids = self._encode_prompt(observation, actions)
        scores = self._candidate_scores(prompt_ids, actions)
        if self.action_score_calibration != "pmi":
            return scores
        return scores - self._cached_prior_scores(actions).detach()

    def _cached_prior_scores(self, actions: list[str]):
        key = tuple(actions)
        cached = self._prior_score_cache.get(key)
        if cached is not None:
            return cached
        was_training = self.model.training
        self.model.eval()
        with self.torch.no_grad():
            prior_ids = self._encode_prompt(self._reference_observation(), actions)
            prior_scores = self._candidate_scores(prior_ids, actions).detach()
        if was_training:
            self.model.train()
        self._prior_score_cache[key] = prior_scores
        return prior_scores

    def _candidate_log_distribution(self, observation: str, actions: list[str]):
        scores = self._candidate_scores_for_observation(observation, actions)
        return self._candidate_log_distribution_from_scores(scores)

    def _candidate_log_distribution_from_scores(self, scores):
        torch = self.torch
        scaled = scores / max(self.temperature, 1e-6)
        log_probs = torch.log_softmax(scaled, dim=0)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum()
        return log_probs, entropy

    def _candidate_entropy_no_grad(self, observation: str, actions: list[str]):
        with self.torch.no_grad():
            _, entropy = self._candidate_log_distribution(observation, actions)
        return entropy.detach()

    def _old_logprob_for_action(self, scores, log_probs, action_idx: int):
        if self.update_score_mode == "selected":
            return scores[action_idx]
        return log_probs[action_idx]

    def _candidate_action_logprob(self, observation: str, actions: list[str], action: str):
        if action not in actions:
            actions = list(actions) + [action]
        log_probs, entropy = self._candidate_log_distribution(observation, actions)
        action_idx = actions.index(action)
        return log_probs[action_idx], entropy

    def _encode_prompt(self, observation: str, action_space: list[str]) -> list[int]:
        prompt = self._format_prompt(observation, action_space)
        return self.tokenizer(prompt, add_special_tokens=True).input_ids

    def _format_prompt(self, observation: str, action_space: list[str]) -> str:
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            actions = "\n".join(f"- {action}" for action in action_space)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an agent solving a long-horizon interactive task. "
                        "Choose exactly one available action and output only that action."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Observation:\n{observation}\n\n"
                        f"Available actions:\n{actions}\n\n"
                        "Next action:"
                    ),
                },
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return format_agent_prompt(observation, action_space)

    def _reference_observation(self) -> str:
        return "No task-specific observation is available."

    def _encode_response(self, action: str) -> list[int]:
        cached = self._response_id_cache.get(action)
        if cached is not None:
            return list(cached)
        suffix = self.tokenizer.eos_token or ""
        ids = self.tokenizer(action + suffix, add_special_tokens=False).input_ids
        if not ids:
            ids = [self.eos_token_id or self.pad_token_id]
        self._response_id_cache[action] = list(ids)
        return ids

    def _decode_cached_response(self, response_ids: list[int]) -> str | None:
        normalized = list(response_ids)
        for action, cached in self._response_id_cache.items():
            if cached == normalized:
                return action
        return None

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

    def _sample_index(self, probs: list[float]) -> int:
        r = self.rng.random()
        acc = 0.0
        for idx, prob in enumerate(probs):
            acc += prob
            if r <= acc:
                return idx
        return max(0, len(probs) - 1)
