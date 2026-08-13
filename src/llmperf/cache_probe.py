"""Deterministic request plans for paired external KV-cache probes."""

from dataclasses import dataclass
import hashlib
import hmac
import random
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CacheProbeRequest:
    request_id: str
    family_id: str
    role: str
    occurrence: int
    dispatch_index: int
    prompt: str
    prompt_hash: str
    local_input_tokens: int
    expected_shared_prefix_tokens: int
    delay_seconds: float
    seed: int
    persist_prompt_text: bool = False

    def metadata(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "request_id": self.request_id,
            "family_id": self.family_id,
            "role": self.role,
            "occurrence": self.occurrence,
            "dispatch_index": self.dispatch_index,
            "prompt_hash": self.prompt_hash,
            "local_input_tokens": self.local_input_tokens,
            "expected_shared_prefix_tokens": self.expected_shared_prefix_tokens,
            "delay_seconds": self.delay_seconds,
            "seed": self.seed,
        }
        if self.persist_prompt_text:
            result["prompt"] = self.prompt
        return result


def _hash_prompt(prompt: str, key: Optional[bytes]) -> str:
    payload = prompt.encode("utf-8")
    if key:
        return "hmac-sha256:" + hmac.new(key, payload, hashlib.sha256).hexdigest()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _replace_one_token(prompt: str, offset: int, tokenizer: Any) -> str:
    token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    if not token_ids:
        raise ValueError("Cannot mutate an empty token sequence")
    if offset < 0 or offset >= len(token_ids):
        raise ValueError(
            f"Mutation token offset {offset} is outside prompt length {len(token_ids)}"
        )
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size <= 1 or not hasattr(tokenizer, "decode"):
        raise ValueError("Selected tokenizer cannot construct token-exact mutations")
    original = token_ids[offset]
    for step in range(1, min(vocab_size, 4096)):
        candidate_ids = list(token_ids)
        candidate_ids[offset] = (original + step) % vocab_size
        try:
            candidate = tokenizer.decode(
                candidate_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            round_trip = tokenizer.encode(candidate, add_special_tokens=False)
        except (TypeError, ValueError):
            continue
        if list(round_trip) == candidate_ids:
            return candidate
    raise ValueError("Unable to construct a round-trip-safe token mutation")


def build_cache_probe_plan(
    base_prompts: Sequence[Tuple[str, int]],
    config: Dict[str, Any],
    tokenizer: Any,
    seed: int,
    hash_key: Optional[bytes] = None,
) -> List[CacheProbeRequest]:
    """Build a deterministic family-block plan with prime-before-warm ordering."""

    trials = int(config["trials"])
    repeats = int(config.get("repeats_after_prime", 1))
    if len(base_prompts) < trials:
        raise ValueError(f"Cache probe requires {trials} base prompts")
    family_indices = list(range(trials))
    if config.get("schedule", "randomized_family_blocks") == "randomized_family_blocks":
        random.Random(seed).shuffle(family_indices)

    mode = str(config["mode"])
    configured_prefix = int(config.get("shared_prefix_tokens") or 0)
    mutation_offset = config.get("mutation_token_offset")
    delay = float(config.get("delay_seconds", 0))
    persist = bool(config.get("persist_prompt_text", False))
    plan: List[CacheProbeRequest] = []
    dispatch_index = 0

    for family_number in family_indices:
        prime_prompt, prime_length = base_prompts[family_number]
        family_id = f"family-{family_number:05d}"
        warm_prompt = prime_prompt
        expected_prefix = prime_length
        if mode != "exact_repeat":
            if mode == "shared_prefix":
                offset = configured_prefix
            elif mode == "early_mutation":
                offset = int(mutation_offset if mutation_offset is not None else 8)
            elif mode == "late_mutation":
                offset = int(
                    mutation_offset
                    if mutation_offset is not None
                    else max(1, prime_length - 8)
                )
            else:
                raise ValueError(f"Unsupported cache probe mode: {mode}")
            warm_prompt = _replace_one_token(prime_prompt, offset, tokenizer)
            expected_prefix = offset

        for occurrence in range(repeats + 1):
            role = "prime" if occurrence == 0 else "warm"
            prompt = prime_prompt if role == "prime" else warm_prompt
            prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
            identity = (
                f"{seed}:{family_id}:{occurrence}:{_hash_prompt(prompt, hash_key)}"
            )
            request_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
            plan.append(
                CacheProbeRequest(
                    request_id=request_id,
                    family_id=family_id,
                    role=role,
                    occurrence=occurrence,
                    dispatch_index=dispatch_index,
                    prompt=prompt,
                    prompt_hash=_hash_prompt(prompt, hash_key),
                    local_input_tokens=prompt_length,
                    expected_shared_prefix_tokens=expected_prefix,
                    delay_seconds=delay if occurrence else 0,
                    seed=seed,
                    persist_prompt_text=persist,
                )
            )
            dispatch_index += 1
    return plan


class DependentPlanQueue:
    """Thread-safe plan queue that releases one request per family at a time."""

    def __init__(self, plan: Sequence[CacheProbeRequest]):
        self._families: Dict[str, List[CacheProbeRequest]] = {}
        self._ready: List[CacheProbeRequest] = []
        self._skipped: List[CacheProbeRequest] = []
        self._inflight = 0
        self._closed = False
        self._condition = threading.Condition()
        for request in plan:
            self._families.setdefault(request.family_id, []).append(request)
        for requests in self._families.values():
            requests.sort(key=lambda item: item.occurrence)
            self._ready.append(requests.pop(0))
        self._ready.sort(key=lambda item: item.dispatch_index)

    def claim(self, deadline: Optional[float] = None) -> Optional[CacheProbeRequest]:
        with self._condition:
            while not self._ready and self._inflight and not self._closed:
                if deadline is not None:
                    import time

                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._closed = True
                        return None
                    self._condition.wait(min(remaining, 0.25))
                else:
                    self._condition.wait()
            if self._closed or not self._ready:
                return None
            request = self._ready.pop(0)
            self._inflight += 1
            return request

    def complete(self, request: CacheProbeRequest, success: bool) -> None:
        with self._condition:
            self._inflight -= 1
            remaining = self._families[request.family_id]
            if success and remaining:
                next_request = remaining.pop(0)
                if next_request.delay_seconds:
                    # P0 delays are deliberately short and stay inside one Worker.
                    # Durable long-delay scheduling belongs to P1.
                    timer = threading.Timer(
                        next_request.delay_seconds, self._release, args=(next_request,)
                    )
                    timer.daemon = True
                    self._inflight += 1
                    timer.start()
                else:
                    self._ready.append(next_request)
                    self._ready.sort(key=lambda item: item.dispatch_index)
            elif not success and remaining:
                self._skipped.extend(remaining)
                remaining.clear()
            self._condition.notify_all()

    def _release(self, request: CacheProbeRequest) -> None:
        with self._condition:
            self._inflight -= 1
            self._ready.append(request)
            self._ready.sort(key=lambda item: item.dispatch_index)
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def skipped(self) -> List[CacheProbeRequest]:
        with self._condition:
            return list(self._skipped)
