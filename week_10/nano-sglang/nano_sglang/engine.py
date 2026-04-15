"""Part 2: Inference Engine

Two phases:
  Prefill:  process entire prompt in one pass -> compute-bound
  Decode:   generate one token at a time from cache -> memory-bound
"""

import torch
import torch.nn.functional as F
from .model import Model, Tokenizer
from .sampling import SamplingParams, sample_token
from .sequence import Sequence, SequenceStatus


class Engine:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model = Model(model_path, device=device)
        self.tokenizer = Tokenizer(model_path)
        self.device = device

    @staticmethod
    def _get_layer_kv(cache, layer_idx: int):
        """Read one layer KV from different HF cache representations."""
        if isinstance(cache, tuple):
            return cache[layer_idx]

        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            return cache.key_cache[layer_idx], cache.value_cache[layer_idx]

        if hasattr(cache, "layers"):
            layer = cache.layers[layer_idx]
            for key_name, value_name in (
                ("keys", "values"),
                ("key_states", "value_states"),
                ("key", "value"),
                ("k", "v"),
            ):
                if hasattr(layer, key_name) and hasattr(layer, value_name):
                    return getattr(layer, key_name), getattr(layer, value_name)

        raise TypeError(f"Unsupported cache type: {type(cache)}")

    @staticmethod
    def _cache_seq_len(cache) -> int:
        k, _ = Engine._get_layer_kv(cache, 0)
        return k.shape[2]

    def prefill(self, seq: Sequence, sampling_params: SamplingParams) -> int:
        """Process all prompt tokens in one forward pass, return first generated token.
        Should also store past_key_values in seq and set status to DECODING."""
        seq.status = SequenceStatus.PREFILLING
        input_ids = torch.tensor([seq.prompt_token_ids], device=self.device)
        logits, past_key_values = self.model.forward(input_ids)
        next_token = sample_token(logits[:, -1, :], sampling_params).item()
        seq.past_key_values = past_key_values
        seq.status = SequenceStatus.DECODING
        return next_token

    def decode_step(self, seq: Sequence, sampling_params: SamplingParams) -> int:
        """Generate one token for a single sequence using cached KV."""
        last_token = seq.output_token_ids[-1]
        input_ids = torch.tensor([[last_token]], device=self.device)
        logits, past_key_values = self.model.forward(input_ids, past_key_values=seq.past_key_values)
        next_token = sample_token(logits[:, -1, :], sampling_params).item()
        seq.past_key_values = past_key_values
        return next_token

    def decode_batch(self, sequences: list[Sequence], sampling_params: SamplingParams) -> list[int]:
        """Generate one token for multiple sequences in a single GPU forward pass."""
        if not sequences:
            return []
        if len(sequences) == 1:
            return [self.decode_step(sequences[0], sampling_params)]

        n = len(sequences)
        input_ids = torch.tensor(
            [[seq.output_token_ids[-1]] for seq in sequences], device=self.device,
        )

        try:
            cache_lens = [self._cache_seq_len(seq.past_key_values) for seq in sequences]
            max_len = max(cache_lens)

            batched_legacy = []
            for layer_idx in range(self.model.num_layers):
                padded_keys, padded_values = [], []
                for i in range(n):
                    k, v = self._get_layer_kv(sequences[i].past_key_values, layer_idx)
                    pad = max_len - k.shape[2]
                    if pad > 0:
                        k = F.pad(k, (0, 0, pad, 0))
                        v = F.pad(v, (0, 0, pad, 0))
                    padded_keys.append(k)
                    padded_values.append(v)
                batched_legacy.append(
                    (torch.cat(padded_keys, dim=0), torch.cat(padded_values, dim=0))
                )

            attn_mask = torch.zeros(n, max_len + 1, device=self.device, dtype=torch.long)
            for i, cl in enumerate(cache_lens):
                attn_mask[i, max_len - cl:] = 1

            position_ids = torch.tensor([[cl] for cl in cache_lens], device=self.device)

            logits, new_cache = self.model.forward(
                input_ids, past_key_values=tuple(batched_legacy),
                position_ids=position_ids, attention_mask=attn_mask,
            )

            tokens = sample_token(logits[:, -1, :], sampling_params)

            for i, seq in enumerate(sequences):
                real_len = cache_lens[i] + 1
                pad = max_len - cache_lens[i]
                per_seq_legacy = []
                for layer_idx in range(self.model.num_layers):
                    layer_k, layer_v = self._get_layer_kv(new_cache, layer_idx)
                    k = layer_k[i:i+1, :, pad:pad + real_len, :].clone()
                    v = layer_v[i:i+1, :, pad:pad + real_len, :].clone()
                    per_seq_legacy.append((k, v))
                seq.past_key_values = tuple(per_seq_legacy)

            return [t.item() for t in tokens]
        except Exception:
            # Transformers cache APIs differ significantly across versions.
            # Fall back to per-sequence decode for compatibility.
            return [self.decode_step(seq, sampling_params) for seq in sequences]

    def generate(self, prompt: str, sampling_params: SamplingParams = None) -> str:
        """Generate text for a single prompt. Wire prefill + decode loop together."""
        if sampling_params is None:
            sampling_params = SamplingParams()
        if sampling_params.max_tokens <= 0:
            return ""
        prompt_ids = self.tokenizer.encode(prompt)
        seq = Sequence(seq_id=0, prompt_token_ids=prompt_ids, max_tokens=sampling_params.max_tokens)

        first_token = self.prefill(seq, sampling_params)
        seq.output_token_ids.append(first_token)
        if first_token == self.tokenizer.eos_token_id:
            seq.status = SequenceStatus.FINISHED
            return self.tokenizer.decode(seq.output_token_ids)

        while seq.num_generated < seq.max_tokens:
            next_token = self.decode_step(seq, sampling_params)
            seq.output_token_ids.append(next_token)
            if next_token == self.tokenizer.eos_token_id:
                break

        seq.status = SequenceStatus.FINISHED
        return self.tokenizer.decode(seq.output_token_ids)