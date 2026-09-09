from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .wan22.wan_video_dit import flash_attention, modulate, rope_apply
from gamewam.utils.logging_config import get_logger

logger = get_logger(__name__)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
        video_action_attention_mode: str = "condition_only",
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        valid_video_action_attention_modes = {
            "condition_only",
            "joint_bidirectional",
        }
        self.video_action_attention_mode = str(video_action_attention_mode)
        if self.video_action_attention_mode not in valid_video_action_attention_modes:
            raise ValueError(
                "`video_action_attention_mode` must be one of "
                f"{sorted(valid_video_action_attention_modes)}, got "
                f"{self.video_action_attention_mode!r}."
            )
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention. This will save memory but use more computation.")

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order:
            if name == self.expert_order[0]:
                continue
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )

        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    @staticmethod
    def _prepare_attention_mask(
        mask: Optional[torch.Tensor],
        *,
        dtype: torch.dtype,
        device: torch.device,
        batch_key_padding: bool = False,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        prepared = mask.to(device=device)
        if prepared.dtype == torch.bool:
            prepared = torch.where(prepared, 0.0, float("-inf")).to(dtype=dtype)
        else:
            prepared = prepared.to(dtype=dtype)
        if prepared.ndim == 2:
            prepared = (
                prepared.unsqueeze(1).unsqueeze(1)
                if batch_key_padding
                else prepared.unsqueeze(0).unsqueeze(0)
            )
        elif prepared.ndim == 3:
            prepared = prepared.unsqueeze(1)
        return prepared

    def _cached_ones_bool(
        self, batch_size: int, length: int, device: torch.device
    ) -> torch.Tensor:
        """All-True bool mask parts depend only on (batch, length, device);
        cache them across steps. Cached tensors are only ever read
        (``torch.cat``/``unsqueeze``), never mutated in place."""
        cache = getattr(self, "_ones_bool_cache", None)
        if cache is None:
            cache = {}
            self._ones_bool_cache = cache
        key = (int(batch_size), int(length), device)
        out = cache.get(key)
        if out is None:
            out = torch.ones(key[0], key[1], dtype=torch.bool, device=device)
            cache[key] = out
        return out

    def _get_clean_causal_mask(
        self,
        far_token_count: int,
        num_clean_frames: int,
        tokens_per_frame: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        history_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cached blockwise causal mask for [far + clean] tokens.

        Equivalent to
        ``_prepare_attention_mask(_build_clean_causal_mask(...))`` but
        the structural part (which depends only on geometry, dtype,
        device) is built once and cached as an additive float
        ``[1,1,T,T]`` tensor. Only the per-sample history masking is
        recomputed when ``history_valid_mask`` is provided.

        SDPA never modifies ``attn_mask`` in place, so the cached
        ``[1,1,T,T]`` tensor is shared directly across layers and steps
        when there is no per-sample history masking.
        """
        H = int(far_token_count)
        ncf = int(num_clean_frames)
        tpf = int(tokens_per_frame)
        total = H + ncf * tpf
        cache_key = (H, ncf, tpf, dtype, device)

        cache = getattr(self, "_clean_mask_struct_cache", None)
        if cache is None:
            cache = {}
            self._clean_mask_struct_cache = cache
        struct_additive = cache.get(cache_key)
        if struct_additive is None:
            struct_bool = self._build_clean_causal_mask(
                H, ncf, tpf, device=device, history_valid_mask=None,
            )
            struct_additive = torch.where(
                struct_bool, 0.0, float("-inf")
            ).to(dtype=dtype, device=device)
            struct_additive = struct_additive.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
            cache[cache_key] = struct_additive

        if history_valid_mask is None or H == 0:
            return struct_additive

        B = int(history_valid_mask.shape[0])
        mask = struct_additive.expand(B, 1, total, total).clone()
        invalid_hist = ~history_valid_mask[:, None, :]  # [B, 1, H]
        # Nobody attends to invalid history columns.
        mask[:, :, :, :H].masked_fill_(invalid_hist.unsqueeze(2), float("-inf"))
        # Invalid history rows attend to nothing.
        mask[:, :, :H, :].masked_fill_(invalid_hist.unsqueeze(-1), float("-inf"))
        # Invalid history diagonal attends to self (avoid empty softmax).
        # Valid diagonals are already 0 (far prefix is all-self-attention),
        # so this only restores diagonals the row fill above set to -inf.
        diag = torch.arange(H, device=device)
        mask[:, :, diag, diag] = 0.0
        return mask

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = self._prepare_attention_mask(
            attention_mask, dtype=q_cat.dtype, device=q_cat.device
        )

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    def _apply_expert_post_block(
        self,
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                cache_key = "_prepared_attention_mask"
                context_mask = context_payload.get(cache_key)
                if context_mask is None and context_payload.get("mask") is not None:
                    context_mask = self._prepare_attention_mask(
                        context_payload["mask"], dtype=x.dtype, device=x.device,
                        batch_key_padding=True,
                    )
                    context_payload[cache_key] = context_mask
                x = x + block.cross_attn(
                    block.norm3(x), context, ctx_mask=context_mask
                )

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Returns: ``(q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp,
        use_gradient_checkpointing)``. See the implementation for shapes.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        ``context_payload`` is an optional dict ``{"context", "mask"}`` for
        cross-attention. Returns the updated tokens after self-attn residual,
        optional cross-attn, and MLP.
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    @staticmethod
    def _payloads_share_context(post_specs: list) -> bool:
        """Return True when every cross-attention payload references the SAME
        context tensor object (identity), so one K/V computation serves all
        blocks in the layer."""
        context = None
        seen = False
        for spec in post_specs:
            payload = spec[6]
            if payload is None or payload.get("context") is None:
                continue
            if not seen:
                context = payload["context"]
                seen = True
            elif payload["context"] is not context:
                return False
        return seen

    def _apply_noisy_posts_shared_context(
        self,
        block,
        post_specs: list,
        use_gradient_checkpointing: bool,
    ) -> tuple:
        """Apply post-attention computations for all noisy video blocks of one"""
        ca = block.cross_attn
        shared_context = None
        for spec in post_specs:
            payload = spec[6]
            if payload is not None and payload.get("context") is not None:
                shared_context = payload["context"]
                break

        # Prepare each block's additive cross-attention mask outside the
        # checkpoint (mirrors the lazy ``_prepared_attention_mask`` cache in
        # ``_apply_expert_post_block``, keyed per payload).
        prepared_masks = []
        for spec in post_specs:
            payload = spec[6]
            residual_x = spec[1]
            mask = None
            if payload is not None and payload.get("context") is not None:
                mask = payload.get("_prepared_attention_mask")
                if mask is None and payload.get("mask") is not None:
                    mask = self._prepare_attention_mask(
                        payload["mask"],
                        dtype=residual_x.dtype,
                        device=residual_x.device,
                        batch_key_padding=True,
                    )
                    payload["_prepared_attention_mask"] = mask
            prepared_masks.append(mask)

        def _combined_post_fn(*flat_tensors):
            cursor = 0
            if shared_context is not None:
                context = flat_tensors[cursor]
                cursor += 1
                ctx_k = ca.norm_k(ca.k(context))
                ctx_v = ca.v(context)
            else:
                ctx_k = None
                ctx_v = None
            outs = []
            for spec_index in range(len(post_specs)):
                mixed_attn_out = flat_tensors[cursor]
                residual_x = flat_tensors[cursor + 1]
                gate_msa = flat_tensors[cursor + 2]
                shift_mlp = flat_tensors[cursor + 3]
                scale_mlp = flat_tensors[cursor + 4]
                gate_mlp = flat_tensors[cursor + 5]
                cursor += 6
                payload = post_specs[spec_index][6]
                x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))
                if payload is not None and payload.get("context") is not None:
                    mask = prepared_masks[spec_index]
                    if ctx_k is not None:
                        q = ca.norm_q(ca.q(block.norm3(x)))
                        attn = flash_attention(
                            q=q, k=ctx_k, v=ctx_v,
                            num_heads=ca.num_heads, ctx_mask=mask,
                        )
                        x = x + ca.o(attn)
                    else:
                        x = x + ca(
                            block.norm3(x), payload["context"], ctx_mask=mask
                        )
                mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
                x = block.gate(x, gate_mlp, block.ffn(mlp_input))
                outs.append(x)
            return tuple(outs)

        flat_tensors = []
        if shared_context is not None:
            flat_tensors.append(shared_context)
        for spec in post_specs:
            flat_tensors.extend(spec[:6])

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _combined_post_fn,
                *flat_tensors,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return _combined_post_fn(*flat_tensors)

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Prefill video branch once and cache per-layer K/V for action denoising.

        ``video_context_payload`` is an optional ``{"context", "mask"}`` dict
        for cross-attention. Returns a layer-wise cache list ``[{"k","v"}, ...]``.
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            # Video prefill uses only video self-attention mask.
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            # Update video tokens for the next layer and persist current layer K/V.
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V.

        Returns the updated action tokens ``[B, Sa, D]``.
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        # Use the action query rows from the joint mask.
        action_attention_mask = attention_mask[int(video_seq_len):total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )

            k_video = video_kv_cache[layer_idx]["k"]
            v_video = video_kv_cache[layer_idx]["v"]

            # Build K/V cat: [video, action]
            k_parts = [k_video]
            v_parts = [v_video]
            k_parts.append(k_action)
            v_parts.append(v_action)

            k_cat = torch.cat(k_parts, dim=1)
            v_cat = torch.cat(v_parts, dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ):
        for k in self.expert_order:
            if k not in embeds_all:
                raise ValueError(f"Missing expert tokens for '{k}'")
            if k not in freqs_all:
                raise ValueError(f"Missing expert freqs for '{k}'")
            if k not in t_mod_all:
                raise ValueError(f"Missing expert t_mod for '{k}'")

        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        tokens_all = {k: v for k, v in embeds_all.items()}

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]

                # Video/Action expert: AdaLN + RoPE
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]
                (
                    q, k, v,
                    residual_x,
                    gate_msa, shift_mlp, scale_mlp, gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert, block=block, x=x, freqs=freqs, t_mod=t_mod,
                )
                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            # Concat all tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if attention_mask.shape[0] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                )

            mixed = self._mixed_attention(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask)

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]

                # Video/Action expert: gated residual + cross-attn + AdaLN FFN
                context_payload = context_all.get(name)
                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=cached_expert["block"],
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        return tokens_all

    def forward_ar(
        self,
        clean_video_tokens: torch.Tensor,
        noisy_video_tokens_list: list[torch.Tensor],
        action_tokens: torch.Tensor,
        clean_video_freqs: torch.Tensor,
        noisy_video_freqs_list: list[torch.Tensor],
        action_freqs: torch.Tensor,
        clean_video_t_mod: torch.Tensor,
        noisy_video_t_mod_list: list[torch.Tensor],
        action_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        action_context_payload: Optional[dict],
        num_obs_latents: int,
        tokens_per_frame: int,
        far_token_count: int = 0,
        history_valid_mask: Optional[torch.Tensor] = None,
        noisy_video_context_payloads_list: Optional[list[Optional[dict]]] = None,
    ) -> dict[str, torch.Tensor]:
        """AR Teacher-Forcing forward with layer-by-layer block-wise attention.

        Memory-efficient: clean/noisy/action processed within each layer,
        only current-layer K/V alive (no cross-layer cache). Per layer:
        clean prefill (causal) → each noisy block (visible clean + self)
        → action (visible clean + self) → discard this layer's K/V.
        """
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]

        num_noisy_blocks = len(noisy_video_tokens_list)
        if noisy_video_context_payloads_list is not None and len(noisy_video_context_payloads_list) != num_noisy_blocks:
            raise ValueError(
                "`noisy_video_context_payloads_list` length must match noisy_video_tokens_list, "
                f"got {len(noisy_video_context_payloads_list)} vs {num_noisy_blocks}."
            )
        S_clean = clean_video_tokens.shape[1]

        # Current token states (updated layer by layer)
        x_clean = clean_video_tokens
        x_noisy_list = list(noisy_video_tokens_list)  # mutable copy
        x_action = action_tokens

        num_clean_frames = (S_clean - far_token_count) // tokens_per_frame
        clean_mask = self._get_clean_causal_mask(
            far_token_count, num_clean_frames, tokens_per_frame,
            dtype=clean_video_tokens.dtype, device=clean_video_tokens.device,
            history_valid_mask=history_valid_mask,
        )
        action_seq_len = int(x_action.shape[1])
        if self.video_action_attention_mode == "joint_bidirectional" and (
            num_noisy_blocks <= 0 or action_seq_len <= 0
        ):
            raise ValueError(
                "joint_bidirectional AR forward requires at least one noisy "
                "video block and one noisy action token."
            )
        if num_noisy_blocks > 0 and action_seq_len > 0:
            if action_seq_len % num_noisy_blocks != 0:
                raise ValueError(
                    f"action_seq_len={action_seq_len} must be divisible by "
                    f"num_noisy_blocks={num_noisy_blocks}"
                )
            action_per_block = action_seq_len // num_noisy_blocks
        else:
            action_per_block = action_seq_len

        def _history_context_mask(
            visible_end: int, self_token_count: int
        ) -> Optional[torch.Tensor]:
            if history_valid_mask is None or far_token_count <= 0:
                return None
            batch_size = int(clean_video_tokens.shape[0])
            parts = [
                history_valid_mask,
                self._cached_ones_bool(
                    batch_size,
                    int(visible_end) - int(far_token_count),
                    clean_video_tokens.device,
                ),
            ]
            parts.append(self._cached_ones_bool(
                batch_size, int(self_token_count),
                clean_video_tokens.device,
            ))
            return self._prepare_attention_mask(
                torch.cat(parts, dim=1).unsqueeze(1),
                dtype=clean_video_tokens.dtype,
                device=clean_video_tokens.device,
            )

        noisy_context_masks = []
        for block_idx, noisy_tokens in enumerate(x_noisy_list):
            visible_end = min(
                far_token_count + (num_obs_latents + block_idx) * tokens_per_frame,
                S_clean,
            )
            noisy_context_masks.append(_history_context_mask(
                visible_end, int(noisy_tokens.shape[1])
            ))
        action_context_masks = []
        if action_seq_len > 0:
            action_mask_blocks = max(num_noisy_blocks, 1)
            for block_idx in range(action_mask_blocks):
                visible_end = min(
                    far_token_count
                    + (num_obs_latents + (block_idx if num_noisy_blocks > 1 else 0))
                    * tokens_per_frame,
                    S_clean,
                )
                self_tokens = (
                    action_per_block if num_noisy_blocks > 1 else action_seq_len
                )
                action_context_masks.append(_history_context_mask(
                    visible_end, self_tokens
                ))

        for layer_idx in range(self.num_layers):
            # Step 1: Clean video prefill (blockwise causal)
            block_v = video_expert.blocks[layer_idx]
            (q_c, k_c, v_c, res_c, gate_c, shift_c, scale_c, gate_mlp_c, ckpt_c
             ) = self._build_expert_attention_io(
                expert=video_expert, block=block_v, x=x_clean,
                freqs=clean_video_freqs, t_mod=clean_video_t_mod,
            )
            mixed_c = flash_attention(q=q_c, k=k_c, v=v_c, num_heads=self.num_heads, ctx_mask=clean_mask)
            x_clean = self._apply_post_with_optional_checkpoint(
                block=block_v, residual_x=res_c, gate_msa=gate_c,
                shift_mlp=shift_c, scale_mlp=scale_c, gate_mlp=gate_mlp_c,
                use_gradient_checkpointing=ckpt_c, mixed_slice=mixed_c,
                context_payload=video_context_payload,
            )
            # Keep this layer's clean K/V for noisy and action phases
            layer_k_clean = k_c
            layer_v_clean = v_c

            if self.video_action_attention_mode == "joint_bidirectional":
                # The video block and its aligned action chunk use Q/K/V
                # from the same pre-layer states and share one attention call.
                # This gives true bidirectional coupling without allowing
                # either prediction region to contaminate the clean prefix.
                block_a = action_expert.blocks[layer_idx]
                (q_a, k_a, v_a, res_a, gate_a, shift_a, scale_a, gate_mlp_a, ckpt_a
                 ) = self._build_expert_attention_io(
                    expert=action_expert, block=block_a, x=x_action,
                    freqs=action_freqs, t_mod=action_t_mod,
                )
                mixed_action_parts = []
                for block_idx in range(num_noisy_blocks):
                    x_n = x_noisy_list[block_idx]
                    (q_n, k_n, v_n, res_n, gate_n, shift_n, scale_n, gate_mlp_n, ckpt_n
                     ) = self._build_expert_attention_io(
                        expert=video_expert, block=block_v, x=x_n,
                        freqs=noisy_video_freqs_list[block_idx],
                        t_mod=noisy_video_t_mod_list[block_idx],
                    )
                    a_start = block_idx * action_per_block
                    a_end = (
                        a_start + action_per_block
                        if block_idx < num_noisy_blocks - 1
                        else action_seq_len
                    )
                    q_a_i = q_a[:, a_start:a_end]
                    k_a_i = k_a[:, a_start:a_end]
                    v_a_i = v_a[:, a_start:a_end]
                    visible_end = min(
                        far_token_count
                        + (num_obs_latents + block_idx) * tokens_per_frame,
                        S_clean,
                    )
                    k_parts = [layer_k_clean[:, :visible_end]]
                    v_parts = [layer_v_clean[:, :visible_end]]
                    k_parts.extend([k_n, k_a_i])
                    v_parts.extend([v_n, v_a_i])
                    q_joint = torch.cat([q_n, q_a_i], dim=1)
                    joint_mask = _history_context_mask(
                        visible_end,
                        int(k_n.shape[1] + k_a_i.shape[1]),
                    )
                    mixed_joint = flash_attention(
                        q=q_joint,
                        k=torch.cat(k_parts, dim=1),
                        v=torch.cat(v_parts, dim=1),
                        num_heads=self.num_heads,
                        ctx_mask=joint_mask,
                    )
                    video_width = int(q_n.shape[1])
                    mixed_n = mixed_joint[:, :video_width]
                    mixed_action_parts.append(mixed_joint[:, video_width:])
                    x_noisy_list[block_idx] = self._apply_post_with_optional_checkpoint(
                        block=block_v, residual_x=res_n, gate_msa=gate_n,
                        shift_mlp=shift_n, scale_mlp=scale_n, gate_mlp=gate_mlp_n,
                        use_gradient_checkpointing=ckpt_n, mixed_slice=mixed_n,
                        context_payload=(
                            noisy_video_context_payloads_list[block_idx]
                            if noisy_video_context_payloads_list is not None
                            else None
                        ),
                    )
                mixed_a = torch.cat(mixed_action_parts, dim=1)
                x_action = self._apply_post_with_optional_checkpoint(
                    block=block_a, residual_x=res_a, gate_msa=gate_a,
                    shift_mlp=shift_a, scale_mlp=scale_a, gate_mlp=gate_mlp_a,
                    use_gradient_checkpointing=ckpt_a, mixed_slice=mixed_a,
                    context_payload=action_context_payload,
                )
                continue

            # Step 3: Noisy video blocks (each attends to visible clean + self)
            for block_idx in range(num_noisy_blocks):
                x_n = x_noisy_list[block_idx]
                (q_n, k_n, v_n, res_n, gate_n, shift_n, scale_n, gate_mlp_n, ckpt_n
                 ) = self._build_expert_attention_io(
                    expert=video_expert, block=block_v, x=x_n,
                    freqs=noisy_video_freqs_list[block_idx],
                    t_mod=noisy_video_t_mod_list[block_idx],
                )
                # Visible clean: far + clean[0..obs+block_idx-1]
                visible_end = far_token_count + (num_obs_latents + block_idx) * tokens_per_frame
                visible_end = min(visible_end, S_clean)
                k_ctx = layer_k_clean[:, :visible_end]
                v_ctx = layer_v_clean[:, :visible_end]
                k_parts_n = [k_ctx]
                v_parts_n = [v_ctx]
                k_parts_n.append(k_n)
                v_parts_n.append(v_n)
                k_cat_n = torch.cat(k_parts_n, dim=1)
                v_cat_n = torch.cat(v_parts_n, dim=1)
                ctx_mask_n = noisy_context_masks[block_idx]
                mixed_n = flash_attention(q=q_n, k=k_cat_n, v=v_cat_n, num_heads=self.num_heads, ctx_mask=ctx_mask_n)
                x_noisy_list[block_idx] = self._apply_post_with_optional_checkpoint(
                    block=block_v, residual_x=res_n, gate_msa=gate_n,
                    shift_mlp=shift_n, scale_mlp=scale_n, gate_mlp=gate_mlp_n,
                    use_gradient_checkpointing=ckpt_n, mixed_slice=mixed_n,
                    context_payload=(
                        noisy_video_context_payloads_list[block_idx]
                        if noisy_video_context_payloads_list is not None
                        else None
                    ),
                )

            # Step 4: Action blocks (split by noisy block)
            #
            # action_block[i] sees: visible_clean[0..obs+i-1] + self.
            # It does not attend to noisy video K/V, matching inference.
            # This ensures no information leakage from future clean frames.
            if x_action.shape[1] > 0:
                block_a = action_expert.blocks[layer_idx]
                # Compute action Q/K/V for the full sequence first
                (q_a, k_a, v_a, res_a, gate_a, shift_a, scale_a, gate_mlp_a, ckpt_a
                 ) = self._build_expert_attention_io(
                    expert=action_expert, block=block_a, x=x_action,
                    freqs=action_freqs, t_mod=action_t_mod,
                )

                if num_noisy_blocks <= 1:
                    # Single block: action sees only obs clean + self
                    # (matches non-AR first_k_frames_causal: action only sees obs tokens)
                    visible_end = far_token_count + num_obs_latents * tokens_per_frame
                    visible_end = min(visible_end, S_clean)
                    k_ctx = layer_k_clean[:, :visible_end]
                    v_ctx = layer_v_clean[:, :visible_end]

                    k_parts = [k_ctx]
                    v_parts = [v_ctx]
                    k_parts.append(k_a)
                    v_parts.append(v_a)
                    k_cat_a = torch.cat(k_parts, dim=1)
                    v_cat_a = torch.cat(v_parts, dim=1)
                    ctx_mask_a = action_context_masks[0]
                    mixed_a = flash_attention(q=q_a, k=k_cat_a, v=v_cat_a, num_heads=self.num_heads, ctx_mask=ctx_mask_a)
                else:
                    # Multiple blocks: action block[i] sees visible_clean[0..obs+i-1] + self
                    # Action does NOT attend to noisy video (train-inference consistency:
                    # at inference only action is denoised, video is condition-only).
                    mixed_a_parts = []
                    for blk_i in range(num_noisy_blocks):
                        a_start = blk_i * action_per_block
                        a_end = a_start + action_per_block if blk_i < num_noisy_blocks - 1 else action_seq_len
                        q_a_i = q_a[:, a_start:a_end]
                        k_a_i = k_a[:, a_start:a_end]
                        v_a_i = v_a[:, a_start:a_end]

                        # Same visible clean as noisy block[i]
                        visible_end = far_token_count + (num_obs_latents + blk_i) * tokens_per_frame
                        visible_end = min(visible_end, S_clean)
                        k_ctx = layer_k_clean[:, :visible_end]
                        v_ctx = layer_v_clean[:, :visible_end]

                        k_parts_i = [k_ctx]
                        v_parts_i = [v_ctx]
                        k_parts_i.append(k_a_i)
                        v_parts_i.append(v_a_i)

                        k_cat_i = torch.cat(k_parts_i, dim=1)
                        v_cat_i = torch.cat(v_parts_i, dim=1)

                        ctx_mask_i = action_context_masks[blk_i]

                        mixed_a_i = flash_attention(q=q_a_i, k=k_cat_i, v=v_cat_i, num_heads=self.num_heads, ctx_mask=ctx_mask_i)
                        mixed_a_parts.append(mixed_a_i)

                    mixed_a = torch.cat(mixed_a_parts, dim=1)

                x_action = self._apply_post_with_optional_checkpoint(
                    block=block_a, residual_x=res_a, gate_msa=gate_a,
                    shift_mlp=shift_a, scale_mlp=scale_a, gate_mlp=gate_mlp_a,
                    use_gradient_checkpointing=ckpt_a, mixed_slice=mixed_a,
                    context_payload=action_context_payload,
                )
            # Layer K/V are now out of scope → freed by GC

        return {"noisy_video": x_noisy_list, "action": x_action}

    # AR Inference with KV cache persistence (DreamZero-style)

    def prefill_clean_kv_cache(
        self,
        clean_video_tokens: torch.Tensor,
        clean_video_freqs: torch.Tensor,
        clean_video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        far_token_count: int = 0,
        tokens_per_frame: int = 49,
        history_valid_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Prefill clean tokens through all layers, cache per-layer K/V.

        ``video_context_payload`` is cache-stationary context only. Dynamic
        state/proprio belongs in runtime action/noisy-video payloads, not in
        this persistent clean cache path.
        """
        video_expert = self.mixtures["video"]

        x_clean = clean_video_tokens

        S_clean = clean_video_tokens.shape[1]
        num_clean_frames = (S_clean - far_token_count) // tokens_per_frame

        # Build blockwise causal mask once (used every layer for clean prefill)
        clean_mask = self._get_clean_causal_mask(
            far_token_count, num_clean_frames, tokens_per_frame,
            dtype=clean_video_tokens.dtype, device=clean_video_tokens.device,
            history_valid_mask=history_valid_mask,
        )

        clean_kv_cache = []

        for layer_idx in range(self.num_layers):
            block_v = video_expert.blocks[layer_idx]
            (q_c, k_c, v_c, res_c, gate_c, shift_c, scale_c, gate_mlp_c, ckpt_c
             ) = self._build_expert_attention_io(
                expert=video_expert, block=block_v, x=x_clean,
                freqs=clean_video_freqs, t_mod=clean_video_t_mod,
            )
            mixed_c = flash_attention(q=q_c, k=k_c, v=v_c, num_heads=self.num_heads, ctx_mask=clean_mask)
            x_clean = self._apply_post_with_optional_checkpoint(
                block=block_v, residual_x=res_c, gate_msa=gate_c,
                shift_mlp=shift_c, scale_mlp=scale_c, gate_mlp=gate_mlp_c,
                use_gradient_checkpointing=ckpt_c, mixed_slice=mixed_c,
                context_payload=video_context_payload,
            )
            clean_kv_cache.append((k_c, v_c))

        return {
            "clean_kv": clean_kv_cache,
            "x_clean": x_clean,
            "far_token_count": far_token_count,
            "tokens_per_frame": tokens_per_frame,
            "history_valid_mask": history_valid_mask,
            # Full per-timestep validity for the x_clean sequence
            # (``[B, S_clean]``). Equal to ``history_valid_mask``
            # over the first ``far_token_count`` slots and ``True``
            # for the obs tokens (which are always valid — they
            # were produced by the noisy forward that produced
            # ``x_clean``). Consumers that need a ``[B, S]`` mask
            # aligned with ``x_clean`` should use this key, not
            # ``history_valid_mask`` which only covers the
            # prefill history.
            "clean_valid_mask": (
                torch.cat(
                    [
                        history_valid_mask
                        if history_valid_mask is not None
                        else torch.ones(
                            x_clean.shape[0],
                            far_token_count,
                            dtype=torch.bool,
                            device=x_clean.device,
                        ),
                        torch.ones(
                            x_clean.shape[0],
                            x_clean.shape[1] - far_token_count,
                            dtype=torch.bool,
                            device=x_clean.device,
                        ),
                    ],
                    dim=1,
                )
                if history_valid_mask is not None
                else None
            ),
            # The clean context is part of the persistent-cache contract.
            # Keep a lightweight identity/version signature so append and
            # denoise cannot silently switch (or mutate) it mid-cycle.
            "clean_context_payload_signature": self._context_payload_signature(
                video_context_payload
            ),
        }

    @staticmethod
    def _context_payload_signature(value):
        """Return a recursive identity/version signature for cache context.

        Tensor ``_version`` changes on normal in-place writes, while identity
        catches tensor/dict replacement. No tensor values are copied, so this
        check is cheap enough to run for every append/denoise call.
        """
        if isinstance(value, torch.Tensor):
            return (
                "tensor",
                id(value),
                int(getattr(value, "_version", -1)),
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
            )
        if isinstance(value, dict):
            return (
                "dict",
                id(value),
                tuple(
                    (key, MoT._context_payload_signature(item))
                    for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                ),
            )
        if isinstance(value, (list, tuple)):
            return (
                type(value).__name__,
                id(value),
                tuple(MoT._context_payload_signature(item) for item in value),
            )
        return (type(value).__name__, value)

    def _assert_clean_context_payload_stable(
        self, cache: dict, video_context_payload: Optional[dict]
    ) -> None:
        expected = cache.get("clean_context_payload_signature")
        if expected is None:
            raise RuntimeError(
                "Clean KV cache has no context signature; it was not created "
                "by the current prefill_clean_kv_cache contract."
            )
        actual = self._context_payload_signature(video_context_payload)
        if actual != expected:
            raise RuntimeError(
                "Clean cache context payload changed within an AR cycle. "
                "End the cycle and prefill a new cache instead of mixing contexts."
            )

    def forward_ar_from_clean_cache(
        self,
        cache: dict,
        noisy_video_tokens_list: list[torch.Tensor],
        action_tokens: torch.Tensor,
        noisy_video_freqs_list: list[torch.Tensor],
        action_freqs: torch.Tensor,
        noisy_video_t_mod_list: list[torch.Tensor],
        action_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        action_context_payload: Optional[dict],
        num_obs_latents: int,
        noisy_video_context_payloads_list: Optional[list[Optional[dict]]] = None,
        num_action_blocks: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        self._assert_clean_context_payload_stable(cache, video_context_payload)
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        num_noisy_blocks = len(noisy_video_tokens_list)
        if num_action_blocks is None:
            num_action_blocks = num_noisy_blocks
        if noisy_video_context_payloads_list is not None and len(noisy_video_context_payloads_list) != num_noisy_blocks:
            raise ValueError("noisy_video_context_payloads_list length must match noisy_video_tokens_list")
        history_valid_mask = cache["history_valid_mask"]
        far_token_count = int(cache["far_token_count"])
        tokens_per_frame = int(cache["tokens_per_frame"])
        S_clean = cache["clean_kv"][0][0].shape[1]
        if int(num_action_blocks) <= 0 or num_noisy_blocks % int(num_action_blocks) != 0:
            raise ValueError(
                "num_noisy_blocks must be divisible by num_action_blocks, "
                f"got {num_noisy_blocks} and {num_action_blocks}"
            )
        plan_subblocks = max(1, num_noisy_blocks // max(1, int(num_action_blocks)))
        x_noisy_list = list(noisy_video_tokens_list)
        x_action = action_tokens
        action_seq_len = int(x_action.shape[1])
        if action_seq_len % int(num_action_blocks) != 0:
            raise ValueError(
                "action token count must be divisible by num_action_blocks, "
                f"got {action_seq_len} and {num_action_blocks}."
            )
        action_per_block = action_seq_len // int(num_action_blocks)
        if self.video_action_attention_mode == "joint_bidirectional" and (
            num_noisy_blocks <= 0 or action_per_block <= 0
        ):
            raise ValueError(
                "joint_bidirectional cached AR forward requires aligned noisy "
                "video blocks and action chunks."
            )
        action_attention_masks = self._build_action_attention_masks_from_clean_cache(
            cache=cache,
            x_action=x_action,
            num_obs_latents=num_obs_latents,
            num_action_blocks=int(num_action_blocks),
        )

        noisy_attention_masks = []
        for anchor_idx in range(int(num_action_blocks)):
            block_start = anchor_idx * plan_subblocks
            block_end = block_start + plan_subblocks
            visible_end = min(
                far_token_count + (num_obs_latents + anchor_idx) * tokens_per_frame,
                S_clean,
            )
            if history_valid_mask is not None and far_token_count > 0:
                batch_size = int(x_noisy_list[block_start].shape[0])
                group_tokens = sum(
                    int(x_noisy_list[index].shape[1])
                    for index in range(block_start, block_end)
                )
                valid_parts = [
                    history_valid_mask,
                    self._cached_ones_bool(
                        batch_size, visible_end - far_token_count,
                        x_action.device,
                    ),
                    self._cached_ones_bool(
                        batch_size, group_tokens,
                        x_action.device,
                    ),
                ]
                if self.video_action_attention_mode == "joint_bidirectional":
                    valid_parts.append(self._cached_ones_bool(
                        batch_size, action_per_block, x_action.device
                    ))
                noisy_attention_masks.append(self._prepare_attention_mask(
                    torch.cat(valid_parts, dim=1).unsqueeze(1),
                    dtype=x_action.dtype, device=x_action.device,
                ))
            else:
                noisy_attention_masks.append(None)

        for layer_idx in range(self.num_layers):
            block_v = video_expert.blocks[layer_idx]
            layer_k_clean, layer_v_clean = cache["clean_kv"][layer_idx]

            if self.video_action_attention_mode == "joint_bidirectional":
                block_a = action_expert.blocks[layer_idx]
                (q_a, k_a, v_a, res_a, gate_a, shift_a, scale_a, gate_mlp_a, ckpt_a
                 ) = self._build_expert_attention_io(
                    expert=action_expert, block=block_a, x=x_action,
                    freqs=action_freqs, t_mod=action_t_mod,
                )
                mixed_action_parts = []
                for anchor_idx in range(int(num_action_blocks)):
                    block_start = anchor_idx * plan_subblocks
                    block_end = block_start + plan_subblocks
                    q_video_parts = []
                    k_video_parts = []
                    v_video_parts = []
                    video_post_inputs = []
                    for block_idx in range(block_start, block_end):
                        x_n = x_noisy_list[block_idx]
                        (q_n, k_n, v_n, res_n, gate_n, shift_n, scale_n, gate_mlp_n, ckpt_n
                         ) = self._build_expert_attention_io(
                            expert=video_expert, block=block_v, x=x_n,
                            freqs=noisy_video_freqs_list[block_idx],
                            t_mod=noisy_video_t_mod_list[block_idx],
                        )
                        q_video_parts.append(q_n)
                        k_video_parts.append(k_n)
                        v_video_parts.append(v_n)
                        video_post_inputs.append((
                            block_idx, res_n, gate_n, shift_n, scale_n,
                            gate_mlp_n, ckpt_n, int(q_n.shape[1]),
                        ))
                    q_video = torch.cat(q_video_parts, dim=1)
                    k_video = torch.cat(k_video_parts, dim=1)
                    v_video = torch.cat(v_video_parts, dim=1)
                    a_start = anchor_idx * action_per_block
                    a_end = a_start + action_per_block
                    q_action = q_a[:, a_start:a_end]
                    k_action = k_a[:, a_start:a_end]
                    v_action = v_a[:, a_start:a_end]
                    visible_end = min(
                        far_token_count
                        + (num_obs_latents + anchor_idx) * tokens_per_frame,
                        S_clean,
                    )
                    q_joint = torch.cat([q_video, q_action], dim=1)
                    mixed_joint = flash_attention(
                        q=q_joint,
                        k=torch.cat([
                            layer_k_clean[:, :visible_end], k_video, k_action
                        ], dim=1),
                        v=torch.cat([
                            layer_v_clean[:, :visible_end], v_video, v_action
                        ], dim=1),
                        num_heads=self.num_heads,
                        ctx_mask=noisy_attention_masks[anchor_idx],
                    )
                    video_width = int(q_video.shape[1])
                    mixed_video = mixed_joint[:, :video_width]
                    mixed_action_parts.append(mixed_joint[:, video_width:])
                    cursor = 0
                    for (
                        block_idx, res_n, gate_n, shift_n, scale_n,
                        gate_mlp_n, ckpt_n, width,
                    ) in video_post_inputs:
                        mixed_n = mixed_video[:, cursor:cursor + width]
                        cursor += width
                        x_noisy_list[block_idx] = self._apply_post_with_optional_checkpoint(
                            block=block_v, residual_x=res_n, gate_msa=gate_n,
                            shift_mlp=shift_n, scale_mlp=scale_n, gate_mlp=gate_mlp_n,
                            use_gradient_checkpointing=ckpt_n, mixed_slice=mixed_n,
                            context_payload=(
                                noisy_video_context_payloads_list[block_idx]
                                if noisy_video_context_payloads_list is not None
                                else None
                            ),
                        )
                x_action = self._apply_post_with_optional_checkpoint(
                    block=block_a, residual_x=res_a, gate_msa=gate_a,
                    shift_mlp=shift_a, scale_mlp=scale_a, gate_mlp=gate_mlp_a,
                    use_gradient_checkpointing=ckpt_a,
                    mixed_slice=torch.cat(mixed_action_parts, dim=1),
                    context_payload=action_context_payload,
                )
                continue

            layer_post_block_indices = []
            layer_post_specs = []
            layer_ckpt_flags = []

            for anchor_idx in range(int(num_action_blocks)):
                block_start = anchor_idx * plan_subblocks
                block_end = block_start + plan_subblocks
                q_parts = []
                k_self_parts = []
                v_self_parts = []
                post_inputs = []
                for block_idx in range(block_start, block_end):
                    x_n = x_noisy_list[block_idx]
                    (q_n, k_n, v_n, res_n, gate_n, shift_n, scale_n, gate_mlp_n, ckpt_n
                     ) = self._build_expert_attention_io(
                        expert=video_expert, block=block_v, x=x_n,
                        freqs=noisy_video_freqs_list[block_idx],
                        t_mod=noisy_video_t_mod_list[block_idx],
                    )
                    q_parts.append(q_n)
                    k_self_parts.append(k_n)
                    v_self_parts.append(v_n)
                    post_inputs.append((block_idx, res_n, gate_n, shift_n, scale_n, gate_mlp_n, ckpt_n))

                q_group = torch.cat(q_parts, dim=1)
                k_group = torch.cat(k_self_parts, dim=1)
                v_group = torch.cat(v_self_parts, dim=1)
                visible_end = far_token_count + (num_obs_latents + anchor_idx) * tokens_per_frame
                visible_end = min(visible_end, S_clean)
                ctx_mask = noisy_attention_masks[anchor_idx]
                mixed_group = flash_attention(
                    q=q_group,
                    k=torch.cat([layer_k_clean[:, :visible_end], k_group], dim=1),
                    v=torch.cat([layer_v_clean[:, :visible_end], v_group], dim=1),
                    num_heads=self.num_heads,
                    ctx_mask=ctx_mask,
                )
                cursor = 0
                for block_idx, res_n, gate_n, shift_n, scale_n, gate_mlp_n, ckpt_n in post_inputs:
                    width = q_parts[block_idx - block_start].shape[1]
                    mixed_n = mixed_group[:, cursor:cursor + width]
                    cursor += width
                    layer_post_block_indices.append(block_idx)
                    layer_post_specs.append((
                        mixed_n,
                        res_n,
                        gate_n,
                        shift_n,
                        scale_n,
                        gate_mlp_n,
                        (
                            noisy_video_context_payloads_list[block_idx]
                            if noisy_video_context_payloads_list is not None
                            else None
                        ),
                    ))
                    layer_ckpt_flags.append(ckpt_n)

            # Post blocks are independent per noisy block, so deferring them
            # past the anchor loop is semantics-preserving. When every block's
            # cross-attention shares the same context tensor (and checkpoint
            # flags agree), one combined checkpoint computes the context K/V
            # once per layer instead of once per block.
            use_shared_posts = (
                len(layer_post_specs) > 0
                and all(flag == layer_ckpt_flags[0] for flag in layer_ckpt_flags)
                and self._payloads_share_context(layer_post_specs)
            )
            if use_shared_posts:
                updated = self._apply_noisy_posts_shared_context(
                    block=block_v,
                    post_specs=layer_post_specs,
                    use_gradient_checkpointing=layer_ckpt_flags[0],
                )
                for block_idx, x_new in zip(layer_post_block_indices, updated):
                    x_noisy_list[block_idx] = x_new
            else:
                for block_idx, spec, ckpt_n in zip(
                    layer_post_block_indices, layer_post_specs, layer_ckpt_flags
                ):
                    (mixed_n, res_n, gate_n, shift_n, scale_n, gate_mlp_n, payload) = spec
                    x_noisy_list[block_idx] = self._apply_post_with_optional_checkpoint(
                        block=block_v, residual_x=res_n, gate_msa=gate_n,
                        shift_mlp=shift_n, scale_mlp=scale_n, gate_mlp=gate_mlp_n,
                        use_gradient_checkpointing=ckpt_n, mixed_slice=mixed_n,
                        context_payload=payload,
                    )

            x_action = self._forward_action_layer_from_clean_cache(
                layer_idx=layer_idx,
                cache=cache,
                x_action=x_action,
                action_freqs=action_freqs,
                action_t_mod=action_t_mod,
                action_context_payload=action_context_payload,
                num_obs_latents=num_obs_latents,
                num_action_blocks=int(num_action_blocks),
                action_attention_masks=action_attention_masks,
            )

        return {"noisy_video": x_noisy_list, "action": x_action}

    def _build_action_attention_masks_from_clean_cache(
        self,
        *,
        cache: dict,
        x_action: torch.Tensor,
        num_obs_latents: int,
        num_action_blocks: int,
    ) -> list[Optional[torch.Tensor]]:
        """Build layer-invariant cached-AR action masks once per forward."""
        action_seq_len = int(x_action.shape[1])
        blocks = max(1, int(num_action_blocks))
        if int(num_action_blocks) < 0:
            raise ValueError("num_action_blocks must be non-negative")
        if action_seq_len == 0:
            return [None] * blocks
        if int(num_action_blocks) > 0 and action_seq_len % int(num_action_blocks) != 0:
            raise ValueError(
                "action_seq_len must be divisible by num_action_blocks"
            )
        action_per_block = (
            action_seq_len // int(num_action_blocks)
            if int(num_action_blocks) > 0
            else action_seq_len
        )
        history_valid_mask = cache["history_valid_mask"]
        far_token_count = int(cache["far_token_count"])
        tokens_per_frame = int(cache["tokens_per_frame"])
        first_clean_k = cache["clean_kv"][0][0]
        clean_seq_len = int(first_clean_k.shape[1])
        masks: list[Optional[torch.Tensor]] = []
        for block_idx in range(blocks):
            visible_end = min(
                far_token_count
                + (int(num_obs_latents) + block_idx) * tokens_per_frame,
                clean_seq_len,
            )
            if not (history_valid_mask is not None and far_token_count > 0):
                masks.append(None)
                continue

            batch_size = int(x_action.shape[0])
            masks.append(
                self._prepare_attention_mask(
                    torch.cat(
                        [
                            history_valid_mask,
                            self._cached_ones_bool(
                                batch_size,
                                visible_end - far_token_count,
                                x_action.device,
                            ),
                            self._cached_ones_bool(
                                batch_size,
                                action_per_block,
                                x_action.device,
                            ),
                        ],
                        dim=1,
                    ).unsqueeze(1),
                    dtype=x_action.dtype,
                    device=x_action.device,
                )
            )
        return masks

    def _forward_action_layer_from_clean_cache(
        self,
        *,
        layer_idx: int,
        cache: dict,
        x_action: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        num_obs_latents: int,
        num_action_blocks: int,
        action_attention_masks: Optional[list[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Run one cached-AR action layer without evaluating noisy video.

        This is the single implementation used by both the ordinary cached AR
        forward and the antithetic action-only branch.  Keeping visibility,
        history masking, and per-block action self-attention here prevents the
        paired path from silently drifting away from the main training path.
        """
        if x_action.shape[1] == 0:
            return x_action
        if not 0 <= int(layer_idx) < self.num_layers:
            raise ValueError(f"layer_idx must be in [0, {self.num_layers}), got {layer_idx}")
        if int(num_action_blocks) < 0:
            raise ValueError(f"num_action_blocks must be non-negative, got {num_action_blocks}")

        action_expert = self.mixtures["action"]
        block_a = action_expert.blocks[layer_idx]
        layer_k_clean, layer_v_clean = cache["clean_kv"][layer_idx]
        history_valid_mask = cache["history_valid_mask"]
        far_token_count = int(cache["far_token_count"])
        tokens_per_frame = int(cache["tokens_per_frame"])
        clean_seq_len = int(layer_k_clean.shape[1])

        action_seq_len = int(x_action.shape[1])
        blocks = max(1, int(num_action_blocks))
        if int(num_action_blocks) > 0 and action_seq_len % int(num_action_blocks) != 0:
            raise ValueError(
                f"action_seq_len={action_seq_len} must be divisible by "
                f"num_action_blocks={num_action_blocks}"
            )
        action_per_block = (
            action_seq_len // int(num_action_blocks)
            if int(num_action_blocks) > 0
            else action_seq_len
        )
        (
            q_a,
            k_a,
            v_a,
            res_a,
            gate_a,
            shift_a,
            scale_a,
            gate_mlp_a,
            ckpt_a,
        ) = self._build_expert_attention_io(
            expert=action_expert,
            block=block_a,
            x=x_action,
            freqs=action_freqs,
            t_mod=action_t_mod,
        )

        mixed_parts = []
        for block_idx in range(blocks):
            action_start = block_idx * action_per_block
            action_end = (
                action_seq_len
                if block_idx == blocks - 1
                else action_start + action_per_block
            )
            q_i = q_a[:, action_start:action_end]
            k_i = k_a[:, action_start:action_end]
            v_i = v_a[:, action_start:action_end]
            visible_end = far_token_count + (
                int(num_obs_latents) + block_idx
            ) * tokens_per_frame
            visible_end = min(visible_end, clean_seq_len)

            context_mask = (
                action_attention_masks[block_idx]
                if action_attention_masks is not None
                else self._build_action_attention_masks_from_clean_cache(
                    cache=cache,
                    x_action=x_action,
                    num_obs_latents=num_obs_latents,
                    num_action_blocks=num_action_blocks,
                )[block_idx]
            )

            mixed_parts.append(
                flash_attention(
                    q=q_i,
                    k=torch.cat([layer_k_clean[:, :visible_end], k_i], dim=1),
                    v=torch.cat([layer_v_clean[:, :visible_end], v_i], dim=1),
                    num_heads=self.num_heads,
                    ctx_mask=context_mask,
                )
            )

        x_action = self._apply_post_with_optional_checkpoint(
            block=block_a,
            residual_x=res_a,
            gate_msa=gate_a,
            shift_mlp=shift_a,
            scale_mlp=scale_a,
            gate_mlp=gate_mlp_a,
            use_gradient_checkpointing=ckpt_a,
            mixed_slice=torch.cat(mixed_parts, dim=1),
            context_payload=action_context_payload,
        )
        return x_action


    def forward_action_from_clean_cache(
        self,
        *,
        cache: dict,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        action_context_payload: Optional[dict],
        num_obs_latents: int,
        num_action_blocks: int,
    ) -> torch.Tensor:
        """Run an action-only AR branch against an existing clean KV cache."""
        self._assert_clean_context_payload_stable(cache, video_context_payload)
        x_action = action_tokens
        action_attention_masks = self._build_action_attention_masks_from_clean_cache(
            cache=cache,
            x_action=x_action,
            num_obs_latents=num_obs_latents,
            num_action_blocks=num_action_blocks,
        )
        for layer_idx in range(self.num_layers):
            x_action = self._forward_action_layer_from_clean_cache(
                layer_idx=layer_idx,
                cache=cache,
                x_action=x_action,
                action_freqs=action_freqs,
                action_t_mod=action_t_mod,
                action_context_payload=action_context_payload,
                num_obs_latents=num_obs_latents,
                num_action_blocks=num_action_blocks,
                action_attention_masks=action_attention_masks,
            )
        return x_action

    def append_to_clean_kv_cache(
        self,
        cache: dict,
        new_tokens: torch.Tensor,
        new_freqs: torch.Tensor,
        new_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
    ) -> dict:
        """Append new latent frame tokens to existing clean KV cache.

        ``video_context_payload`` must be the same cache-stationary context
        used by prefill. Dynamic state/proprio must not be supplied here.
        """
        self._assert_clean_context_payload_stable(cache, video_context_payload)
        video_expert = self.mixtures["video"]

        x_new = new_tokens
        history_valid_mask = cache["history_valid_mask"]
        far_token_count = cache["far_token_count"]

        for layer_idx in range(self.num_layers):
            block_v = video_expert.blocks[layer_idx]
            (q_n, k_n, v_n, res_n, gate_n, shift_n, scale_n, gate_mlp_n, ckpt_n
             ) = self._build_expert_attention_io(
                expert=video_expert, block=block_v, x=x_new,
                freqs=new_freqs, t_mod=new_t_mod,
            )
            # Attend new tokens to all cached clean K/V + self
            old_k, old_v = cache["clean_kv"][layer_idx]
            k_cat = torch.cat([old_k, k_n], dim=1)
            v_cat = torch.cat([old_v, v_n], dim=1)

            # Build mask: new tokens see all valid clean + self
            ctx_mask_new = None
            if history_valid_mask is not None and far_token_count > 0:
                B = q_n.shape[0]
                clean_tokens_count = old_k.shape[1] - far_token_count
                kv_valid = torch.cat([
                    history_valid_mask,
                    torch.ones(B, clean_tokens_count, dtype=torch.bool, device=q_n.device),
                    torch.ones(B, k_n.shape[1], dtype=torch.bool, device=q_n.device),
                ], dim=1)
                ctx_mask_new = kv_valid[:, None, :]

            mixed_n = flash_attention(q=q_n, k=k_cat, v=v_cat, num_heads=self.num_heads, ctx_mask=ctx_mask_new)
            x_new = self._apply_post_with_optional_checkpoint(
                block=block_v, residual_x=res_n, gate_msa=gate_n,
                shift_mlp=shift_n, scale_mlp=scale_n, gate_mlp=gate_mlp_n,
                use_gradient_checkpointing=ckpt_n, mixed_slice=mixed_n,
                context_payload=video_context_payload,
            )
            # Append new K/V to cache
            cache["clean_kv"][layer_idx] = (k_cat, v_cat)

        cache["x_clean"] = torch.cat([cache["x_clean"], x_new], dim=1)
        return cache

    def forward_ar_single_step(
        self,
        cache: dict,
        noisy_video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        noisy_video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        noisy_video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        action_context_payload: Optional[dict],
        num_obs_latents: int,  # reserved; visibility is derived from cache length
        noisy_video_context_payload: Optional[dict] = None,
    ) -> dict[str, torch.Tensor]:
        """Denoise 1 noisy block + action using cached clean K/V.

        Per-diffusion-step forward within an AR step. The clean K/V cache
        is fixed across all diffusion steps. Returns ``{"noisy_video",
        "action"}`` denoised token dict.
        """
        self._assert_clean_context_payload_stable(cache, video_context_payload)
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]

        history_valid_mask = cache["history_valid_mask"]
        far_token_count = cache["far_token_count"]

        x_noisy = noisy_video_tokens
        x_action = action_tokens

        for layer_idx in range(self.num_layers):
            block_v = video_expert.blocks[layer_idx]
            layer_k_clean, layer_v_clean = cache["clean_kv"][layer_idx]

            # Noisy block attention
            (q_n, k_n, v_n, res_n, gate_n, shift_n, scale_n, gate_mlp_n, ckpt_n
             ) = self._build_expert_attention_io(
                expert=video_expert, block=block_v, x=x_noisy,
                freqs=noisy_video_freqs, t_mod=noisy_video_t_mod,
            )
            if self.video_action_attention_mode == "joint_bidirectional":
                if x_action.shape[1] <= 0:
                    raise ValueError(
                        "joint_bidirectional AR inference requires noisy action tokens."
                    )
                block_a = action_expert.blocks[layer_idx]
                (q_a, k_a, v_a, res_a, gate_a, shift_a, scale_a, gate_mlp_a, ckpt_a
                 ) = self._build_expert_attention_io(
                    expert=action_expert, block=block_a, x=x_action,
                    freqs=action_freqs, t_mod=action_t_mod,
                )
                joint_mask = None
                if history_valid_mask is not None and far_token_count > 0:
                    batch_size = int(q_n.shape[0])
                    clean_token_count = int(layer_k_clean.shape[1]) - far_token_count
                    valid_parts = [
                        history_valid_mask,
                        self._cached_ones_bool(
                            batch_size, clean_token_count, q_n.device
                        ),
                        self._cached_ones_bool(
                            batch_size, int(k_n.shape[1]), q_n.device
                        ),
                        self._cached_ones_bool(
                            batch_size, int(k_a.shape[1]), q_n.device
                        ),
                    ]
                    joint_mask = self._prepare_attention_mask(
                        torch.cat(valid_parts, dim=1).unsqueeze(1),
                        dtype=q_n.dtype, device=q_n.device,
                    )
                q_joint = torch.cat([q_n, q_a], dim=1)
                mixed_joint = flash_attention(
                    q=q_joint,
                    k=torch.cat([layer_k_clean, k_n, k_a], dim=1),
                    v=torch.cat([layer_v_clean, v_n, v_a], dim=1),
                    num_heads=self.num_heads,
                    ctx_mask=joint_mask,
                )
                noisy_width = int(q_n.shape[1])
                x_noisy = self._apply_post_with_optional_checkpoint(
                    block=block_v, residual_x=res_n, gate_msa=gate_n,
                    shift_mlp=shift_n, scale_mlp=scale_n, gate_mlp=gate_mlp_n,
                    use_gradient_checkpointing=ckpt_n,
                    mixed_slice=mixed_joint[:, :noisy_width],
                    context_payload=noisy_video_context_payload,
                )
                x_action = self._apply_post_with_optional_checkpoint(
                    block=block_a, residual_x=res_a, gate_msa=gate_a,
                    shift_mlp=shift_a, scale_mlp=scale_a, gate_mlp=gate_mlp_a,
                    use_gradient_checkpointing=ckpt_a,
                    mixed_slice=mixed_joint[:, noisy_width:],
                    context_payload=action_context_payload,
                )
                continue

            k_cat_n = torch.cat([layer_k_clean, k_n], dim=1)
            v_cat_n = torch.cat([layer_v_clean, v_n], dim=1)

            ctx_mask_n = None
            if history_valid_mask is not None and far_token_count > 0:
                B = q_n.shape[0]
                clean_tokens_count = layer_k_clean.shape[1] - far_token_count
                kv_valid = torch.cat([
                    history_valid_mask,
                    torch.ones(B, clean_tokens_count, dtype=torch.bool, device=q_n.device),
                    torch.ones(B, k_n.shape[1], dtype=torch.bool, device=q_n.device),
                ], dim=1)
                ctx_mask_n = kv_valid[:, None, :]

            mixed_n = flash_attention(q=q_n, k=k_cat_n, v=v_cat_n, num_heads=self.num_heads, ctx_mask=ctx_mask_n)
            x_noisy = self._apply_post_with_optional_checkpoint(
                block=block_v, residual_x=res_n, gate_msa=gate_n,
                shift_mlp=shift_n, scale_mlp=scale_n, gate_mlp=gate_mlp_n,
                use_gradient_checkpointing=ckpt_n, mixed_slice=mixed_n,
                context_payload=noisy_video_context_payload,
            )

            # Action block attention
            if x_action.shape[1] > 0:
                block_a = action_expert.blocks[layer_idx]
                (q_a, k_a, v_a, res_a, gate_a, shift_a, scale_a, gate_mlp_a, ckpt_a
                 ) = self._build_expert_attention_io(
                    expert=action_expert, block=block_a, x=x_action,
                    freqs=action_freqs, t_mod=action_t_mod,
                )

                # Action sees clean visual tokens + self (not noisy video).
                k_cat_a = torch.cat([layer_k_clean, k_a], dim=1)
                v_cat_a = torch.cat([layer_v_clean, v_a], dim=1)

                ctx_mask_a = None
                if history_valid_mask is not None and far_token_count > 0:
                    B = q_a.shape[0]
                    clean_tokens_count = layer_k_clean.shape[1] - far_token_count
                    kv_valid_a = torch.cat([
                        history_valid_mask,
                        torch.ones(B, clean_tokens_count, dtype=torch.bool, device=q_a.device),
                        torch.ones(B, k_a.shape[1], dtype=torch.bool, device=q_a.device),
                    ], dim=1)
                    ctx_mask_a = kv_valid_a[:, None, :]

                mixed_a = flash_attention(q=q_a, k=k_cat_a, v=v_cat_a, num_heads=self.num_heads, ctx_mask=ctx_mask_a)
                x_action = self._apply_post_with_optional_checkpoint(
                    block=block_a, residual_x=res_a, gate_msa=gate_a,
                    shift_mlp=shift_a, scale_mlp=scale_a, gate_mlp=gate_mlp_a,
                    use_gradient_checkpointing=ckpt_a, mixed_slice=mixed_a,
                    context_payload=action_context_payload,
                )

        return {"noisy_video": x_noisy, "action": x_action}

    @staticmethod
    def _build_clean_causal_mask(
        far_token_count: int,
        num_clean_frames: int,
        tokens_per_frame: int,
        device: torch.device,
        history_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Blockwise causal mask for [far + clean] tokens.

        Far prefix: all valid far tokens see each other. Clean frames:
        frame ``i`` sees far + frames 0..i (inclusive).

        If ``history_valid_mask`` is given, returns ``[B, total, total]`` with
        per-sample history validity; otherwise ``[total, total]``.
        """
        H = far_token_count
        total = H + num_clean_frames * tokens_per_frame

        struct = torch.zeros((total, total), dtype=torch.bool, device=device)

        # Far prefix: full self-attention
        if H > 0:
            struct[:H, :H] = True

        # Clean frames: blockwise causal
        c_start = H
        for i in range(num_clean_frames):
            row_s = c_start + i * tokens_per_frame
            row_e = row_s + tokens_per_frame
            # See far
            if H > 0:
                struct[row_s:row_e, :H] = True
            # See clean[0..i] inclusive
            struct[row_s:row_e, c_start:row_e] = True

        if history_valid_mask is None:
            return struct

        # Per-sample masking: nobody attends to invalid history columns
        B = history_valid_mask.shape[0]
        mask = struct.unsqueeze(0).expand(B, -1, -1).clone()

        # Nobody attends to invalid history columns
        mask[:, :, :H] &= history_valid_mask.unsqueeze(1)  # [B, total, H]

        # Invalid history rows attend to nothing except self (avoid empty softmax)
        mask[:, :H, :] &= history_valid_mask.unsqueeze(2)  # [B, H, total]
        diag = torch.arange(H, device=device)
        mask[:, diag, diag] = True

        return mask
