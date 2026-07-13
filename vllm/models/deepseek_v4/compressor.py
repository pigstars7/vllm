# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
import os
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_compressed_slot_mapping,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    save_partial_states,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


def _vllm_v4_minmax(t: torch.Tensor | None) -> tuple[int | None, int | None]:
    if t is None or t.numel() == 0:
        return None, None
    return int(t.min().item()), int(t.max().item())


def _vllm_v4_kv_block_table_from_metadata(k_cache_metadata: Any) -> torch.Tensor | None:
    for container in (
        getattr(k_cache_metadata, "prefill", None),
        getattr(k_cache_metadata, "decode", None),
        k_cache_metadata,
    ):
        if container is None:
            continue
        block_table = getattr(container, "block_table", None)
        if block_table is not None:
            return block_table
        block_table = getattr(container, "block_table_tensor", None)
        if block_table is not None:
            return block_table
    return None


def _vllm_v4_rebuild_compressed_kv_slot_mapping_torch(
    module: nn.Module,
    *,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor | None,
    k_cache_metadata: Any,
    kv_cache: torch.Tensor,
    num_actual: int,
) -> torch.Tensor:
    # _vllm_v4_compressor_torch_rebuild_kv_slot_mapping:
    # Rebuild the final MLA KV-cache slot mapping without calling the Triton
    # helper. The helper uses the right formula, but in packed long-prefill
    # cases it hit IMA inside its own block-table load. The fused compressor
    # still needs a sane compressed mapping immediately before writing KV.
    current = getattr(k_cache_metadata, "slot_mapping", None)
    fallback = (
        current[:num_actual].clone()
        if current is not None
        else torch.full((num_actual,), -1, dtype=torch.int64, device=positions.device)
    )
    if num_actual == 0:
        return fallback

    prefix = getattr(module, "prefix", module.__class__.__name__)
    block_table = _vllm_v4_kv_block_table_from_metadata(k_cache_metadata)
    kv_block_size = int(kv_cache.shape[1]) if kv_cache.dim() > 1 else 0
    compress_ratio = int(getattr(module, "compress_ratio", 1))
    logical_block_size = kv_block_size * compress_ratio
    if block_table is None or block_table.dim() < 2 or kv_block_size <= 0:
        if os.environ.get("VLLM_DSV4_COMPRESSOR_BOUNDARY_DIAG") == "1":
            print(
                "[vllm-v4-compressor-rebuild] "
                f"pid={os.getpid()} prefix={prefix} fallback_missing_metadata "
                f"block_table_shape={None if block_table is None else tuple(block_table.shape)} "
                f"kv_block_size={kv_block_size}",
                flush=True,
            )
        return fallback

    block_table = block_table.to(device=positions.device)
    positions_i64 = positions[:num_actual].to(torch.long)
    if token_to_req_indices is None:
        req_indices = torch.zeros((num_actual,), dtype=torch.long, device=positions.device)
    else:
        req_indices = token_to_req_indices[:num_actual].to(
            device=positions.device, dtype=torch.long
        )

    compressed_pos = positions_i64 // compress_ratio
    # The block table is indexed by the semantic/original KV block size
    # (for DeepSeek-V4 C4: 256 tokens), while the physical cache stores
    # compressed slots with storage_block_size=64.
    block_ids = positions_i64 // logical_block_size
    valid = (
        (positions_i64 >= 0)
        & (((positions_i64 + 1) % compress_ratio) == 0)
        & (req_indices >= 0)
        & (req_indices < int(block_table.shape[0]))
        & (block_ids >= 0)
        & (block_ids < int(block_table.shape[1]))
    )
    safe_req = req_indices.clamp(min=0, max=int(block_table.shape[0]) - 1)
    safe_block = block_ids.clamp(min=0, max=int(block_table.shape[1]) - 1)
    block_numbers = block_table[safe_req, safe_block].to(torch.long)
    slot_ids = block_numbers * kv_block_size + (compressed_pos % kv_block_size)
    valid = valid & (block_numbers >= 0)
    rebuilt = torch.where(
        valid,
        slot_ids,
        torch.full_like(slot_ids, -1),
    )

    if (
        os.environ.get("VLLM_DSV4_COMPRESSOR_BOUNDARY_DIAG") == "1"
        and "model.layers.16" in prefix
    ):
        rb_min, rb_max = _vllm_v4_minmax(rebuilt)
        cur_min, cur_max = _vllm_v4_minmax(current)
        print(
            "[vllm-v4-compressor-rebuild] "
            f"pid={os.getpid()} prefix={prefix} "
            f"num_actual={num_actual} kv_block_size={kv_block_size} "
            f"logical_block_size={logical_block_size} "
            f"block_table_shape={tuple(block_table.shape)} "
            f"current_minmax=({cur_min},{cur_max}) rebuilt_minmax=({rb_min},{rb_max})",
            flush=True,
        )
    elif os.environ.get("VLLM_DSV4_COMPRESSOR_REBUILD_WARN") == "1":
        rb_min, rb_max = _vllm_v4_minmax(rebuilt)
        cur_min, cur_max = _vllm_v4_minmax(current)
        should_warn = (
            (cur_min is not None and cur_min < -1)
            or (cur_max is not None and cur_max >= int(kv_cache.shape[0]) * kv_block_size)
            or (rb_min == -1 and rb_max == -1 and int((positions_i64 >= 0).sum().item()) > 0)
        )
        if should_warn:
            print(
                "[vllm-v4-compressor-rebuild-warn] "
                f"pid={os.getpid()} prefix={prefix} "
                f"compress_ratio={compress_ratio} num_actual={num_actual} "
                f"kv_block_size={kv_block_size} logical_block_size={logical_block_size} "
                f"block_table_shape={tuple(block_table.shape)} "
                f"positions_minmax={_vllm_v4_minmax(positions[:num_actual])} "
                f"req_minmax={_vllm_v4_minmax(token_to_req_indices)} "
                f"current_minmax=({cur_min},{cur_max}) rebuilt_minmax=({rb_min},{rb_max})",
                flush=True,
            )
    return rebuilt


def _vllm_v4_compressor_boundary_diag(
    module: nn.Module,
    *,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    token_to_req_indices: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    state_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    num_actual: int,
) -> None:
    # _vllm_v4_compressor_boundary_diag: metadata bounds before compressed KV write.
    if os.environ.get("VLLM_DSV4_COMPRESSOR_BOUNDARY_DIAG") != "1":
        return
    prefix = getattr(module, "prefix", module.__class__.__name__)
    # Keep logs narrow: the crash is the second chunk and layer16, but printing
    # layer16 first chunk is still useful as a known-good comparator.
    if "model.layers.16" not in prefix and int(positions.shape[0]) == 4096:
        return

    try:
        kv_slot_mapping = getattr(k_cache_metadata, "slot_mapping", None)
        pos_min, pos_max = _vllm_v4_minmax(positions)
        slot_min, slot_max = _vllm_v4_minmax(slot_mapping)
        kv_slot_min, kv_slot_max = _vllm_v4_minmax(kv_slot_mapping)
        req_min, req_max = _vllm_v4_minmax(token_to_req_indices)
        bt_min, bt_max = _vllm_v4_minmax(block_table)

        actual_positions = positions[:num_actual]
        boundary = (actual_positions + 1) % getattr(module, "compress_ratio", 1) == 0
        boundary_count = int(boundary.sum().item())
        if boundary_count:
            boundary_indices = torch.nonzero(boundary, as_tuple=False).flatten()
            boundary_idx_min = int(boundary_indices.min().item())
            boundary_idx_max = int(boundary_indices.max().item())
        else:
            boundary_idx_min = boundary_idx_max = None

        state_total_slots = int(state_cache.shape[0]) * int(block_size)
        kv_cache_block_size = int(kv_cache.shape[1]) if kv_cache.dim() > 1 else None
        kv_total_slots = (
            int(kv_cache.shape[0]) * kv_cache_block_size
            if kv_cache_block_size is not None
            else None
        )
        print(
            "[vllm-v4-compressor-diag] "
            f"pid={os.getpid()} prefix={prefix} "
            f"compress_ratio={getattr(module, 'compress_ratio', None)} "
            f"num_actual={num_actual} positions_shape={tuple(positions.shape)} "
            f"positions_minmax=({pos_min},{pos_max}) "
            f"slot_shape={tuple(slot_mapping.shape)} slot_minmax=({slot_min},{slot_max}) "
            f"state_cache_shape={tuple(state_cache.shape)} state_total_slots={state_total_slots} "
            f"kv_slot_shape={None if kv_slot_mapping is None else tuple(kv_slot_mapping.shape)} "
            f"kv_slot_minmax=({kv_slot_min},{kv_slot_max}) "
            f"kv_cache_shape={tuple(kv_cache.shape)} kv_total_slots={kv_total_slots} "
            f"block_table_shape={tuple(block_table.shape)} block_table_minmax=({bt_min},{bt_max}) "
            f"block_size={block_size} req_minmax=({req_min},{req_max}) "
            f"boundary_count={boundary_count} boundary_idx_minmax=({boundary_idx_min},{boundary_idx_max})",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as exc:
        print(
            f"[vllm-v4-compressor-diag] pid={os.getpid()} prefix={prefix} diag_error={exc!r}",
            flush=True,
        )


class CompressorBackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CompressorBackend"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 1024]

    @staticmethod
    def get_builder_cls() -> type["CompressorMetadataBuilder"]:
        return CompressorMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class CompressorMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]


class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.block_size = mla_spec.block_size

        self.token_to_req_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_reqs = common_attn_metadata.num_reqs
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
        token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
        token_to_req_indices.copy_(x, non_blocking=True)
        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
        )


class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        # Block size is constrained by tensor sharing between compressor states
        # and KV blocks. Since compressor states share the same physical tensor
        # as KV blocks, they must use the same page size.
        # The KV block shape [256//4, head_dim] = [64, 584] determines:
        # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
        # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
        # TODO(yifan): make block size automatically determined and configurable.
        if compress_ratio == 4:
            self.block_size = 4
        elif compress_ratio == 128:
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # FlashMLA's UE8M0 paged layout needs 576B alignment; the FlashInfer
        # full-cache path shares state pages with contiguous KV pages, so
        # padding would break page matching.
        is_flashmla = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576 if is_flashmla else None,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CompressorBackend


class DeepseekCompressor(nn.Module):
    """DeepSeek V4 KV/score compressor.

    Owns the linear / norm / state-cache / ape state and the shared forward
    prologue (kv/score split, save_partial_states launch). The
    compress → norm → RoPE → store step is dispatched to a triton kernel
    (``compress_norm_rope_store_triton``) by default, except for the NVIDIA
    head_dim=128 indexer path which uses the cutedsl kernel
    (``compress_norm_rope_store_cutedsl``) for better performance.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.use_fp4_cache = use_fp4_cache

        config = vllm_config.model_config.hf_config
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap

        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(
                (compress_ratio, self.coff * self.head_dim),
                dtype=state_dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            [self.coff * self.head_dim, self.coff * self.head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        self.state_cache = CompressorStateCache(
            state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
            dtype=state_dtype,
            compress_ratio=compress_ratio,
            prefix=f"{prefix}.state_cache",
        )

        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

        if self.head_dim == 512:
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            self._quant_block = 64
            self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
            self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
        elif self.head_dim == 128:
            if use_fp4_cache:
                self._quant_block = MXFP4_BLOCK_SIZE
                self._token_stride = self.head_dim // 2
                self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
            else:
                self._quant_block = 128
                self._token_stride = self.head_dim
                self._scale_dim = 4  # single float32 scale
        else:
            raise ValueError(
                f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
            )

    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        # Get the metadata and handle dummy profiling run.
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2
        pdl_kwargs = (
            {}
            if current_platform.is_rocm() or current_platform.is_xpu()
            else {"launch_pdl": False}
        )

        # Store the KV and score (with fused APE addition) in the state.
        # NOTE: PDL is disabled — both this kernel and the compress kernels
        # below depend on preceding kernel outputs (kv/score from the cublas
        # GEMM; state_cache from this kernel) but neither emits/waits on PDL
        # grid dependency primitives, so launch_pdl=True caused a
        # read-after-write race and non-deterministic output.
        save_partial_states(
            kv=kv,
            score=score,
            ape=self.ape,
            positions=positions,
            state_cache=state_cache,
            slot_mapping=slot_mapping,
            block_size=block_size,
            state_width=state_width,
            compress_ratio=self.compress_ratio,
            pdl_kwargs=pdl_kwargs,
        )

        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
        # RoPE requirements (kernel applies forward GPT-J style rotation):
        # - is_neox_style=False (interleaved pairs, NOT split-half)
        # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
        #   second half sin (per-pair, length rope_head_dim // 2 each)
        # - applied to LAST rope_head_dim elements of head_dim
        # - position used: (positions // compress_ratio) * compress_ratio
        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache

        if (
            os.environ.get("VLLM_DSV4_COMPRESSOR_REBUILD_KV_SLOT_MAPPING") == "1"
            and self.compress_ratio > 1
            and ".indexer." not in self.prefix
        ):
            # _vllm_v4_compressor_rebuild_kv_slot_mapping:
            # The backend metadata may share a compressed slot_mapping buffer
            # that is later overwritten by another layer/group before this
            # compressor runs. Rebuild the final MLA KV-cache mapping with
            # torch from the current positions and the final KV block table.
            k_cache_metadata.slot_mapping = _vllm_v4_rebuild_compressed_kv_slot_mapping_torch(
                self,
                positions=positions,
                token_to_req_indices=token_to_req_indices,
                k_cache_metadata=k_cache_metadata,
                kv_cache=kv_cache,
                num_actual=num_actual,
            )

        _vllm_v4_compressor_boundary_diag(
            self,
            positions=positions,
            slot_mapping=slot_mapping,
            token_to_req_indices=token_to_req_indices,
            block_table=block_table,
            block_size=block_size,
            state_cache=state_cache,
            kv_cache=kv_cache,
            k_cache_metadata=k_cache_metadata,
            num_actual=num_actual,
        )

        # FlashInfer V4 reads a contiguous bf16 / per-tensor fp8 cache row; the
        # legacy FlashMLA path uses the UE8M0 paged uint8 layout.
        store_full_kv = self.head_dim == 512 and kv_cache.dtype != torch.uint8
        store_full_fp8 = kv_cache.dtype == torch.float8_e4m3fn
        fp8_scale = (
            getattr(k_cache_layer, "_flashinfer_fp8_kv_scale", None)
            if store_full_fp8
            else None
        )

        # cutedsl (head=512) accepts the full-cache flags; triton (indexer/AMD)
        # does not, so the two callables have different signatures.
        compress_norm_rope_store_fn: Any
        if (current_platform.is_cuda() and self.head_dim == 512
                and __import__("os").environ.get(
                    "VLLM_DSV4_DISABLE_CUTEDSL_INDEXER") != "1"):
            from .nvidia.ops.sparse_attn_compress_cutedsl import (
                compress_norm_rope_store_cutedsl,
            )

            # head=512 on CUDA always uses cutedsl, for both the legacy UE8M0
            # layout and the FlashInfer full-cache layout. The full-cache flags
            # are consumed only here.
            compress_norm_rope_store_fn = compress_norm_rope_store_cutedsl
            extra_kwargs: dict[str, Any] = dict(
                store_full_kv=store_full_kv,
                store_full_fp8=store_full_fp8,
                fp8_scale=fp8_scale,
            )
        else:
            # Indexer path (head_dim == 128) or non-CUDA GPUs (AMD, XPU, etc.).
            compress_norm_rope_store_fn = compress_norm_rope_store_triton
            extra_kwargs = {}

        compress_norm_rope_store_fn(
            state_cache=state_cache,
            num_actual=num_actual,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=block_size,
            state_width=state_width,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=k_cache_metadata,
            pdl_kwargs=pdl_kwargs,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            use_fp4_cache=self.use_fp4_cache,
            rms_norm_weight=self.norm.weight,
            rms_norm_eps=self.rms_norm_eps,
            quant_block=self._quant_block,
            token_stride=self._token_stride,
            scale_dim=self._scale_dim,
            **extra_kwargs,
        )
