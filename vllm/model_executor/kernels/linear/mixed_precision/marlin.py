# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MARLIN_SUPPORTED_GROUP_SIZES,
    apply_gptq_marlin_linear,
    marlin_act_int8_process_scales,
    marlin_make_empty,
    marlin_make_workspace_new,
    marlin_pad_dim,
    marlin_pad_qweight,
    marlin_pad_scales,
    marlin_padded_nk,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_zero_points,
    query_marlin_supported_quant_types,
    unpack_cols,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class MarlinLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        # Marlin uses inline PTX, so it can only be compatible with Nvidia
        if not current_platform.is_cuda():
            return False, "Marlin only supported on CUDA"

        quant_types = query_marlin_supported_quant_types(c.zero_points)
        if c.weight_type not in quant_types:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by"
                f"  Marlin, supported types are: {quant_types}",
            )

        if c.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Group size ({c.group_size}) not supported by "
                "Marlin, supported group sizes are: "
                f"{MARLIN_SUPPORTED_GROUP_SIZES}",
            )

        # A group straddling TP ranks cannot be fixed by padding.
        if (
            c.group_size != -1
            and c.group_size < c.full_weight_shape[0]
            and c.partition_weight_shape[0] % c.group_size != 0
        ):
            return False, (
                f"in_features per partition {c.partition_weight_shape[0]} is "
                f"not divisible by group_size = {c.group_size}."
            )

        # Tile misalignment is fixed by zero-padding at weight prep.
        return True, None

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = getattr(layer, self.w_q_name).device
        c = self.config
        is_a_8bit = c.act_type is not None and c.act_type.itemsize == 1

        from vllm.model_executor.layers.quantization.utils import lp_w4a8g

        self._lp_g8 = bool(
            is_a_8bit
            and lp_w4a8g.enabled()
            and c.weight_type == scalar_types.uint4
            and c.zero_points
            and c.group_size == 128
            and c.partition_weight_shape[0] % 128 == 0
        )
        if is_a_8bit:
            # Lane LP: the s8 x u4 (zero-point, AWQ-style) kernels are compiled
            # (generate_kernels.py "AWQ-INT4 with INT8 activation", sm75 too):
            # int4 - zp lands exactly in int8 [-15, 15] via sub_zp_and_dequant.
            assert c.weight_type in (scalar_types.uint4b8, scalar_types.uint4), (
                "W8A8 is not supported by marlin kernel."
            )

        if c.act_type == torch.float8_e4m3fn:
            ops.marlin_int4_fp8_preprocess(getattr(layer, self.w_q_name), inplace=True)
            getattr(layer, self.w_s_name).data = (
                getattr(layer, self.w_s_name).data * 512
            )

        size_k, size_n = c.partition_weight_shape
        padded_n, padded_k = marlin_padded_nk(size_n, size_k, c.group_size)
        if self._lp_g8 and (padded_n, padded_k) != (size_n, size_k):
            self._lp_g8 = False  # LP W4A8G: tile-aligned shapes only; fall back to stock W4A8
        self._lp_mode, self._lp_emax, self._lp_wglob = lp_w4a8g.mode(), lp_w4a8g.emax(), 1.0
        if self._lp_g8:
            lp_w4a8g.load_ext(self._lp_mode, self._lp_emax)  # JIT load (cached build) before graph capture

        # Allocate marlin workspace, reusing existing storage on reload.
        self.workspace = marlin_make_workspace_new(
            device, existing=getattr(self, "workspace", None)
        )

        # Default name since marlin requires empty parameter for zp,
        # TODO: remove this requirement from marlin (allow optional tensors)
        if self.w_zp_name is None:
            self.w_zp_name = "w_zp"

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = ops.gptq_marlin_repack(
                marlin_pad_qweight(
                    x.data.contiguous(), size_n, size_k, padded_n, padded_k
                ),
                size_k=padded_k,
                size_n=padded_n,
                num_bits=c.weight_type.size_bits,
                is_a_8bit=is_a_8bit,
            )
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = marlin_permute_scales(
                marlin_pad_scales(
                    x.data.contiguous(),
                    size_n,
                    size_k,
                    padded_n,
                    padded_k,
                    c.group_size,
                ),
                size_k=padded_k,
                size_n=padded_n,
                group_size=c.group_size,
                is_a_8bit=is_a_8bit,
            )

            if c.group_size == -1:
                num_groups = 1
            else:
                num_groups = c.partition_weight_shape[0] // c.group_size

            if c.act_type == torch.int8 and num_groups > 1 and self._lp_g8:
                # Lane LP W4A8G. mode g: keep the real fp16 group scales (kernel applies
                # a_scale[row, group] * w_scale[group, col] per int32 group partial).
                # mode e: stock-style int16 levels (fewer levels keep the int32 headroom
                # when the kernel shifts them left by EMAX - e[row, group]).
                if lp_w4a8g.mode() == "e":
                    x.data, wglob = lp_w4a8g.process_scales_e(x.data, lp_w4a8g.level())
                    self._lp_wglob = wglob
                layer.input_global_scale = None
            elif c.act_type == torch.int8 and num_groups > 1:
                x.data, input_global_scale = marlin_act_int8_process_scales(x.data)
                layer.register_parameter(
                    "input_global_scale",
                    torch.nn.Parameter(input_global_scale, requires_grad=False),
                )
            else:
                layer.input_global_scale = None
            return x

        if c.zero_points:
            grouped_k = size_k // c.group_size if c.group_size != -1 else 1
            padded_grouped_k = padded_k // c.group_size if c.group_size != -1 else 1
            self._transform_param(
                layer,
                self.w_zp_name,
                lambda x: marlin_zero_points(
                    marlin_pad_scales(
                        unpack_cols(
                            x.t(),
                            c.weight_type.size_bits,
                            grouped_k,
                            size_n,
                        ),
                        size_n,
                        size_k,
                        padded_n,
                        padded_k,
                        c.group_size,
                    ),
                    size_k=padded_grouped_k,
                    size_n=padded_n,
                    num_bits=c.weight_type.size_bits,
                    is_a_8bit=is_a_8bit,
                ),
            )
        else:
            setattr(layer, self.w_zp_name, marlin_make_empty(device))
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.data = marlin_permute_bias(
                marlin_pad_dim(layer.bias, size_n, padded_n)
            )

        # Lane LP W8X: large-M prefill on int8 IMMA (CUTLASS W8A8) from a just-in-time int8 expansion of these Marlin
        # tensors; decode keeps Marlin W4A16. Only s_ch [1, N] fp32 is new memory.
        from vllm.model_executor.layers.quantization.utils import lp_w8x

        self._lp_w8x = None
        if (
            lp_w8x.enabled()
            and not is_a_8bit
            and c.zero_points
            and c.weight_type == scalar_types.uint4
            and c.group_size == 128
            and (padded_n, padded_k) == (size_n, size_k)
            and size_n % 64 == 0
            and lp_w8x.allowed(getattr(self, "_lp_prefix", None))
        ):
            w_q, w_s, w_zp = self._get_weight_params(layer)
            self._lp_w8x = lp_w8x.W8XState(w_q, w_s, w_zp, size_k, size_n).s_ch
            self._lp_w8x_min_m = lp_w8x.min_m()

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp = self._get_weight_params(layer)
        if getattr(self, "_lp_w8x", None) is not None:
            x2 = x.reshape(-1, x.shape[-1])
            out = torch.ops.lp.w4_prefill_gemm(
                x2, w_q, w_s, w_zp, self._lp_w8x, self.workspace,
                c.partition_weight_shape[0], c.partition_weight_shape[1], self._lp_w8x_min_m,
            )
            if bias is not None:
                out = out + bias
            return out.reshape(x.shape[:-1] + (c.partition_weight_shape[1],))
        if getattr(self, "_lp_g8", False):
            x2 = x.reshape(-1, x.shape[-1])
            if x2.stride(-1) != 1 or x2.stride(0) % 16 != 0:
                x2 = x2.contiguous()
            out = torch.ops.lp.w4a8g_gemm(
                x2, w_q, w_s, w_zp, self.workspace, c.partition_weight_shape[1],
                self._lp_wglob, self._lp_emax, self._lp_mode,
            )
            if bias is not None:
                out = out + bias
            return out.reshape(x.shape[:-1] + (c.partition_weight_shape[1],))
        return apply_gptq_marlin_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            workspace=self.workspace,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],
            output_size_per_partition=c.partition_weight_shape[1],
            input_global_scale=getattr(layer, "input_global_scale", None),
            bias=bias,
            input_dtype=c.act_type,
        )
