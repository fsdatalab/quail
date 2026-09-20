"""FP8 experts with fused input and activation quantization."""


class FP8Experts:
    """Run vLLM's routed matrix multiplications on prequantized input."""

    def __init__(self, module, engine):
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )

        self.module = module
        self.engine = engine
        self.weights = module.routed_experts
        method = self.weights.quant_method
        self.quant = method.moe_quant_config
        parallel = module.moe_config.moe_parallel_config
        if not (isinstance(method.moe_kernel.fused_experts, TritonExperts)
                and self.quant.use_fp8_w8a8
                and self.quant.per_act_token_quant
                and self.quant.block_shape is None
                and self.quant.a1_scale is None and self.quant.a2_scale is None
                and self.quant.w1_bias is None and self.quant.w2_bias is None
                and module.moe_config.activation == MoEActivation.GELU_TANH
                and parallel.tp_size == parallel.ep_size == parallel.dp_size == 1
                and not module.moe_config.is_lora_enabled
                and not self.weights.apply_router_weight_on_input
                and self.weights.expert_map is None):
            raise ValueError("Expert fusion requires local Triton FP8 experts "
                             "with dynamic per-token scales and GELU gating")

    def __call__(self, inputs, scales, logits):
        """Return expert output valid until the next workspace allocation."""
        import triton.language as tl
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            _prepare_expert_assignment,
            invoke_fused_moe_triton_kernel,
            try_get_optimal_moe_config,
        )
        from vllm.v1.worker.workspace import current_workspace_manager

        torch = self.engine.torch
        method = self.weights.quant_method
        # Gemma's router reads logits, not the quantized hidden states.
        top_weights, top_ids = self.module.router.select_experts(
            hidden_states=inputs, router_logits=logits,
            topk_indices_dtype=method.topk_indices_dtype)
        w1, w2 = self.weights.w13_weight, self.weights.w2_weight
        rows, hidden = inputs.shape
        experts, doubled, _ = w1.shape
        topk = top_ids.shape[1]
        config = try_get_optimal_moe_config(
            w1.shape, w2.shape, topk, self.quant.config_name(inputs.dtype), rows)
        sorted_ids, expert_ids, padded = _prepare_expert_assignment(
            top_ids, config, rows, topk, experts, None)
        workspace, output = current_workspace_manager().get_simultaneous(
            ((rows * topk * max(doubled, hidden),), torch.bfloat16),
            ((rows, hidden), torch.bfloat16))
        gate_up = workspace[:rows * topk * doubled].view(rows, topk, doubled)
        expert_out = workspace[:rows * topk * hidden].view(rows, topk, hidden)

        def multiply(x, weight, out, x_scale, weight_scale, count, weighted):
            invoke_fused_moe_triton_kernel(
                x, weight, out, x_scale, weight_scale,
                top_weights if weighted else None, sorted_ids, expert_ids,
                padded, weighted, count, config, compute_type=tl.bfloat16,
                use_fp8_w8a8=True, use_int8_w8a8=False, use_int8_w8a16=False,
                use_int4_w4a16=False, per_channel_quant=True)

        multiply(inputs, w1, gate_up, scales, self.quant.w1_scale, topk, False)
        activated, activation_scales = self.engine.gelu_mul_quant(
            gate_up.view(rows * topk, doubled), round_activation=True)
        multiply(activated, w2, expert_out, activation_scales,
                 self.quant.w2_scale, 1, True)
        ops.moe_sum(expert_out, output)
        return output
