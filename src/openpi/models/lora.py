import math
import re

import flax.linen as nn
import flax.struct as struct
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


@struct.dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    # LoRA rank.
    rank: int
    # LoRA scaling factor.
    alpha: float = 1.0
    # Initialization function for LoRA parameters.
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # Axes in the weight to apply LoRA to. Should typically be the last two axes.
    axes: tuple[int, int] = (-2, -1)
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.
    label: str = "L"
    # MoE
    num_experts: int = 1

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


class Einsum(nn.Module):
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum."""

    # Shape of the weight.
    shape: tuple[int, ...]
    # Initialization function for the weight.
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # Setup LoRA parameters.
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            
            if config.num_experts > 1:
                num_experts = config.num_experts
                self.w_a = self.param("lora_a_experts", config.init_fn, (num_experts, *shape_a))
                self.w_b = self.param("lora_b_experts", config.init_fn, (num_experts, *shape_b))
                # Router: 将输入的最后一个特征维度映射到专家概率
                in_features = self.shape[config.axes[0]]
                self.router = self.param("lora_router", nn.initializers.normal(stddev=0.02), (in_features, num_experts))
            else:
                self.w_a = self.param("lora_a", config.init_fn, shape_a)
                self.w_b = self.param("lora_b", config.init_fn, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x):
        dtype = x.dtype  # original dtype, could be half-precision
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            
            if config.num_experts > 1:
                # 1. 计算 Router 权重 (Token-level Soft Routing)
                router_logits = jnp.dot(x, self.router.astype(dtype))
                gate_weights = jax.nn.softmax(router_logits, axis=-1)
                
                # 2. 纯函数定义：单个专家如何计算 LoRA
                def compute_expert(wa, wb):
                    lora_a_out = jnp.einsum(eqn_a, x, wa.astype(dtype))
                    return jnp.einsum(eqn_b, lora_a_out, wb.astype(dtype))
                
                # 3. jax.vmap 自动处理多专家并行计算
                # 输出 lora_out 的形状变为: [E, *out_shape]
                lora_out = jax.vmap(compute_expert)(self.w_a, self.w_b)
                
                # 4. 动态 Einsum 混合门控权重
                # 提取输入的轴标签 (如 "BSD") 和输出的轴标签 (如 "3BSKH")
                lhs = eqn.split(',')[0]
                out_str = eqn.split('->')[1]
                
                # 自动构建路由 Einsum，例如: "BSE,E3BSKH->3BSKH"
                gate_eqn = f"{lhs[:-1]}E,E{out_str}->{out_str}"
                
                lora = jnp.einsum(gate_eqn, gate_weights, lora_out)
                
                # 寻找那些在 lhs 中存在，但在 out_str 中消失的额外维度（例如 BTNH -> BTD 里的 N）
                extra_chars = [c for c in lhs[:-1] if c not in out_str]
                scale_factor = 1.0
                if extra_chars:
                    for c in extra_chars:
                        dim_idx = lhs.index(c)
                        scale_factor *= x.shape[dim_idx]
                    
                    # 将错误累加的倍数除回来，把 Sum 还原为 Mean
                    lora = lora / scale_factor
                
                result = result + lora * config.scaling_value
            
            else:
                lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
                lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
                result = result + lora * config.scaling_value

        return result

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        assert self.lora_config is not None
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        label = self.lora_config.label

        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b


class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if config := self.lora_config:
            num_experts = getattr(config, 'num_experts', 1)
            if num_experts > 1:
                self.w_gating_lora = (
                    self.param("gating_einsum_lora_a_experts", config.init_fn, (num_experts, 2, self.features, config.rank)),
                    self.param("gating_einsum_lora_b_experts", config.init_fn, (num_experts, 2, config.rank, self.hidden_dim)),
                )
                self.w_linear_lora = (
                    self.param("linear_lora_a_experts", config.init_fn, (num_experts, self.hidden_dim, config.rank)),
                    self.param("linear_lora_b_experts", config.init_fn, (num_experts, config.rank, self.features)),
                )
                # 初始化 Router，基于模块初始输入的特征维度 self.features
                self.lora_router = self.param("lora_router", nn.initializers.normal(stddev=0.02), (self.features, num_experts))
            else:
                self.w_gating_lora = (
                    self.param("gating_einsum_lora_a", config.init_fn, (2, self.features, config.rank)),
                    self.param("gating_einsum_lora_b", config.init_fn, (2, config.rank, self.hidden_dim)),
                )
                self.w_linear_lora = (
                    self.param("linear_lora_a", config.init_fn, (self.hidden_dim, config.rank)),
                    self.param("linear_lora_b", config.init_fn, (config.rank, self.features)),
                )

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        gate_weights = None
        num_experts = getattr(self.lora_config, 'num_experts', 1) if self.lora_config else 1
        
        if num_experts > 1:
            # 在模块入口处，根据初始 x 计算唯一的路由权重
            router_logits = jnp.dot(x, self.lora_router.astype(dtype))
            gate_weights = jax.nn.softmax(router_logits, axis=-1)

        # 传递切片后的专家权重给 _dot
        gating_a = None if self.w_gating_lora is None else (self.w_gating_lora[0][:, 0] if num_experts > 1 else self.w_gating_lora[0][0])
        gating_b = None if self.w_gating_lora is None else (self.w_gating_lora[1][:, 0] if num_experts > 1 else self.w_gating_lora[1][0])
        ff_gate = self._dot(x, self.w_gating[0], None if self.w_gating_lora is None else (gating_a, gating_b), gate_weights)
        gate_value = nn.gelu(ff_gate)

        ff1_a = None if self.w_gating_lora is None else (self.w_gating_lora[0][:, 1] if num_experts > 1 else self.w_gating_lora[0][1])
        ff1_b = None if self.w_gating_lora is None else (self.w_gating_lora[1][:, 1] if num_experts > 1 else self.w_gating_lora[1][1])
        ff1 = self._dot(x, self.w_gating[1], None if self.w_gating_lora is None else (ff1_a, ff1_b), gate_weights)
        activations = gate_value * ff1

        outputs = self._dot(activations, self.w_linear, self.w_linear_lora, gate_weights)
        assert outputs.dtype == dtype
        return outputs

    def _dot(self, x: at.Array, w: at.Array, lora_weights: tuple[at.Array, at.Array] | None, gate_weights: at.Array | None = None) -> at.Array:
        base = jnp.dot(x, w.astype(x.dtype))
        if lora_weights is None:
            return base
            
        num_experts = getattr(self.lora_config, 'num_experts', 1) if self.lora_config else 1
        if num_experts > 1 and gate_weights is not None:
            w_a_experts, w_b_experts = lora_weights
            
            def compute_expert(wa, wb):
                return jnp.dot(jnp.dot(x, wa.astype(x.dtype)), wb.astype(x.dtype))
                
            lora_out = jax.vmap(compute_expert)(w_a_experts, w_b_experts) 
            
            lora_out_transposed = jnp.moveaxis(lora_out, 0, -2)
            gate_expanded = jnp.expand_dims(gate_weights, axis=-1)
            lora_blended = jnp.sum(lora_out_transposed * gate_expanded, axis=-2)
            
            return base + lora_blended
        else:
            return base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))
