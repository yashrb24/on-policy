import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import init
from onpolicy.envs.toyproblem.channels import build_channel


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


"""TransformerEncoder Modules with Communication Channel."""


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, masked=False, use_comms_channel=False, num_messages=15,
                 use_fake_quantization=False, quant_bits=8,
                 channel_name="tpdf", delta=1.0,
                 delta_learnable=False, delta_global_learnable=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        self.n_embd = n_embd

        self.use_comms_channel = use_comms_channel
        self.num_messages = num_messages

        # Quantization parameters
        self.use_fake_quantization = use_fake_quantization
        self.quant_bits = quant_bits
        self.quant_min = 0
        self.quant_max = 2 ** quant_bits - 1

        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))

        self.att_bp = None

        # Initialize communication tracking
        self.comm_loss = 0
        self.comm_bits = 0

        # Channel instances: key_channel operates on (B, nh, L, hs), out_channel on (B, L, n_embd)
        hs = n_embd // n_head
        if use_comms_channel:
            self.key_channel = build_channel(
                channel_name, delta=delta,
                delta_learnable=delta_learnable,
                delta_global_learnable=delta_global_learnable,
                zdim=hs,
            )
            self.out_channel = build_channel(
                channel_name, delta=delta,
                delta_learnable=delta_learnable,
                delta_global_learnable=delta_global_learnable,
                zdim=n_embd,
            )
        else:
            self.key_channel = None
            self.out_channel = None

    def _masked_comms_loss(self, channel, z, active_masks):
        """Apply channel.comms_loss(z) and reduce with active_masks."""
        raw = channel.comms_loss(z)
        batch_size = z.size(0)
        if active_masks is not None:
            if z.dim() == 4:
                mask_expanded = active_masks.unsqueeze(1).expand(
                    batch_size, z.size(1), z.size(2), 1).expand_as(z)
            elif z.dim() == 3:
                mask_expanded = active_masks.expand_as(z)
            else:
                raise ValueError(f"Expected 3D or 4D tensor, got {z.dim()}D")
            raw = raw * mask_expanded
            active_elements = mask_expanded.sum()
            if active_elements > 0:
                return raw.sum() / active_elements
            else:
                return torch.tensor(0.0, device=z.device, dtype=z.dtype)
        else:
            return torch.mean(torch.sum(raw.reshape(batch_size, -1), dim=1))

    def get_delta_metrics(self) -> dict:
        """Return current delta values for key and out channels as a flat dict.

        Only populated when use_comms_channel=True and at least one delta mode is
        learnable. For fixed delta returns an empty dict (nothing to track).
        Keys:
          key_delta       — scalar (global learnable) or 1-D tensor (per-dim learnable)
          out_delta       — same, for the out_channel
        Values are detached from the graph.
        """
        if not self.use_comms_channel:
            return {}
        key_ch = self.key_channel
        out_ch = self.out_channel
        is_learnable = (
            getattr(key_ch, '_delta_learnable', False) or
            getattr(key_ch, '_delta_global_learnable', False)
        )
        if not is_learnable:
            return {}
        key_d = key_ch.delta
        out_d = out_ch.delta
        if isinstance(key_d, torch.Tensor):
            key_d = key_d.detach()
        if isinstance(out_d, torch.Tensor):
            out_d = out_d.detach()
        return {"key_delta": key_d, "out_delta": out_d}

    def compute_component_log_loss(self, z, active_masks=None):
        """
        Computes the communication penalty loss given by log2(2 * |M| * |z| + 1)
        """
        M = self.num_messages

        batch_size = z.size(0)
        loss = torch.log2(2 * M * z.abs() + 1)

        if active_masks is not None:
            # Mask out inactive agents
            # Handle both 3D and 4D tensor shapes
            if z.dim() == 4:
                # z shape: [B, nh, L, hs], active_masks shape: [B, L, 1]
                mask_expanded = active_masks.unsqueeze(1).expand(batch_size, z.size(1), z.size(2), 1)
                mask_expanded = mask_expanded.expand_as(z)
            elif z.dim() == 3:
                # z shape: [B, L, D], active_masks shape: [B, L, 1]
                mask_expanded = active_masks.expand_as(z)
            else:
                raise ValueError(f"Expected 3D or 4D tensor, got {z.dim()}D")

            loss = loss * mask_expanded
            # Average only over active agents
            active_elements = torch.sum(mask_expanded)
            if active_elements > 0:
                return torch.sum(loss) / active_elements
            else:
                return torch.tensor(0.0, device=z.device, dtype=z.dtype)
        else:
            return torch.mean(torch.sum(loss.reshape(batch_size, -1), dim=1))

    def compute_num_bits_used(self, target, active_masks=None):
        """Track the number of bits used in communication."""
        bits_used = torch.ones_like(target) * 32  # float32
        if active_masks is not None:
            # Expand masks to match target dimensions
            # Handle both 3D and 4D tensor shapes
            if target.dim() == 4:
                # target shape: [B, nh, L, hs], active_masks shape: [B, L, 1]
                mask_expanded = active_masks.unsqueeze(1).expand(target.size(0), target.size(1), target.size(2), 1)
                mask_expanded = mask_expanded.expand_as(target)
            elif target.dim() == 3:
                # target shape: [B, L, D], active_masks shape: [B, L, 1]
                mask_expanded = active_masks.expand_as(target)
            else:
                raise ValueError(f"Expected 3D or 4D tensor, got {target.dim()}D")

            bits_used = bits_used * mask_expanded
        return torch.sum(bits_used)

    def apply_fake_quantization(self, tensor):
        """
        Manual implementation of fake quantization with proper gradient flow
        """
        # Create a copy of the input tensor to avoid modifying the original
        original = tensor.clone()

        min_val = torch.tensor(0.0, device=tensor.device)
        max_val = torch.tensor(1.0, device=tensor.device)

        if min_val == max_val:
            min_val = min_val - 0.01
            max_val = max_val + 0.01

        # Calculate scale and zero_point from detached values
        scale = (max_val - min_val) / (self.quant_max - self.quant_min)
        zero_point = torch.round(self.quant_min - min_val / scale)

        # Apply quantization formula without modifying the original
        scaled = tensor / scale + zero_point
        rounded = torch.round(scaled)
        clamped = torch.clamp(rounded, self.quant_min, self.quant_max)
        shifted = clamped - zero_point
        dequantized = shifted * scale

        return dequantized

    def compute_quantization_loss(self, tensor, active_masks=None):
        """
        Compute loss based on the quantized tensor values
        """
        # Compute the loss directly on all tensor values
        loss = torch.log2(2 * self.quant_max * torch.abs(tensor) + 1)

        if active_masks is not None:
            # Mask out inactive agents
            # Handle both 3D and 4D tensor shapes
            if tensor.dim() == 4:
                # tensor shape: [B, nh, L, hs], active_masks shape: [B, L, 1]
                mask_expanded = active_masks.unsqueeze(1).expand(tensor.size(0), tensor.size(1), tensor.size(2), 1)
                mask_expanded = mask_expanded.expand_as(tensor)
            elif tensor.dim() == 3:
                # tensor shape: [B, L, D], active_masks shape: [B, L, 1]
                mask_expanded = active_masks.expand_as(tensor)
            else:
                raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")

            loss = loss * mask_expanded
            return torch.sum(loss)
        else:
            return torch.sum(loss)

    def compute_quantization_bits(self, tensor, active_masks=None):
        """
        Compute the actual bits used based on the tensor values
        """
        if active_masks is not None:
            # Only count bits for active agents
            # Handle both 3D and 4D tensor shapes
            if tensor.dim() == 4:
                # tensor shape: [B, nh, L, hs], active_masks shape: [B, L, 1]
                mask_expanded = active_masks.unsqueeze(1).expand(tensor.size(0), tensor.size(1), tensor.size(2), 1)
                mask_expanded = mask_expanded.expand_as(tensor)
            elif tensor.dim() == 3:
                # tensor shape: [B, L, D], active_masks shape: [B, L, 1]
                mask_expanded = active_masks.expand_as(tensor)
            else:
                raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")

            active_values = torch.sum(mask_expanded)
            total_bits = self.quant_bits * active_values
        else:
            # Count total number of values in the tensor
            num_values = tensor.numel()
            # Each value uses quant_bits bits
            total_bits = self.quant_bits * num_values

        return total_bits

    def forward(self, key, value, query, active_masks=None):
        B, L, D = query.size()

        # Reset communication metrics for this forward pass
        self.comm_loss = 0
        self.comm_bits = 0

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # Apply communication channel to keys if enabled.
        if self.use_comms_channel:
            self.comm_loss += self._masked_comms_loss(self.key_channel, k, active_masks)
            k, _ = self.key_channel(k)
            self.comm_bits += self.compute_num_bits_used(k, active_masks)

        elif self.use_fake_quantization:
            self.comm_loss += self.compute_quantization_loss(k, active_masks)
            k = self.apply_fake_quantization(k)
            self.comm_bits += self.compute_quantization_bits(k, active_masks)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.masked:
            raise NotImplementedError("Masked attention is not supported in this implementation.")

        # Apply active masks if provided
        if active_masks is not None:
            # active_masks shape: [B, L, 1]
            # Create attention mask: [B, 1, L, 1] -> [B, nh, L, L]
            mask = active_masks.squeeze(-1).unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L]
            mask = mask.expand(-1, self.n_head, L, -1)  # [B, nh, L, L]
            mask_t = mask.transpose(-2, -1)  # [B, nh, L, L]
            combined_mask = mask * mask_t  # Both query and key must be active
            # Set attention to -inf where either query or key agent is inactive
            att = att.masked_fill(combined_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)

        # Handle NaN from softmax of all -inf (when all agents are inactive)
        if active_masks is not None:
            att = torch.nan_to_num(att, nan=0.0)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # Apply communication channel to output if enabled.
        if self.use_comms_channel:
            self.comm_loss += self._masked_comms_loss(self.out_channel, y, active_masks)
            y, _ = self.out_channel(y)
            self.comm_bits += self.compute_num_bits_used(y, active_masks)

        elif self.use_fake_quantization:
            self.comm_loss += self.compute_quantization_loss(y, active_masks)
            y = self.apply_fake_quantization(y)
            self.comm_bits += self.compute_quantization_bits(y, active_masks)

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, use_comms_channel=False, num_messages=15, use_fake_quantization=False,
                 quant_bits=8, channel_name="tpdf", delta=1.0,
                 delta_learnable=False, delta_global_learnable=False):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, masked=False,
                                  use_comms_channel=use_comms_channel,
                                  num_messages=num_messages,
                                  use_fake_quantization=use_fake_quantization,
                                  quant_bits=quant_bits,
                                  channel_name=channel_name,
                                  delta=delta,
                                  delta_learnable=delta_learnable,
                                  delta_global_learnable=delta_global_learnable)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, active_masks=None):
        x = self.ln1(x + self.attn(x, x, x, active_masks))
        x = self.ln2(x + self.mlp(x))
        return x

    def get_comm_metrics(self):
        """Return communication loss, bits, and delta metrics from attention layer."""
        return self.attn.comm_loss, self.attn.comm_bits, self.attn.get_delta_metrics()


class TransformerEncoderLayer(nn.Module):

    def __init__(self, obs_shape, n_block, n_embd, n_head, use_comms_channel=False, num_messages=15,
                 use_fake_quantization=False, quant_bits=8, channel_name="tpdf", delta=1.0,
                 delta_learnable=False, delta_global_learnable=False):
        super(TransformerEncoderLayer, self).__init__()

        self.obs_dim = obs_shape
        self.n_embd = n_embd
        self.n_block = n_block

        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_shape),
                                         init_(nn.Linear(obs_shape, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.ModuleList([EncodeBlock(n_embd, n_head,
                                                 use_comms_channel=use_comms_channel,
                                                 num_messages=num_messages,
                                                 use_fake_quantization=use_fake_quantization,
                                                 quant_bits=quant_bits,
                                                 channel_name=channel_name,
                                                 delta=delta,
                                                 delta_learnable=delta_learnable,
                                                 delta_global_learnable=delta_global_learnable)
                                     for _ in range(n_block)])

    def forward(self, obs, active_masks=None):
        # obs: (batch, n_agent, obs_dim)
        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings
        x = self.ln(x)

        # Track total communication metrics across all blocks
        total_comm_loss = 0
        total_comm_bits = 0
        # delta_metrics: flat dict keyed "block{i}_key_delta" / "block{i}_out_delta"
        delta_metrics: dict = {}

        for i, block in enumerate(self.blocks):
            x = block(x, active_masks)
            comm_loss, comm_bits, blk_delta = block.get_comm_metrics()
            total_comm_loss += comm_loss
            total_comm_bits += comm_bits
            for k, v in blk_delta.items():
                delta_metrics[f"block{i}_{k}"] = v

        return x, (total_comm_loss, total_comm_bits, delta_metrics)


class TransformerEncoderBase(nn.Module):
    """A TransformerEncoder base module for actor and critic."""

    def __init__(self, args, obs_shape, calc_comm_metrics=True):
        super(TransformerEncoderBase, self).__init__()

        n_block = args.n_block
        n_embd = args.n_embd
        n_head = args.n_head

        obs_dim = obs_shape[0]

        # Check if communication channel is enabled
        use_comms_channel = args.use_comms_channel
        num_messages = args.num_messages

        # Check if fake quantization is enabled
        use_fake_quantization = args.use_fake_quantization
        quant_bits = args.quant_bits

        # Resolve channel name: prefer --channel, fall back to --ddcl_variation mapping
        channel_name = getattr(args, 'channel', None)
        if channel_name is None:
            _map = {"old": "sd", "new": "tpdf"}
            ddcl_variation = getattr(args, 'ddcl_variation', 'new')
            channel_name = _map.get(ddcl_variation, "tpdf")
        if not use_comms_channel:
            channel_name = "none"

        delta = getattr(args, 'delta', None)
        if delta is None:
            delta = 1.0 / num_messages  # backwards compat: old delta = 1/num_messages

        delta_learnable        = getattr(args, 'delta_learnable', False)
        delta_global_learnable = getattr(args, 'delta_global_learnable', False)

        # Store flag for communication metrics calculation
        self.calc_comm_metrics = calc_comm_metrics and use_comms_channel

        self.transformer_encoder = TransformerEncoderLayer(
            obs_dim,
            n_block, n_embd, n_head,
            use_comms_channel=use_comms_channel,
            num_messages=num_messages,
            use_fake_quantization=use_fake_quantization,
            quant_bits=quant_bits,
            channel_name=channel_name,
            delta=delta,
            delta_learnable=delta_learnable,
            delta_global_learnable=delta_global_learnable,
        )

    def forward(self, x, active_masks=None):
        """
        Forward pass through transformer encoder.

        Args:
            x: Input observations (batch, n_agent, obs_dim)
            active_masks: Optional agent activity mask (batch, n_agent, 1)

        Returns:
            If calc_comm_metrics is False:
                - x: Encoded representations
            If calc_comm_metrics is True:
                - x: Encoded representations
                - (comm_loss, comm_bits): Communication metrics tuple
        """
        x, (comm_loss, comm_bits, delta_metrics) = self.transformer_encoder(x, active_masks)

        if self.calc_comm_metrics:
            return x, (comm_loss, comm_bits, delta_metrics)
        else:
            return x
