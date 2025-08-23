import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import init


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


"""TransformerEncoder Modules with Communication Channel."""


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, masked=False, use_comms_channel=False, num_messages=256):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        self.n_embd = n_embd

        # Communication channel parameters (backward compatible)
        self.use_comms_channel = use_comms_channel
        self.num_messages = num_messages

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

    def get_comms_noise(self, target):
        """Generate communication channel noise."""
        # calculate noise as per comms protocol
        noise = (torch.rand_like(target, device=target.device) - 0.5) * 2
        delta = (1 / self.num_messages)
        noise = noise * delta * 0.5

        return noise

    def compute_component_log_loss(self, z):
        """
        Computes the communication penalty loss given by log2(2 * |M| * |z| + 1)
        """
        M = self.num_messages

        loss = torch.log2(2 * M * z.abs() + 1)
        return torch.sum(loss)

    def compute_num_bits_used(self, target):
        """Track the number of bits used in communication."""
        # All agents active, so count all elements
        bits_used = torch.ones_like(target) * 32  # float32
        return torch.sum(bits_used)

    def forward(self, key, value, query):
        B, L, D = query.size()

        # Reset communication metrics for this forward pass
        self.comm_loss = 0
        self.comm_bits = 0

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # Apply communication channel noise if enabled
        if self.use_comms_channel:
            # Add noise to keys
            k_noise = self.get_comms_noise(k)
            k = k + k_noise

            # Track communication metrics for keys
            self.comm_loss += self.compute_component_log_loss(k)
            self.comm_bits += self.compute_num_bits_used(k)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.masked:
            raise NotImplementedError("Masked attention is not supported in this implementation.")

        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # Apply communication channel noise to output if enabled
        if self.use_comms_channel:
            y_noise = self.get_comms_noise(y)
            y = y + y_noise

            # Track communication metrics for output
            self.comm_loss += self.compute_component_log_loss(y)
            self.comm_bits += self.compute_num_bits_used(y)

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, use_comms_channel=False, num_messages=256):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, masked=False,
                                  use_comms_channel=use_comms_channel,
                                  num_messages=num_messages)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x

    def get_comm_metrics(self):
        """Return communication loss and bits from attention layer."""
        return self.attn.comm_loss, self.attn.comm_bits


class TransformerEncoderLayer(nn.Module):

    def __init__(self, obs_shape, n_block, n_embd, n_head,
                 use_comms_channel=False, num_messages=15):
        super(TransformerEncoderLayer, self).__init__()

        self.obs_dim = obs_shape
        self.n_embd = n_embd
        self.n_block = n_block

        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_shape),
                                         init_(nn.Linear(obs_shape, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.ModuleList([EncodeBlock(n_embd, n_head,
                                                 use_comms_channel=use_comms_channel,
                                                 num_messages=num_messages)
                                     for _ in range(n_block)])

    def forward(self, obs):
        # obs: (batch, n_agent, obs_dim)
        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings
        x = self.ln(x)

        # Track total communication metrics across all blocks
        total_comm_loss = 0
        total_comm_bits = 0

        for block in self.blocks:
            x = block(x)
            comm_loss, comm_bits = block.get_comm_metrics()
            total_comm_loss += comm_loss
            total_comm_bits += comm_bits

        return x, (total_comm_loss, total_comm_bits)


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

        # Store flag for communication metrics calculation
        self.calc_comm_metrics = calc_comm_metrics and use_comms_channel

        self.transformer_encoder = TransformerEncoderLayer(
            obs_dim,
            n_block, n_embd, n_head,
            use_comms_channel=use_comms_channel,
            num_messages=num_messages
        )

    def forward(self, x):
        """
        Forward pass through transformer encoder.

        Args:
            x: Input observations (batch, n_agent, obs_dim)

        Returns:
            If calc_comm_metrics is False:
                - x: Encoded representations
            If calc_comm_metrics is True:
                - x: Encoded representations
                - (comm_loss, comm_bits): Communication metrics tuple
        """
        x, comm_metrics = self.transformer_encoder(x)

        # Return based on whether communication metrics calculation is enabled
        if self.calc_comm_metrics:
            # Return with communication metrics when calculation is enabled
            return x, comm_metrics
        else:
            # return just x when metrics calculation is disabled
            return x
