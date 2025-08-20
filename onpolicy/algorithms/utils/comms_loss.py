import torch

# constants
NOISE_WIDTH = 1.0 / 15
DEBUG = True


def component_log_loss(x, delta, mask=None):
    """
    Log loss to penalize the number of bits communicated during training
    """
    # print("input shape for loss", x.shape)
    # Loss: log2(|M|z + 1)

    loss = torch.log2(2 * x.abs() / delta + 1).sum(-1)

    # print("loss shape", loss.shape)
    # msg_shape = len(x.shape)
    # if msg_shape == 4:  # message type: key
    #     out = loss[0, 0]  # 1D vector
    # else:  # message type: weighted value
    #     out = loss[0, 0]  # 2D matrix
    # if reduction == "none":
    #     return loss
    return loss.sum(), loss.sum(1).squeeze()


def compute_num_bits_used(target, mask):
    bits_used = mask.expand_as(target) * 32
    return torch.sum(bits_used)
