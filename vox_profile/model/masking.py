# Code taken from SpeechBrain
#   https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/dataio/dataio.py
#   https://github.com/speechbrain/speechbrain/blob/release-v1.0.0/speechbrain/lobes/models/huggingface_transformers/huggingface.py
#
#   Credit to the Authors.

import torch


def length_to_mask(length, max_len=None, dtype=None, device=None):
    """Creates a binary mask for each sequence.

    Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.

    Returns
    -------
    mask : tensor
        The binary mask.

    Example
    -------
    >>> length = torch.Tensor([1, 2, 3])
    >>> mask = length_to_mask(length)
    >>> mask
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    """
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask

    mask = torch.arange(
            max_len, device=length.device, dtype=length.dtype
        ).expand(len(length), max_len) < length.unsqueeze(1)
        
    if dtype is None:
            dtype = length.dtype
            
    if device is None:
        device = length.device
                
    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask

def make_padding_masks(src, wav_len=None, pad_idx=0):
    """This method generates the padding masks.

    Arguments
    ---------
    src : tensor
        The sequence to the encoder (required).
    wav_len : tensor
        The relative length of the wav given in SpeechBrain format.
    pad_idx : int
        The index for <pad> token (default=0).

    Returns
    -------
    src_key_padding_mask : tensor
        The padding mask.
    """
    src_key_padding_mask = None
    if wav_len is not None:
        abs_len = torch.round(wav_len * src.shape[1])
        src_key_padding_mask = length_to_mask(abs_len).bool()
            
    return src_key_padding_mask
