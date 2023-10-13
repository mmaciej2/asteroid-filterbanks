import torch
import numpy as np
from .enc_dec import Filterbank


class PhaConSTFTFB(Filterbank):
    """STFT filterbank with consistent phase in Fourier basis relative to
        the input signal rather than the window function.

    Args:
        n_filters (int): Number of filters. Determines the length of the STFT
            filters before windowing.
        kernel_size (int): Length of the filters (i.e the window).
        stride (int, optional): Stride of the convolution (hop size). If None
            (default), set to ``kernel_size // 2``.
        window (:class:`numpy.ndarray`, optional): If None, defaults to
            ``np.sqrt(np.hanning())``.
        sample_rate (float): Sample rate of the expected audio.
            Defaults to 8000.

    Attributes:
        n_feats_out (int): Number of output filters.
    """

    def __init__(
        self, n_filters, kernel_size, stride=None, window=None, sample_rate=8000.0, **kwargs
    ):
        super().__init__(n_filters, kernel_size, stride=stride, sample_rate=sample_rate)
        assert n_filters >= kernel_size
        if n_filters % 2 != 0:
            raise ValueError(f"n_filters must be even, got {n_filters}")
        self.cutoff = int(n_filters / 2 + 1)
        self.n_feats_out = 2 * self.cutoff

        if window is None:
            window = np.hanning(kernel_size + 1)[:-1] ** 0.5
        else:
            if isinstance(window, torch.Tensor):
                window = window.data.numpy()
            ws = window.size
            if not (ws == kernel_size):
                raise AssertionError(
                    f"Expected window of size {kernel_size}. Received {ws} instead."
                )
        # Create and normalize DFT filters (can be overcomplete)
        filters = np. fft.fft(np.eye(n_filters))
        filters /= 0.5 * np.sqrt(kernel_size * n_filters / self.stride)

        filters = np.vstack(
            [np.real(filters[: self.cutoff, :]), np.imag(filters[: self.cutoff, :])]
        )

        filters[0, :] /= np.sqrt(2)
        filters[n_filters // 2, :] /= np.sqrt(2)

        # These variables represent the "true" filters and window, even though
        #   the implementation uses them in a non-standard way
        self.register_buffer("_filters", torch.from_numpy(filters).float())
        self.register_buffer("_window", torch.from_numpy(window).float())

    def filters(self):
        # We assume the signal has been pre-filtered according to the Fourier
        #   basis so the "filters" are just the convolutional windowing
        return self._window.unsqueeze(0).unsqueeze(0)

    def pre_analysis(self, wav: torch.Tensor):
        """This filters the full signal according to the Fourier basis."""
        if wav.ndim < 3:
            raise AssertionError(
                "We assume the data is at least 3D for consistency"
                "(mono-channel should have a channel dimension of 1)"
            )
        if wav.shape[-2] != 1:
            wav = wav.unsqueeze(-2)
        tiled_filters = self._filters.tile([1]*(wav.ndim-1)+[(wav.shape[-1]-1)//1+1])[..., :wav.shape[-1]]
        return wav * tiled_filters

    def post_analysis(self, spec: torch.Tensor):
        # remove the "frequency" dimension created by the convolutional windowing operation
        return spec.squeeze(-2)

    def pre_synthesis(self, spec: torch.Tensor):
        # reintroduce the convolutional window "frequency" dimension
        return spec.unsqueeze(-2)

    def post_synthesis(self, wav: torch.Tensor):
        tiled_filters = self._filters.tile([1]*(wav.ndim-1)+[(wav.shape[-1]-1)//1+1])[..., :wav.shape[-1]]
        wav = wav * tiled_filters
        return wav.sum(-2)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Compat loader to avoid breaking when _window is missing."""
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        # Window won't change, it's just a convenience.
        to_remove = [key for key in missing_keys if key.endswith("_window")]
        for key in to_remove:
            missing_keys.remove(key)


def perfect_synthesis_window(analysis_window, hop_size):
    """Computes a window for perfect synthesis given an analysis window and
        a hop size.

    Args:
        analysis_window (np.array): Analysis window of the transform.
        hop_size (int): Hop size in number of samples.

    Returns:
        np.array : the synthesis window to use for perfectly inverting the STFT.
    """
    win_size = len(analysis_window)
    den = np.zeros_like(analysis_window)

    loop_on = (win_size - 1) // hop_size
    for win_idx in range(-loop_on, loop_on + 1):
        shifted = np.roll(analysis_window ** 2, win_idx * hop_size)
        if win_idx < 0:
            shifted[win_idx * hop_size :] = 0
        elif win_idx > 0:
            shifted[: win_idx * hop_size] = 0
        den += shifted
    den = np.where(den != 0.0, den, np.finfo(den.dtype).tiny)
    correction = int(0.5 * len(analysis_window) / hop_size)
    return correction * analysis_window / den
