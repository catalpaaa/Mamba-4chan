from dataclasses import dataclass, field


@dataclass
class mamba_4chan_130m_config:
    d_model: int = 768
    n_layer: int = 24
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True


@dataclass
class mamba_4chan_370m_config:
    d_model: int = 1024
    n_layer: int = 48
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
