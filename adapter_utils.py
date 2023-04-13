"""
1, READ: REcurrent ADaptation, preprint

    A parameter efficient adaptation technique for multi-layer backbone archtecture using recurrent networks.

2, Adapter: https://arxiv.org/abs/1902.00751
"""
from dataclasses import dataclass
from functools import partial
from typing import Iterable, List, Optional, Tuple, Union
from transformers.adapters import LoRAConfig
import hydra

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch import nn, Tensor


@dataclass
class FTConf:
    ft_type: str = MISSING


@dataclass
class PartialFTConf(FTConf):
    ft_type: str = "partial"
    ft_key: str = MISSING


def get_class_name_str(klass) -> str:
    """
    Args:
        klass: The class definition.

    Returns:
        The fully qualified name of the given class.
    """
    return ".".join([klass.__module__, klass.__name__])


def create_config(ft_type,
                  hidden_dim,
                  scaling_factor=1.0,
                  bidirectional=False,
                  r=8,
                  alpha=8):
    if ft_type == "full":
        peft_confg = FTConf(ft_type="full")
    elif ft_type == "partial":
        peft_confg = FTConf(ft_type="partial")
    elif ft_type == "bias":
        peft_confg = PartialFTConf(ft_key="bias")
    elif ft_type == "adapters":
        peft_confg = AdapterConf(
            input_dim=768,
            hidden_dim=hidden_dim,
            output_dim=768,
        )
    elif ft_type == "rnn":
        peft_confg = RNNAdapterConf(
            outputs_scaling_factor=scaling_factor,
            input_dim=768,
            rnn_dim=hidden_dim,
            num_layers=1,
            rnn_type="gru",
            bidirectional=bidirectional,
        )
    elif ft_type == "lora":
        peft_confg = LoRAConf(r=r, alpha=alpha)
    else:
        raise Exception(f"Unknown ft_type {ft_type}")
    return peft_confg


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def enable_full_tuning(model, num_orig_params):
    for p in model.decoder.parameters():
        p.requires_grad = True
    print(f"Enables full tuning; num trainable params {num_orig_params:,}")


def enable_partial_tuning(model, num_orig_params, ft_key):
    print(ft_key)
    p_num = 0
    for n, p in model.named_parameters():
        if ft_key in n:
            p.requires_grad = True
            p_num += p.numel()
        else:
            p.requires_grad = False
    ratio = round(p_num / num_orig_params, 3)
    print(
        f"Enables partial tuning on {ft_key}; num trainable params {p_num:,}, ratio {ratio}"
    )


def enable_adapter_tuning(model, num_orig_params, peft_conf):
    freeze(model)
    encoder_adapters = adapter_init(
        model=model.encoder,
        layers=model.encoder.encoder.layer,
        adapter_conf=peft_conf,
        std=None,
        register_hooks=True,
    )
    decoder_adapters = adapter_init(
        model=model.decoder,
        layers=model.decoder.bert.encoder.layer,
        adapter_conf=peft_conf,
        std=None,
        register_hooks=True,
    )
    encoder_p_num = sum(p.numel() for adapter in encoder_adapters
                        for p in adapter.parameters())
    decoder_p_num = sum(p.numel() for adapter in decoder_adapters
                        for p in adapter.parameters())
    p_num = encoder_p_num + decoder_p_num
    ratio = round(p_num / num_orig_params, 3)
    print(
        f"Enables adapter tuning; num trainable params {p_num:,}, ratio {ratio}"
    )


def enable_rnn_adapter_tuning(model, num_orig_params, peft_conf):
    freeze(model)

    encoder_rnn_adapter = rnn_adapter_init(
        model=model.encoder,
        layers=model.encoder.encoder.layer,
        merge_module=model.encoder.layernorm,
        rnn_adapter_conf=peft_conf,
    )
    decoder_rnn_adapter = rnn_adapter_init(
        model=model.decoder,
        layers=model.decoder.bert.encoder.layer,
        merge_module=model.decoder.bert.encoder.layer[11].output.LayerNorm,
        rnn_adapter_conf=peft_conf,
    )
    encoder_p_num = sum(p.numel() for p in encoder_rnn_adapter.parameters())
    decoder_p_num = sum(p.numel() for p in decoder_rnn_adapter.parameters())
    p_num = encoder_p_num + decoder_p_num
    ratio = round(p_num / num_orig_params, 3)
    print(
        f"Enables rnn-adapter tuning; num trainable params {p_num:,}, ratio {ratio}"
    )


def enable_lora_tuning(model, num_orig_params, peft_conf):
    freeze(model)
    config = LoRAConfig(r=peft_conf.r, alpha=peft_conf.alpha)
    model.decoder.add_adapter("lora", config=config, set_active=True)
    p_num = count_parameters(model)
    ratio = round(p_num / num_orig_params, 3)
    print(
        f"Enables lora tuning; num trainable params {p_num:,}, ratio:{ratio}")


@dataclass
class LoRAConf(FTConf):
    _target_: str = get_class_name_str(LoRAConfig)
    ft_type: str = "lora"
    r: int = 8
    alpha: int = 8


def setup_ft(model, peft_conf):
    num_orig_params = count_parameters(model)
    if peft_conf.ft_type == "full":
        enable_full_tuning(model, num_orig_params)
    elif peft_conf.ft_type == "partial":
        enable_partial_tuning(model, num_orig_params, peft_conf.ft_key)
    elif peft_conf.ft_type == "adapter":
        enable_adapter_tuning(model, num_orig_params, peft_conf)
    elif peft_conf.ft_type == "rnn_adapter":
        enable_rnn_adapter_tuning(model, num_orig_params, peft_conf)
    elif peft_conf.ft_type == "bias":
        enable_partial_tuning(model, num_orig_params, peft_conf.ft_key)
    elif peft_conf.ft_type == "lora":
        enable_lora_tuning(model, num_orig_params, peft_conf)
    else:
        raise ValueError(f"Unsupported FT type {peft_conf.ft_type}")


class T5LayerNorm(nn.Module):
    """
    Copy-pasta from fbcode/pytorch/text/torchtext/models/t5/modules.py
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:

        variance = hidden_states.to(torch.float32).pow(2).mean(-1,
                                                               keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)

        # Convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class RNNAdapter(nn.Module):

    def __init__(
        self,
        input_dim: int,
        rnn_dim: int,
        num_layers: int,
        adapted_layers: Optional[List[int]] = None,
        rnn_type: str = "rnn",
        embedding_init_hidden: bool = False,
        layer_norm_embedding: bool = True,
        layer_norm_before: bool = True,
        layer_norm_after: bool = False,
        h0_scaling_factor: float = 1.0,
        inputs_scaling_factor: float = 1.0,
        outputs_scaling_factor: float = 1.0,
        bidirectional: bool = False,
        agg_type: str = "mean",
        ft_type: str = "rnn_adapter",
    ) -> None:
        """
        Implementation of a recurrent adaptation design for multi-layer backbone.
        An RNN module takes the i^th layer outputs as its i^th timestep inputs X_i
        and its final hidden states will be used as adapted states.

        Args
            input_dim:
                hidden dimension of a backbone layer
            rnn_dim
                hidden dimension of rnn module
            num_layers:
                number of rnn layers
            adapted_layers:
                indices of backbone layers to adapt
            rnn_type:
                the variant name of RNN; must be one of 'rnn', 'lstm', or 'gru'.
            embedding_init_hidden:
                whether to initialize hidden states as embeddings or zeros
            layer_norm_embedding:
                whether to apply layer normalization to embedding (initial hidden states)
            layer_norm_before:
                whether to apply layer normalization to inputs
            layer_norm_after:
                whether to apply layer normalization to outputs
            h0_scaling_factor:
                the coeffifient multiplied to the RNN initial hidden states
            inputs_scaling_factor:
                the coeffifient multiplied to the RNN inputs
            outputs_scaling_factor:
                the coeffifient multiplied to the outputs
            agg_type:
                aggregation type for RNN states; must be one of 'mean' or 'last'.
        """

        super().__init__()

        self.rnn_dim = rnn_dim
        self.num_layers = num_layers
        self.adapted_layers = adapted_layers
        self.rnn_type = rnn_type
        self.agg_type = agg_type

        self.embedding_init_hidden = embedding_init_hidden

        self.layer_norm_embedding = layer_norm_embedding
        self.h0_layer_norm: Optional[nn.Module] = None
        if layer_norm_embedding and embedding_init_hidden:
            self.h0_layer_norm: Optional[nn.Module] = T5LayerNorm(input_dim)

        self.layer_norm_before = layer_norm_before
        self.inputs_layer_norm: Optional[nn.Module] = None
        if layer_norm_before:
            self.inputs_layer_norm: Optional[nn.Module] = T5LayerNorm(
                input_dim)

        self.layer_norm_after = layer_norm_after
        self.outputs_layer_norm: Optional[nn.Module] = None
        if layer_norm_after:
            self.outputs_layer_norm: Optional[nn.Module] = T5LayerNorm(
                input_dim)

        self.h0_scaling_factor = h0_scaling_factor
        self.inputs_scaling_factor = inputs_scaling_factor
        self.outputs_scaling_factor = outputs_scaling_factor
        self.bidirectional = bidirectional

        if rnn_type == "lstm":
            rnn_module = nn.LSTM
        elif rnn_type == "gru":
            rnn_module = nn.GRU
        elif rnn_type == "rnn":
            rnn_module = nn.RNN
        else:
            raise ValueError(f"unsupported rnn type {rnn_type}")

        self.input_projection: Optional[nn.Module] = None
        if embedding_init_hidden:
            self.input_projection: Optional[nn.Module] = nn.Linear(
                input_dim, rnn_dim)
        self.rnn: nn.Module = rnn_module(
            input_size=input_dim,
            hidden_size=rnn_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.output_projection = nn.Linear(rnn_dim * 2 if bidirectional else 1,
                                           input_dim)

        self._cached_states: List[Tensor] = []

    def empty_cache(self) -> None:

        self._cached_states = []

    def forward(self) -> Tensor:
        if self.embedding_init_hidden:
            inputs = self._cached_states[1:]
        else:
            inputs = self._cached_states

        if (torch.jit.isinstance(self.adapted_layers, List[int])
                # pyre-ignore
                and len(self.adapted_layers) > 0):
            # pyre-ignore
            inputs = [inputs[idx] for idx in self.adapted_layers]

        assert len(inputs) > 0, "cached states not found"
        inputs_dim = inputs[0].dim()
        if inputs_dim == 3:
            bsz, seq_len, _ = inputs[0].shape
            rnn_bsz = bsz * seq_len
        elif inputs_dim == 2:
            bsz, _ = inputs[0].shape
            rnn_bsz = bsz
        else:
            raise ValueError(f"Unsupported inputs dim {inputs_dim}")

        if self.embedding_init_hidden:
            h0 = self._cached_states[0]
            # convert h0 from (bsz, seq_len, input_dim) to (1, rnn_bsz, input_dim)
            h0 = h0.view(rnn_bsz, -1).unsqueeze(0)
            if self.layer_norm_embedding:
                assert isinstance(self.h0_layer_norm, nn.Module)
                h0 = self.h0_layer_norm(h0)
            h0 = h0 * self.h0_scaling_factor
            assert isinstance(self.input_projection, nn.Module)
            h0 = self.input_projection(h0)
        else:
            h0 = torch.zeros(
                (2 if self.bidirectional else 1, rnn_bsz, self.rnn_dim),
                device=self._cached_states[0].device)

        # prepare hx argument for RNN's forward.
        if self.rnn_type == "lstm":
            # Initialize LSTM cell state as zeros.
            hx = (h0, torch.zeros_like(h0,
                                       device=self._cached_states[0].device))
        else:
            hx = h0

        # prepare input argument for RNN's forward.
        # convert inputs to (rnn_bsz, num_time_steps, input_dim);
        # e.g. num_time_steps = #backbone_layers
        inputs = [states.view(rnn_bsz, -1) for states in inputs]
        inputs = torch.cat(inputs, 0).view(len(inputs), rnn_bsz,
                                           -1).transpose(0, 1)
        if self.layer_norm_before:
            assert isinstance(self.inputs_layer_norm, nn.Module)
            inputs = self.inputs_layer_norm(inputs)
        inputs = inputs * self.inputs_scaling_factor

        # rnn_outputs is a 3-tuple. The first one is the hidden states
        # of shape (rnn_bsz, num_time_steps, rnn_dim), the second is the last
        # hidden state of shape (1, rnn_bsz, rnn_dim).
        rnn_outputs = self.rnn(inputs, hx=hx)

        if self.agg_type == "last":
            rnn_states = rnn_outputs[0][:, -1, :]
        elif self.agg_type == "mean":
            rnn_states = rnn_outputs[0].mean(1)
        else:
            raise ValueError(f"Unsupported agg_type {self.agg_type}")

        if inputs_dim == 3:
            # pyre-ignore
            rnn_states = rnn_states.view(bsz, seq_len, -1)

        outputs = self.output_projection(rnn_states)
        if self.layer_norm_after:
            assert isinstance(self.outputs_layer_norm, nn.Module)
            outputs = self.outputs_layer_norm(outputs)
        outputs = outputs * self.outputs_scaling_factor

        return outputs


@dataclass
class RNNAdapterConf(FTConf):
    _target_: str = get_class_name_str(RNNAdapter)
    ft_type: str = "rnn_adapter"
    input_dim: int = MISSING
    rnn_dim: int = MISSING
    num_layers: int = MISSING
    adapted_layers: Optional[List[int]] = None
    rnn_type: str = "rnn"
    agg_type: str = "mean"
    embedding_init_hidden: bool = False
    layer_norm_embedding: bool = True
    layer_norm_before: bool = True
    layer_norm_after: bool = False
    h0_scaling_factor: float = 1.0
    inputs_scaling_factor: float = 1.0
    outputs_scaling_factor: float = 1.0
    bidirectional: bool = False


def cache_states(
    module: nn.Module,
    inputs: Tuple[Tensor],
    outputs: Union[Tensor, Tuple[Tensor]],
    rnn_adapter: nn.Module,
    empty_cache: bool,
) -> Union[Tensor, Tuple[Tensor]]:
    """
    Cache the hidden states at a backbone layer. This function
    will be registered as a forward hook at a specified layer
    during training.

    Args
        module:
            a backbone layer
        inputs:
            original inputs of backbone layer
        outputs:
            original outputs of backbone layer
        rnn_adapter:
            the RNNAdapter module attached to backbone
        empty_cache:
            if True, clear cached states.

    Returns:
        original outputs of backbone layer
    """
    if empty_cache:
        # pyre-ignore
        rnn_adapter.empty_cache()
    if isinstance(outputs, Tensor):
        # pyre-ignore
        rnn_adapter._cached_states.append(outputs)
    else:
        assert isinstance(outputs, Iterable)
        # pyre-ignore
        rnn_adapter._cached_states.append(outputs[0])
    return outputs


def merge_states(
    module: nn.Module,
    inputs: Tuple[Tensor],
    outputs: Union[Tensor, Tuple[Tensor]],
    rnn_adapter: nn.Module,
) -> Union[Tensor, Tuple[Tensor]]:
    """
    Merge rnn states with backbone states. This function will
    be registered as a forward hook at the last backbone layer
    during training.

    Args
        module:
            the last backbone layer
        inputs:
            original inputs of backbone layer
        outputs:
            original outputs of backbone layer
        rnn_adapter:
            the RNNAdapter module attached to backbone

    Returns:
        the merged outputs
    """

    rnn_outputs = rnn_adapter()
    if isinstance(outputs, Tensor):
        return outputs + rnn_outputs
    else:
        assert isinstance(outputs, Iterable)
        output_list = [outputs[0] + rnn_outputs] + list(outputs[1:])
        return tuple(output_list)


def rnn_adapter_init(
    model: nn.Module,
    layers: List[nn.Module],
    merge_module: nn.Module,
    rnn_adapter_conf: RNNAdapterConf,
) -> nn.Module:

    rnn_adapter = hydra.utils.instantiate(
        rnn_adapter_conf,
        _recursive_=False,
    )
    rnn_adapter.to(next(model.parameters()).device)

    for p in rnn_adapter.parameters():
        p.requires_grad = True

    # pyre-ignore
    model.cache_hooks = [
        module.register_forward_hook(
            partial(
                cache_states,
                rnn_adapter=rnn_adapter,
                empty_cache=True if idx == 0 else False,
            )) for idx, module in enumerate(layers)
    ]
    # pyre-ignore
    model.merge_hook = merge_module.register_forward_hook(
        partial(merge_states, rnn_adapter=rnn_adapter))

    model.register_module("rnn_adapter", rnn_adapter)

    return rnn_adapter


class Adapter(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        ft_type: str = "adapter",
    ) -> None:
        super().__init__()

        self.projection_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, inputs: Tensor) -> Tensor:

        results = inputs

        results = self.projection_layer(results)
        results = self.activation(results)
        results = self.output_layer(results)

        results = results + inputs

        return results


@dataclass
class AdapterConf(FTConf):
    _target_: str = get_class_name_str(Adapter)
    ft_type: str = "adapter"
    input_dim: int = MISSING
    hidden_dim: int = MISSING


def adapter_insertion(
    module: nn.Module,
    inputs: Tuple[Tensor],
    outputs: Union[Tensor, Tuple[Tensor]],
    adapter: nn.Module,
) -> Union[Tensor, Tuple[Tensor]]:

    if isinstance(outputs, Tensor):
        return adapter(outputs)
    else:
        assert isinstance(outputs, Iterable)
        main_outputs = outputs[0]
        output_list = [adapter(main_outputs)] + list(outputs[1:])
        return tuple(output_list)


# pyre-ignore
def adapter_init(
    model: nn.Module,
    layers: List[nn.Module],
    adapter_conf: AdapterConf,
    std: Optional[float] = None,
):

    device = next(model.parameters()).device
    adapters = [
        hydra.utils.instantiate(
            adapter_conf,
            _recursive_=False,
        ).to(device) for _ in range(len(layers))
    ]

    for adapter in adapters:
        adapter_state_dict = {}
        for name, p in adapter.named_parameters():
            if "layer_norm" not in name and std is not None:
                adapter_state_dict[name] = torch.normal(0, std, p.shape)
            p.requires_grad = True
        adapter.load_state_dict(adapter_state_dict, strict=False)

    # pyre-ignore
    model.hooks = [
        module.register_forward_hook(
            partial(adapter_insertion, adapter=adapter))
        for module, adapter in zip(layers, adapters)
    ]

    model.register_module("adapters", nn.ModuleList(adapters))

    return adapters


cs: ConfigStore = ConfigStore.instance()

cs.store(
    group="task/peft",
    name="adapter",
    node=AdapterConf(
        input_dim=768,
        hidden_dim=200,
    ),
)

cs.store(
    group="task/peft",
    name="rnn_adapter",
    node=RNNAdapterConf(
        input_dim=768,
        rnn_dim=256,
        num_layers=1,
        rnn_type="gru",
    ),
)

cs.store(
    group="task/peft",
    name="full",
    node=FTConf(ft_type="full", ),
)

cs.store(
    group="task/peft",
    name="bias",
    node=PartialFTConf(ft_key="bias", ),
)

cs.store(
    group="task/peft",
    name="head",
    node=PartialFTConf(ft_key="head", ),
)
