from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from typing import Any, Iterable, List, Optional, Tuple, Union

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

class RNNAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        rnn_dim: int,
        num_layers: int,
        num_adapted_layers: int,
        rnn_type: str = "rnn",
        layer_norm_before: bool = True,
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
            num_adapted_layers:
                number of backbone layers to adapt
            rnn_type:
                the variant name of RNN; must be one of 'rnn', 'lstm', or 'gru'.
            layer_norm_before:
                whether to apply layer normalization to inputs
            agg_type:
                aggregation type for RNN states; must be one of 'mean' or 'last'.
        """

        super().__init__()

        self.rnn_dim = rnn_dim
        self.num_layers = num_layers
        self.num_adapted_layers = num_adapted_layers
        self.rnn_type = rnn_type
        self.agg_type = agg_type
        self.layer_norm_before = layer_norm_before
        self.layer_norm: Optional[nn.Module] = None
        if layer_norm_before:
            self.layer_norm: Optional[nn.Module] = nn.LayerNorm(input_dim)

        if rnn_type == "lstm":
            rnn_module = nn.LSTM
        elif rnn_type == "gru":
            rnn_module = nn.GRU
        elif rnn_type == "rnn":
            rnn_module = nn.RNN
        else:
            raise ValueError(f"unsupported rnn type {rnn_type}")

        self.rnn: nn.Module = rnn_module(
            input_size=input_dim,
            hidden_size=rnn_dim,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
        )

        self.output_projection = nn.Linear(rnn_dim, input_dim)

        self._cached_states: List[Tensor] = []

    def empty_cache(self) -> None:

        self._cached_states = []

    def forward(self) -> Tensor:

        inputs = self._cached_states
        # TODO: enable selected layers instead of truncation.
        inputs = inputs[: self.num_adapted_layers]
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

        inputs = [states.view(rnn_bsz, -1) for states in inputs]
        inputs = torch.cat(inputs, 0).view(
            len(inputs), rnn_bsz, -1).transpose(0, 1)
        if self.layer_norm_before:
            assert isinstance(self.layer_norm, nn.Module)
            inputs = self.layer_norm(inputs)

        # rnn_outputs is a 3-tuple. The first one is the hidden states
        # of shape (rnn_bsz, #backbone_layers, rnn_dim)
        rnn_outputs = self.rnn(inputs)

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

        return outputs


@dataclass
class RNNAdapterConf(FTConf):
    _target_: str = get_class_name_str(RNNAdapter)
    ft_type: str = "rnn_adapter"
    input_dim: int = MISSING
    rnn_dim: int = MISSING
    num_layers: int = MISSING
    num_adapted_layers: int = MISSING
    rnn_type: str = "rnn"
    agg_type: str = "mean"
    # Unlike the standard `layer_norm_before` arg, when set to False
    # will simply skip layer normalization of inputs. No layer norm
    # will be applied to the outputs whatsoever.
    layer_norm_before: bool = True


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


# pyre-fixme[24]: Generic type `AbstractContextManager` expects 1 type parameter.
class RNNAdapterContextManager(AbstractContextManager):
    def __init__(
        self, modules: Union[List[nn.Module], nn.ModuleList], rnn_adapter: nn.Module
    ) -> None:
        """
        A context manager that register forward hooks to cache backbone layers' states
        and then run RNNAdapter.

        Args
            modules
                The list of backbone layers to adapt
            rnn_adapter
                The RNNAdapter module to be attached
        """

        self.modules = modules
        self.rnn_adapter = rnn_adapter

    def __enter__(self) -> None:

        # pyre-ignore
        self.cache_hooks = [
            module.register_forward_hook(
                partial(
                    cache_states,
                    rnn_adapter=self.rnn_adapter,
                    empty_cache=True if idx == 0 else False,
                )
            )
            for idx, module in enumerate(self.modules)
        ]

        # pyre-ignore
        self.merge_hook = self.modules[-1].register_forward_hook(
            partial(merge_states, rnn_adapter=self.rnn_adapter)
        )

    # pyre-ignore
    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:

        # pyre-ignore
        for hook in self.cache_hooks:
            hook.remove()

        # pyre-ignore
        self.merge_hook.remove()


def rnn_adapter_init(
    model: nn.Module,
    layers: List[nn.Module],
    merge_module: nn.Module,
    rnn_adapter_conf: RNNAdapterConf,
    register_hooks: bool = True,
) -> nn.Module:

    rnn_adapter_conf.num_adapted_layers = len(layers)
    rnn_adapter = hydra.utils.instantiate(
        rnn_adapter_conf,
        _recursive_=False,
    )
    rnn_adapter.to(next(model.parameters()).device)

    for p in rnn_adapter.parameters():
        p.requires_grad = True

    if register_hooks:
        # pyre-ignore
        model.cache_hooks = [
            module.register_forward_hook(
                partial(
                    cache_states,
                    rnn_adapter=rnn_adapter,
                    empty_cache=True if idx == 0 else False,
                )
            )
            for idx, module in enumerate(layers)
        ]
        # pyre-ignore
        model.merge_hook = merge_module.register_forward_hook(
            partial(merge_states, rnn_adapter=rnn_adapter)
        )

    model.register_module("rnn_adapter", rnn_adapter)

    return rnn_adapter


class Adapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        res: bool,
        layer_norm_before: bool,
        ft_type: str = "adapter",
    ) -> None:
        super().__init__()

        self.layer_norm_before = layer_norm_before
        if layer_norm_before:
            self.layer_norm: nn.Module = torch.nn.LayerNorm(input_dim)
        else:
            self.layer_norm: nn.Module = torch.nn.LayerNorm(output_dim)
        self.projection_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.res: bool = res and (input_dim == output_dim)

    def forward(self, inputs: Tensor) -> Tensor:

        results = inputs

        if self.layer_norm_before:
            results = self.layer_norm(results)

        results = self.projection_layer(results)
        results = self.activation(results)
        results = self.output_layer(results)

        if self.res:
            results = results + inputs

        if not self.layer_norm_before:
            results = self.layer_norm(results)

        return results


@dataclass
class AdapterConf(FTConf):
    _target_: str = get_class_name_str(Adapter)
    ft_type: str = "adapter"
    input_dim: int = MISSING
    hidden_dim: int = MISSING
    output_dim: int = MISSING
    res: bool = True
    # this depends on the type of pretrained-model and should not be arbitrary;
    # for example HuggingFace T5 assumes `layer_norm_before=True`, therefore
    # we have to set it to True to make the adaptation correct.
    layer_norm_before: bool = True


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


# pyre-fixme[24]: Generic type `AbstractContextManager` expects 1 type parameter.
class AdapterContextManager(AbstractContextManager):
    def __init__(self, modules: List[nn.Module], adapters: List[nn.Module]) -> None:

        assert len(modules) == len(adapters)
        self.modules = modules
        self.adapters = adapters

    def __enter__(self) -> None:
        # pyre-ignore
        self.hooks = [
            module.register_forward_hook(
                partial(adapter_insertion, adapter=adapter))
            for module, adapter in zip(self.modules, self.adapters)
        ]

    # pyre-ignore
    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:

        # pyre-ignore
        for hook in self.hooks:
            hook.remove()


# pyre-ignore
def adapter_init(
    model: nn.Module,
    layers: List[nn.Module],
    adapter_conf: AdapterConf,
    std: Optional[float] = None,
    register_hooks: bool = True,
):

    device = next(model.parameters()).device
    adapters = [
        hydra.utils.instantiate(
            adapter_conf,
            _recursive_=False,
        ).to(device)
        for _ in range(len(layers))
    ]

    for adapter in adapters:
        adapter_state_dict = {}
        for name, p in adapter.named_parameters():
            if "layer_norm" not in name and std is not None:
                adapter_state_dict[name] = torch.normal(0, std, p.shape)
            p.requires_grad = True
        adapter.load_state_dict(adapter_state_dict, strict=False)

    if register_hooks:
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
        output_dim=768,
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
    node=FTConf(
        ft_type="full",
    ),
)

cs.store(
    group="task/peft",
    name="bias",
    node=PartialFTConf(
        ft_key="bias",
    ),
)

cs.store(
    group="task/peft",
    name="head",
    node=PartialFTConf(
        ft_key="head",
    ),
)
