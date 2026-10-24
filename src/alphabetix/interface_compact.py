"""Optional compact state/model conversion after the standard builder."""

from . import interface
from .models.compact_network import CompactNetwork
from .models.compact_network_model import CompactNetworkModel


def build(
    dataset,
    decode_timesteps,
    connectivity,
    settings,
    probes,
    *,
    compact,
):
    """Build from TOML settings, with the original dense backend also available."""
    model, network, neurons, probes = interface.build(
        dataset,
        decode_timesteps,
        connectivity,
        interface.Settings(**settings),
        probes,
    )
    if compact:
        model = model.replace(
            network_model=CompactNetworkModel.from_model(model.network_model)
        )
        network = CompactNetwork.from_network(network, neurons)
    return model, network, neurons, probes
