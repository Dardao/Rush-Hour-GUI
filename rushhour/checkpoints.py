"""Load RL and explicitly marked supervised policy checkpoints."""
import numpy as np
from .model import Network
from .expert_policy import PolicyNetwork
from .expanded_rl import ExpandedRL

def load_network(path):
    with np.load(path,allow_pickle=False) as data:
        supervised='format' in data and data['format'].item()==PolicyNetwork.format
        expanded='format' in data and data['format'].item()==ExpandedRL.format
    if expanded:return ExpandedRL.load(path)
    return PolicyNetwork.load(path) if supervised else Network.load(path)
