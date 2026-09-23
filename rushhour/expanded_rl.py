"""Fresh expanded actor-critic; no supervised initialization or expert targets."""
import numpy as np
import os
import tempfile
from pathlib import Path
from .model import Network

class ExpandedRL(Network):
    shapes=((224,512),(512,512),(512,141))
    format='rushhour-expanded-rl-v1'
    no_answer_training=True
    training_scope='RL_self_generated_experience'

    def __init__(self,seed=42):
        super().__init__(seed)
        self.initial_seed=int(seed)

    def copy(self):
        net=type(self)(self.initial_seed)
        for k in ['w','b','m','v']:setattr(net,k,[a.copy() for a in getattr(self,k)])
        net.t=self.t
        return net

    def train_batch(self,*args,**kwargs):
        raise RuntimeError('External supervised targets are disabled for this RL model')

    def save(self,path):
        path=Path(path)
        # Readers must never see a half-written checkpoint during monitoring.
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as f:
            temporary=f.name
            try:
                np.savez_compressed(f,format=np.array(self.format),seed=np.array(self.initial_seed),adam_step=np.array(self.t),
                    initialization=np.array('fresh random; no teacher weights'),
                    **{f'{k}{i}':a for k in ['w','b','m','v'] for i,a in enumerate(getattr(self,k))})
            except BaseException:
                os.unlink(temporary)
                raise
        os.replace(temporary,path)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            if data['format'].item()!=cls.format:raise ValueError('Expected expanded RL checkpoint')
            net=cls(int(data['seed'].item()))
            for k in ['w','b','m','v']:
                for i,p in enumerate(getattr(net,k)):
                    a=data[f'{k}{i}']
                    if a.shape!=p.shape or not np.isfinite(a).all() or (k=='v' and (a<0).any()):raise ValueError('Invalid checkpoint')
                    # Preserve optimizer dtype for faithful within-run comparisons.
                    getattr(net,k)[i]=a.copy()
            net.t=int(data['adam_step'].item())
            if net.t<0:raise ValueError('Invalid Adam step')
        return net
