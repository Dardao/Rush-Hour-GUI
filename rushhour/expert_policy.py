"""A feedforward neural policy. No board database, puzzle IDs, or search."""
import numpy as np

class PolicyNetwork:
    format='rushhour-imitation-mlp-v1'
    policy_only=True
    training_scope='all2500_supervised'
    def __init__(self,widths=(224,128,64,140),seed=42):
        self.training_scope='untrained'
        self.widths=tuple(widths);rng=np.random.default_rng(seed)
        assert self.widths[0]==224 and self.widths[-1]==140
        self.w=[(rng.standard_normal((i,o))*np.sqrt(2/i)).astype(np.float32) for i,o in zip(widths[:-1],widths[1:])]
        self.b=[np.zeros(o,np.float32) for o in widths[1:]]
        self.m=[np.zeros_like(p) for p in self.w+self.b];self.v=[np.zeros_like(p) for p in self.w+self.b];self.t=0

    def copy(self):
        net=type(self)(self.widths)
        net.w=[a.copy() for a in self.w];net.b=[a.copy() for a in self.b]
        net.m=[a.copy() for a in self.m];net.v=[a.copy() for a in self.v];net.t=self.t
        net.training_scope=self.training_scope
        return net

    def forward(self,x):
        h=np.asarray(x,dtype=np.float32)
        for w,b in zip(self.w[:-1],self.b[:-1]):h=np.maximum(h@w+b,0)
        logits=h@self.w[-1]+self.b[-1]
        # Adapter for existing greedy rollout's 140 policy + 1 value output.
        return np.concatenate([logits,np.zeros((len(x),1),np.float32)],axis=1)

    def loss_grad(self,x,y,mask):
        hs=[np.asarray(x,dtype=np.float32)]
        for w,b in zip(self.w[:-1],self.b[:-1]):hs.append(np.maximum(hs[-1]@w+b,0))
        scores=hs[-1]@self.w[-1]+self.b[-1]
        logits=np.where(mask,scores,-1e9);logits-=logits.max(1,keepdims=True)
        probabilities=np.exp(logits);probabilities/=probabilities.sum(1,keepdims=True)
        loss=-np.log(np.maximum(probabilities[np.arange(len(x)),y],1e-12)).mean()
        accuracy=(logits.argmax(1)==y).mean()
        grad=probabilities;grad[np.arange(len(x)),y]-=1;grad/=len(x)
        dw=[None]*len(self.w);db=[None]*len(self.b)
        for j in reversed(range(len(self.w))):
            dw[j]=hs[j].T@grad;db[j]=grad.sum(0)
            if j:grad=(grad@self.w[j].T)*(hs[j]>0)
        return float(loss),float(accuracy),dw+db

    def train_batch(self,x,y,mask,lr=.001):
        loss,acc,grads=self.loss_grad(x,y,mask);self.t+=1
        norm=np.sqrt(sum(float((g*g).sum()) for g in grads));scale=min(1,5/max(norm,1e-12))
        for i,(p,g) in enumerate(zip(self.w+self.b,grads)):
            g*=scale;self.m[i]=.9*self.m[i]+.1*g;self.v[i]=.999*self.v[i]+.001*g*g
            p-=lr*(self.m[i]/(1-.9**self.t))/(np.sqrt(self.v[i]/(1-.999**self.t))+1e-8)
        return loss,acc

    def save(self,path):
        with open(path,'wb') as f:np.savez_compressed(f,format=np.array(self.format),widths=np.array(self.widths),training_method=np.array('supervised APK expert demonstrations'),training_scope=np.array(self.training_scope),adam_step=np.array(self.t),**{f'{k}{i}':a for k in ['w','b','m','v'] for i,a in enumerate(getattr(self,k))})

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            if d['format'].item()!=cls.format:raise ValueError('Unsupported model format')
            net=cls(tuple(int(n) for n in d['widths']))
            net.training_scope=str(d['training_scope'].item()) if 'training_scope' in d else 'all2500_supervised'
            net.t=int(d['adam_step'].item()) if 'adam_step' in d else 0
            for k in ['w','b','m','v']:
                for i,p in enumerate(getattr(net,k)):
                    if f'{k}{i}' not in d and k in ('m','v'):continue
                    a=d[f'{k}{i}'];assert a.shape==p.shape and np.isfinite(a).all();p[:]=a
        return net
