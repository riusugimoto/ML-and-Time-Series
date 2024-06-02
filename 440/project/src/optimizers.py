import torch as t
import torch.optim
import torch.nn as nn




class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, momentum=0, dampening=0):
        params = list(params)
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.t = 0
        self.b = [t.zeros_like(p) for p in self.params]
        defaults = dict(params = params, lr=lr, momentum=momentum, dampening=dampening)
        super(SGD, self).__init__(params, defaults)
        
    @t.inference_mode()
    def step(self):
        self.t += 1
        for i,p in enumerate(self.params):
            grad = p.grad
            if self.momentum != 0:
                if self.t > 1:
                    self.b[i] = self.momentum*self.b[i] + (1-self.dampening)*grad
                else:
                    self.b[i] = grad
                grad = self.b[i]
            self.params[i] -= self.lr*grad

class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, ep=10e-8):
        params = list(params) #generator
        self.params = params
        self.t = 0
        self.m = [t.zeros_like(p) for p in self.params] #first moment vector
        self.v = [t.zeros_like(p) for p in self.params] #second moment vector
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.ep = ep
        defaults = dict(params = self.params, lr = lr, b1 = b1, b2 = b2, ep = ep)
        super(Adam, self).__init__(params, defaults)

    @t.inference_mode()
    def step(self):
        self.t += 1
        for i,p in enumerate(self.params):
            grad = p.grad
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*grad
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*(grad**2)
            m_hat = self.m[i]/(1-self.b1**self.t)
            v_hat = self.v[i]/(1-self.b2**self.t)
            self.params[i] -= self.lr*m_hat/(t.sqrt(v_hat)+self.ep)

class RMSProp(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, ep=1e-8):
        params = list(params) # generator
        self.params = params
        self.sqa = [t.zeros_like(p) for p in self.params] # square average
        self.lr = lr
        self.alpha = alpha
        self.ep = ep
        defaults= dict(params = self.params, lr = lr, alpha = alpha, ep = ep)
        super(RMSProp, self).__init__(params, defaults)

    @t.inference_mode()
    def step(self):
        for i,p in enumerate(self.params):
            grad = p.grad
            self.sqa[i] = self.alpha*self.sqa[i] + (1 - self.alpha)*(grad**2)
            self.params[i] -= self.lr*grad/(t.sqrt(self.sqa[i]) + self.ep)


class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, ep=1e-8):
        params = list(params) #generator
        self.params = params
        self.lr = lr
        self.ep = ep
        self.st = [t.zeros_like(p) for p in self.params] #state sum
        defaults= dict(params = self.params, lr = lr, ep = ep)
        super(AdaGrad, self).__init__(params, defaults)

    @t.inference_mode()
    def step(self):
        for i,p in enumerate(self.params):
            grad = p.grad
            self.st[i] += grad**2
            self.params[i] -= self.lr*grad/(t.sqrt(self.st[i]) + self.ep)