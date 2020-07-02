import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 改动了权重数据的量化
def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** k  - 1)
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply


class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit
    # 符号位 占一位
    self.uniform_q = uniform_quantize(k=w_bit - 1) 

  def forward(self, x):
    # print('===================')
    if self.w_bit == 32:
      # weight_q = x
      weight = torch.tanh(x)
      # weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
      # weight_q = 2 * self.uniform_q(weight) - 1
      weight_q = weight / torch.max(torch.abs(weight))
    elif self.w_bit == 1:
      E = torch.mean(torch.abs(x)).detach()
      weight_q = (self.uniform_q(x / E) + 1) / 2 * E
    else:
      weight = torch.tanh(x)
      # weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
      # weight_q = 2 * self.uniform_q(weight) - 1
      weight = weight / torch.max(torch.abs(weight))
      # 想量化到带符号的 k bit
      weight_q = self.uniform_q(weight)
    return weight_q 


class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit):
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize(k=a_bit)

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = torch.clamp(x, 0, 6)
    else:
      activation_q = self.uniform_q(torch.clamp(x, 0, 1))
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q

class ActQuant_PACT(nn.Module):
    def __init__(self, act_bit=4, scale_coef=1.0):
        super(ActQuant_PACT, self).__init__()
        self.act_bit=act_bit
        self.scale_coef = nn.Parameter(torch.ones(1)*scale_coef)
        
        self.uniform_q = uniform_quantize(k=act_bit)

        # self.uniform_q = uniform_quantize(k=act_bit)

    def forward(self, x):
        if self.act_bit==32:
            out=0.5*(x.abs() - (x-self.scale_coef.abs()).abs()+self.scale_coef.abs())/self.scale_coef.abs()
        else:
            out = 0.5*(x.abs() - (x-self.scale_coef.abs()).abs()+self.scale_coef.abs())
            activation_q = self.uniform_q(out / self.scale_coef)
#        print(self.scale_coef)
	    
#            out = torch.round(out * (2**self.act_bit - 1) / self.scale_coef) / (2**self.act_bit - 1)
        return activation_q


def conv2d_Q_fn(w_bit):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input, order=None):
      weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q

def batchNorm2d_Q_fn(w_bit):
  class BatchNorm2d_Q(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm2d_Q, self).__init__(num_features, eps, momentum, affine,
                 track_running_stats)
        self.w_bit = w_bit
        self.quantize_fn = uniform_quantize(k= w_bit)

    def forward(self, input):
      # return input
      gamma = self.weight
      var = self.running_var
      mean = self.running_mean
      eps = self.eps
      bias = self.bias
      w = gamma / (torch.sqrt(var) + eps)
      b = bias -  (mean / (torch.sqrt(var) + eps)) * gamma

      w = torch.clamp(w, -1, 1) / 2 + 0.5
      # w = w / 2 / torch.max(torch.abs(w)) + 0.5
      w_q = 2 * self.quantize_fn(w) - 1 

      b = torch.clamp(b, -1, 1) / 2 + 0.5
      b_q = 2 * self.quantize_fn(b) - 1
      # b_q = self.quantize_fn(torch.clamp())
      # return w_q * input + b_q
      return F.batch_norm(input, running_mean=mean * 0, running_var=torch.sign(torch.abs(var) + 1), weight=w_q, bias=b_q, eps=eps * 0)

  return BatchNorm2d_Q

def batchNorm1d_Q_fn(w_bit):
  class BatchNorm1d_Q(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm1d_Q, self).__init__(num_features, eps, momentum, affine,
                 track_running_stats)
        self.w_bit = w_bit
        self.quantize_fn = uniform_quantize(k= w_bit)

    # def forward(self, input):
    #   # return input
    #   gamma = self.weight
    #   var = self.running_var
    #   mean = self.running_mean
    #   eps = self.eps
    #   bias = self.bias
    #   w = gamma / (torch.sqrt(var) + eps)
    #   b = (bias -  mean / (torch.sqrt(var) + eps)) * gamma

    #   # w = torch.clamp(w, -1, 1) / 2 + 0.5
    #   # w = w / 2 / torch.max(torch.abs(w)) + 0.5
    #   # w_q = 2 * self.quantize_fn(w) - 1 
    #   w_q = self.quantize_fn(w)

    #   # b = torch.clamp(b, -1, 1) / 2 + 0.5
    #   b_q = self.quantize_fn(b)
    #   # b_q = self.quantize_fn(torch.clamp())
    #   # return w_q * input + b_q
    #   # return F.batch_norm(input, running_mean=mean * 0, running_var=torch.sign(torch.abs(var) + 1), weight=w, bias=b, eps=eps * 0)
    #   return F.batch_norm(input, running_mean=mean, running_var=var, weight=gamma, bias=bias, eps=eps)
    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        
        gamma = self.weight
        var = self.running_var
        mean = self.running_mean
        eps = self.eps
        bias = self.bias
        w = gamma / (torch.sqrt(var) + eps)
        b = bias -  (mean / (torch.sqrt(var) + eps)) * gamma

        # w = torch.clamp(w, -1, 1) / 2 + 0.5
        # w = w / 2 / torch.max(torch.abs(w)) + 0.5
        # w_q = 2 * self.quantize_fn(w) - 1 
        w_q = self.quantize_fn(w)

        # return F.batch_norm(
        #     input, self.running_mean, self.running_var, self.weight, self.bias,
        #     self.training or not self.track_running_stats,
        #     exponential_average_factor, self.eps)
        return F.batch_norm(
            input, mean * 0, torch.sign(var + 1) , w, b,
            self.training or not self.track_running_stats,
            exponential_average_factor, eps * 0)

  return BatchNorm1d_Q



def linear_Q_fn(w_bit):
  class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
      super(Linear_Q, self).__init__(in_features, out_features, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input):
      weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.linear(input, weight_q, self.bias)

  return Linear_Q


if __name__ == '__main__':
  import numpy as np
  import matplotlib.pyplot as plt

  a = torch.rand(1, 3, 32, 32)

  Conv2d = conv2d_Q_fn(w_bit=2)
  conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
  act = activation_quantize_fn(a_bit=3)

  b = conv(a)
  b.retain_grad()
  c = act(b)
  d = torch.mean(c)
  d.retain_grad()

  d.backward()
  pass
