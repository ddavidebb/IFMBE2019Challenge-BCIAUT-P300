import numpy as np
import torch as th
from torch import nn
from torch.nn import init

class Expression(th.nn.Module):
    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if (hasattr(self.expression_fn, 'func') and
                  hasattr(self.expression_fn, 'kwargs')):
                expression_str = "{:s} {:s}".format(
                    self.expression_fn.func.__name__,
                    str(self.expression_fn.kwargs))
        else:
            expression_str = self.expression_fn.__name__
        return (self.__class__.__name__ + '(' +
                'expression=' + str(expression_str) + ')')

def glorot_weight_zero_bias(model):
    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('BatchNorm' in module.__class__.__name__):
                init.xavier_uniform_(module.weight, gain=1)
            else:
                init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                init.constant_(module.bias, 0)

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = th.renorm(self.weight.data, p=2, dim=0,
                                     maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(object):
    def __init__(self,
                 in_chans=8,
                 n_classes=2,
                 input_time_length=140,
                 F1=8,
                 D=2,
                 F2=16,
                 first_pool_size=4,
                 first_pool_stride=4,
                 second_pool_factor=2,
                 temporal_kernel_length=65,
                 separable_kernel_length=17,
                 drop_prob=0.25
                 ):
        self.__dict__.update(locals())
        del self.self
    def create_network(self):
        model = nn.Sequential()
        model.add_module('dimshuffle', Expression(_transpose_to_b_1_c_0))

        model.add_module('conv_time', nn.Conv2d(
                1, self.F1, (1, self.temporal_kernel_length), stride=1, bias=False,
                padding=(0, self.temporal_kernel_length // 2,)))
        model.add_module('bnorm_temporal', nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3))
        model.add_module('conv_spatial', Conv2dWithConstraint(
            self.F1, self.F1 * self.D, (self.in_chans, 1), stride=1, bias=False, max_norm=1,
            groups=self.F1,
            padding=(0, 0)))
        model.add_module('bnorm_1', nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3))
        model.add_module('elu_1', nn.ELU())
        model.add_module('pool_1', nn.AvgPool2d(
            kernel_size=(1, self.first_pool_size), stride=(1, self.first_pool_stride)))
        model.add_module('drop_1', nn.Dropout(p=self.drop_prob))

        model.add_module('conv_separable_depth', nn.Conv2d(
            self.F1 * self.D, self.F1 * self.D, (1, self.separable_kernel_length), stride=1, bias=False, groups=self.F1 * self.D,
            padding=(0, (self.separable_kernel_length // 2))))
        model.add_module('conv_separable_point', nn.Conv2d(
            self.F1 * self.D, self.F2, (1, 1), stride=1, bias=False,
            padding=(0, 0)))
        model.add_module('bnorm_2', nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3))
        model.add_module('elu_2', nn.ELU())
        model.add_module('pool_2', nn.AvgPool2d(
            kernel_size=(1, self.second_pool_factor*self.first_pool_size), stride=(1, self.second_pool_factor*self.first_pool_stride)))
        model.add_module('drop_2', nn.Dropout(p=self.drop_prob))

        out = model(th.tensor(np.ones((1, self.in_chans, self.input_time_length, 1), dtype=np.float32), requires_grad=False))
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]

        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        model.add_module('conv_classifier', Conv2dWithConstraint(
            self.F2, self.n_classes,
            (n_out_virtual_chans, self.final_conv_length,), bias=True, max_norm=0.25))
        model.add_module('logsoftmax', nn.LogSoftmax())
        model.add_module('permute_back', Expression(_transpose_1_0))
        model.add_module('squeeze', Expression(_squeeze_final_output))

        glorot_weight_zero_bias(model)
        return model


def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)


def _transpose_1_0(x):
    return x.permute(0, 1, 3, 2)


def _squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x
