from abc import ABCMeta, abstractmethod

from torch import nn


class NNModuleWrapperBase(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def add_args(cls, parser):
        raise NotImplementedError()

    @abstractmethod
    def add_extra_args(cls, parser):
        raise NotImplementedError()
