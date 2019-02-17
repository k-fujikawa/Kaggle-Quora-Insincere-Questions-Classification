from abc import ABCMeta, abstractmethod


class BaseEnsembler(metaclass=ABCMeta):

    def __init__(self, config, models, results):
        super().__init__()
        self.config = config
        self.models = models
        self.results = results

    @abstractmethod
    def fit(self, X, t, test_size=0.1):
        pass

    @abstractmethod
    def predict_proba(self, X, X2):
        pass

    def predict(self, X, X2):
        y = self.predict_proba(X, X2)
        return (y > self.threshold).astype('i')
