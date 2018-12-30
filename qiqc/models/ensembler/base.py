from abc import ABCMeta, abstractmethod


class BaseEnsembler(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, X, t, test_size=0.1):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    def predict(self, X):
        y = self.predict_proba(X)
        return (y > self.threshold).astype('i')
