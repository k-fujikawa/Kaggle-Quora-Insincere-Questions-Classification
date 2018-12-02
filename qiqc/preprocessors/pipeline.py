class PreprocessPipeline(object):

    def __init__(self, *preprocessors):
        self.preprocessors = preprocessors

    def __call__(self, x):
        for preprocessor in self.preprocessors:
            x = preprocessor(x)
        return x
