class BaseTracker(object):
    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def update(self):
        raise EOFError("Undefined model type.")
