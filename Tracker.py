class BaseTracker():
    def __init__(self):
        super(HaarDetector, self).__init__()
        self.init_model()

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def update(self,dets):
        return bbox,id
