import constants
from qwen_model import QwenModel

class ModelFactory:
    def __init__(self, model_name, algo):
        self.model_name = model_name
        self.algo = algo
    
    def get_model(self):
        match self.model_name:
            case constants.QwenMaxModelName:
                return QwenModel(self.algo, constants.QwenMaxModelName)
            case constants.QwenPlusModelName:
                return QwenModel(self.algo, constants.QwenPlusModelName)
            case _:
                 raise NotImplementedError("Model is not supported.")