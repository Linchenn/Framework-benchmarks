from abc import abstractmethod
import tensorflow as tf
import time


class Profiler:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.input = tf.ones(self.input_shape)

    @abstractmethod
    def predict_func(self, input):
        pass
    
    def benchmark(self, num_warms = 50, num_runs = 200):
        res = None
        for _ in range(num_warms):
            res = self.predict_func(self.input)
        
        start = time.time()
        for _ in range(num_runs):
            inter_res = self.predict_func(self.input)
            res = tf.math.add(x=inter_res, y=res)
        delta = time.time() - start
        return delta / num_runs
    

class Topformer_base_512(Profiler):
    MODEL_PATH = './TensorFlow/models/topformer/topformer_base_512x512'
    INPUT_SHAPE = [1, 512, 512, 3]
    def __init__(self):
        self.model = tf.saved_model.load(self.MODEL_PATH)
        self.model_execute = self.model.signatures["serving_default"]
        super().__init__(self.INPUT_SHAPE)

    def predict_func(self, input):
        return self.model_execute(input)['tf.identity']
    

class Topformer_small_512(Profiler):
    MODEL_PATH = './TensorFlow/models/topformer/topformer_small_512x512'
    INPUT_SHAPE = [1, 512, 512, 3]
    def __init__(self):
        self.model = tf.saved_model.load(self.MODEL_PATH)
        self.model_execute = self.model.signatures["serving_default"]
        super().__init__(self.INPUT_SHAPE)

    def predict_func(self, input):
        return self.model_execute(input)['tf.identity']


class Topformer_tiny_512(Profiler):
    MODEL_PATH = './TensorFlow/models/topformer/topformer_tiny_512x512'
    INPUT_SHAPE = [1, 512, 512, 3]
    def __init__(self):
        self.model = tf.saved_model.load(self.MODEL_PATH)
        self.model_execute = self.model.signatures["serving_default"]
        super().__init__(self.INPUT_SHAPE)

    def predict_func(self, input):
        return self.model_execute(input)['tf.identity']


class Topformer_tiny_448(Profiler):
    MODEL_PATH = './TensorFlow/models/topformer/topformer_tiny_448x448'
    INPUT_SHAPE = [1, 448, 448, 3]
    def __init__(self):
        self.model = tf.saved_model.load(self.MODEL_PATH)
        self.model_execute = self.model.signatures["serving_default"]
        super().__init__(self.INPUT_SHAPE)

    def predict_func(self, input):
        return self.model_execute(input)['tf.identity']
    
    
if __name__ == '__main__':
    tb5 = Topformer_base_512()
    ts5 = Topformer_small_512()
    tt5 = Topformer_tiny_512()
    tt4 = Topformer_tiny_448()
    print('Base 512: ', tb5.benchmark(50, 200))
    print('Small 512: ', ts5.benchmark(50, 200))
    print('Tiny 512: ', tt5.benchmark(50, 200))
    print('Tiny 448: ', tt4.benchmark(50, 200))
        