import numpy as np
from PIL import Image
import tensorflow as tf
import sys

class NeuralNetwork:
    def __init__(self, pathFile):
        self.labels = self.loadLabels('model/class_labels.txt')
        self.graph = "model/trained_model.tflite"
        self.inputPath = pathFile
        self.returnedValue = dict()
    
    def run(self):
        interpreter = tf.lite.Interpreter(model_path=self.graph)
        interpreter.allocate_tensors()
        inputDetails = interpreter.get_input_details()
        outputDetails = interpreter.get_output_details()
        floatingModel = inputDetails[0]['dtype'] == np.float32
        height = inputDetails[0]['shape'][1]
        width = inputDetails[0]['shape'][2]
        img = Image.open(self.inputPath).resize((width, height))
        inputData = np.expand_dims(img, axis=0)

        if floatingModel:
            inputData = np.float32(inputData) / 255

        interpreter.set_tensor(inputDetails[0]['index'], inputData)
        interpreter.invoke()
        outputData = interpreter.get_tensor(outputDetails[0]['index'])
        results = np.squeeze(outputData)
        sortedIndex = results.argsort()[-5:][::-1]
        for i in sortedIndex:
            if floatingModel:
                self.returnedValue[self.labels[i]] = float(results[i])
                #self.returnedValue = dict(zip(self.labels, [float(x) for x in results]))
                #print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
            else:
                self.returnedValue[self.labels[i]] = float(results[i])/255.0
                #self.returnedValue = dict(zip(self.labels, [float(x)/255.0 for x in results]))
                
    def show(self):
        for keys, values in self.returnedValue.items():
            print(keys + ": ")
            print(values)

    def loadLabels(self, filename):
        with open(filename, 'r') as f:
            return [line.strip() for line in f.readlines()]

if __name__ == '__main__':
    helmut = NeuralNetwork(sys.argv[1])
    helmut.run()
    helmut.show()
