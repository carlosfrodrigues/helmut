from flask import Flask, request
from helmut import NeuralNetwork
import json 

app = Flask(__name__)

@app.route('/', methods=["POST"])
def home():
    imgPath = request.files['file']
    helmut = NeuralNetwork(imgPath)
    helmut.run()
    return json.dumps(helmut.returnedValue)

if __name__ == "__main__":
    app.run(debug=True)