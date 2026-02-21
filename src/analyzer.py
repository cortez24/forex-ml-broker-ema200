from flask import Flask, render_template, request, jsonify
import analyzer

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/download", methods=["POST"])
def download():
    pair = request.json["pair"]
    analyzer.download_data(pair)
    return jsonify({"status":"Data Downloaded"})

@app.route("/backtest", methods=["POST"])
def backtest():
    pair = request.json["pair"]
    result = analyzer.backtest(pair)
    return jsonify(result)

@app.route("/train", methods=["POST"])
def train():
    pair = request.json["pair"]
    acc = analyzer.train_ml(pair)
    return jsonify({"accuracy":acc})

@app.route("/predict", methods=["POST"])
def predict():
    pair = request.json["pair"]
    prob = analyzer.ml_predict(pair)
    return jsonify({"probability":prob})

if __name__ == "__main__":
    app.run(debug=True)
