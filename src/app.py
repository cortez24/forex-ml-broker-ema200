from flask import Flask, render_template, request, jsonify
import analyzer

app = Flask(__name__)

def get_pair():
    # Coba ambil dari JSON
    if request.is_json:
        data = request.get_json()
        if data and "pair" in data:
            return data["pair"]

    # Coba ambil dari form
    if "pair" in request.form:
        return request.form.get("pair")

    return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/download", methods=["POST"])
def download():
    pair = get_pair()

    if not pair:
        return jsonify({"error": "Pair kosong"}), 400

    analyzer.download_data(pair)
    return jsonify({"status": "Data Downloaded"})


@app.route("/backtest", methods=["POST"])
def backtest():
    pair = get_pair()

    if not pair:
        return jsonify({"error": "Pair kosong"}), 400

    result = analyzer.backtest(pair)
    return jsonify(result)


@app.route("/train", methods=["POST"])
def train():
    pair = get_pair()

    if not pair:
        return jsonify({"error": "Pair kosong"}), 400

    acc = analyzer.train_ml(pair)
    return jsonify({"accuracy": acc})


@app.route("/predict", methods=["POST"])
def predict():
    pair = get_pair()

    if not pair:
        return jsonify({"error": "Pair kosong"}), 400

    prob = analyzer.ml_predict(pair)
    return jsonify({"probability": prob})


if __name__ == "__main__":
    app.run(debug=True)
