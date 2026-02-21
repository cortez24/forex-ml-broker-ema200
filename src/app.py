from flask import Flask, render_template, request, jsonify, send_file
import analyzer
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

def get_pair():
    if request.is_json:
        return request.get_json().get("pair")
    return request.form.get("pair")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/download", methods=["POST"])
def download():
    pair = get_pair()
    analyzer.download_data(pair)
    return jsonify({"status":"Downloaded"})

@app.route("/signal", methods=["POST"])
def signal():
    pair = get_pair()
    return jsonify(analyzer.generate_signal(pair))

@app.route("/performance", methods=["POST"])
def performance():
    pair = get_pair()
    return jsonify(analyzer.performance_report(pair))

@app.route("/equity_chart", methods=["POST"])
def equity_chart():
    pair = get_pair()
    report = analyzer.performance_report(pair)
    equity = report["equity_curve"]

    plt.figure()
    plt.plot(equity)
    plt.title(f"Equity Curve - {pair}")
    plt.xlabel("Trades")
    plt.ylabel("Equity")

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plt.close()

    return send_file(img, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
