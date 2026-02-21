# app.py
from flask import Flask, render_template, request, jsonify
import analyzer
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body harus JSON'}), 400
        pair = data.get('pair', 'EURUSD')
        news = data.get('news', '')
        valid_pairs = ['EURUSD', 'GBPUSD', 'EURJPY', 'GBPJPY', 'CHFJPY']
        if pair not in valid_pairs:
            return jsonify({'error': f'Pair tidak valid. Pilih dari {valid_pairs}'}), 400
        logging.info(f"Request analisis untuk {pair}")
        result = analyzer.analyze_pair(pair, news)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/backtest', methods=['GET', 'POST'])
def backtest():
    if request.method == 'GET':
        return render_template('backtest.html')
    else:
        try:
            data = request.get_json()
            pair = data.get('pair', 'EURUSD')
            start_date = data.get('start_date', '2023-01-01')
            end_date = data.get('end_date', '2024-01-01')
            min_confidence = float(data.get('min_confidence', 65))
            result = analyzer.run_backtest(pair, start_date, end_date, min_confidence)
            return jsonify(result)
        except Exception as e:
            logging.error(f"Error backtest: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

@app.route('/api/backtest-realtime', methods=['POST'])
def backtest_realtime():
    try:
        data = request.get_json()
        pair = data.get('pair', 'EURUSD')
        interval = data.get('interval', '15min')
        days_back = int(data.get('days_back', 90))
        min_confidence = float(data.get('min_confidence', 65))
        result = analyzer.run_backtest_realtime(pair, interval, days_back, min_confidence)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error realtime backtest: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/live-prediction', methods=['POST'])
def live_prediction():
    try:
        data = request.get_json()
        pair = data.get('pair', 'EURUSD')
        result = analyzer.get_live_prediction(pair)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error live prediction: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
