# app.py
from flask import Flask, render_template, request, jsonify
import analyzer
import logging

app = Flask(__name__)

# Konfigurasi logging
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
        logging.error(f"Error dalam analisis: {str(e)}", exc_info=True)
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
