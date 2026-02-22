from flask import Flask, request, jsonify, render_template
import os
import traceback
from config import PAIRS, TIMEFRAMES, SIGNAL_TIMEFRAME
from analyzer import generate_signal, performance_report, get_latest_candle
from data_loader import load_historical_data, get_realtime_price

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/signal', methods=['POST'])
def api_signal():
    data = request.get_json()
    if not data or 'pair' not in data:
        return jsonify({'error': 'Pair tidak diberikan'}), 400

    pair = data['pair'].upper()
    timeframe = data.get('timeframe', SIGNAL_TIMEFRAME)

    if pair not in PAIRS:
        return jsonify({'error': f'Pair tidak didukung. Pilihan: {", ".join(PAIRS)}'}), 400
    if timeframe not in TIMEFRAMES:
        return jsonify({'error': f'Timeframe tidak didukung. Pilihan: {", ".join(TIMEFRAMES)}'}), 400

    try:
        result = generate_signal(pair, timeframe)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({'error': f'Data historis tidak ditemukan: {str(e)}'}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@app.route('/api/performance', methods=['POST'])
def api_performance():
    data = request.get_json()
    if not data or 'pair' not in data:
        return jsonify({'error': 'Pair tidak diberikan'}), 400

    pair = data['pair'].upper()
    timeframe = data.get('timeframe', SIGNAL_TIMEFRAME)

    if pair not in PAIRS:
        return jsonify({'error': f'Pair tidak didukung. Pilihan: {", ".join(PAIRS)}'}), 400
    if timeframe not in TIMEFRAMES:
        return jsonify({'error': f'Timeframe tidak didukung. Pilihan: {", ".join(TIMEFRAMES)}'}), 400

    try:
        result = performance_report(pair, timeframe)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({'error': f'Data historis tidak ditemukan: {str(e)}'}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@app.route('/api/realtime', methods=['POST'])
def api_realtime():
    data = request.get_json()
    if not data or 'pair' not in data:
        return jsonify({'error': 'Pair tidak diberikan'}), 400

    pair = data['pair'].upper()
    if pair not in PAIRS:
        return jsonify({'error': f'Pair tidak didukung. Pilihan: {", ".join(PAIRS)}'}), 400

    rt = get_realtime_price(pair)
    if rt is None:
        return jsonify({'error': 'Gagal mengambil data real-time'}), 500

    # Konversi datetime ke string
    rt['datetime'] = rt['datetime'].isoformat()
    return jsonify(rt)

@app.route('/api/latest_candle', methods=['POST'])
def api_latest_candle():
    data = request.get_json()
    if not data or 'pair' not in data:
        return jsonify({'error': 'Pair tidak diberikan'}), 400

    pair = data['pair'].upper()
    timeframe = data.get('timeframe', SIGNAL_TIMEFRAME)

    if pair not in PAIRS:
        return jsonify({'error': f'Pair tidak didukung. Pilihan: {", ".join(PAIRS)}'}), 400
    if timeframe not in TIMEFRAMES:
        return jsonify({'error': f'Timeframe tidak didukung. Pilihan: {", ".join(TIMEFRAMES)}'}), 400

    try:
        candle = get_latest_candle(pair, timeframe)
        if candle is None:
            return jsonify({'error': 'Tidak ada data historis'}), 404
        return jsonify(candle)
    except FileNotFoundError as e:
        return jsonify({'error': f'Data historis tidak ditemukan: {str(e)}'}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
