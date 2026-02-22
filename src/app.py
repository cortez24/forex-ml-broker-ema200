from flask import Flask, request, jsonify, render_template
import os
import traceback

# Impor fungsi dari analyzer.py
from analyzer import generate_signal, performance_report, download_data, data_file

app = Flask(__name__)

# Halaman utama (menampilkan index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint untuk menghasilkan sinyal trading
@app.route('/api/signal', methods=['POST'])
def api_signal():
    # Ambil data JSON dari request
    data = request.get_json()
    if not data or 'pair' not in data:
        return jsonify({'error': 'Pair tidak diberikan'}), 400

    pair = data['pair'].upper()

    # Pastikan file data tersedia, jika tidak unduh
    if not os.path.exists(data_file(pair)):
        try:
            download_data(pair)
        except Exception as e:
            return jsonify({'error': f'Gagal mengunduh data: {str(e)}'}), 500

    try:
        # Panggil fungsi generate_signal dari analyzer
        result = generate_signal(pair)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

# Endpoint untuk laporan performa
@app.route('/api/performance', methods=['POST'])
def api_performance():
    data = request.get_json()
    if not data or 'pair' not in data:
        return jsonify({'error': 'Pair tidak diberikan'}), 400

    pair = data['pair'].upper()

    # Pastikan file data tersedia
    if not os.path.exists(data_file(pair)):
        try:
            download_data(pair)
        except Exception as e:
            return jsonify({'error': f'Gagal mengunduh data: {str(e)}'}), 500

    try:
        # Panggil fungsi performance_report dari analyzer
        result = performance_report(pair)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

if __name__ == '__main__':
    # Jalankan server Flask di semua interface, port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
