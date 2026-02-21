# app.py
from flask import Flask, render_template, request, jsonify
import analyzer

app = Flask(__name__)

@app.route('/')
def index():
    """Halaman utama frontend"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Endpoint untuk analisis.
    Menerima JSON dengan field 'pair' dan 'news'.
    Mengembalikan JSON hasil analisis.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body harus JSON'}), 400

        pair = data.get('pair', 'EURUSD')
        news = data.get('news', '')

        # Validasi pair
        valid_pairs = ['EURUSD', 'GBPUSD', 'EURJPY', 'GBPJPY', 'CHFJPY']
        if pair not in valid_pairs:
            return jsonify({'error': f'Pair tidak valid. Pilih dari {valid_pairs}'}), 400

        # Panggil fungsi analisis
        result = analyzer.analyze_pair(pair, news)
        return jsonify(result)

    except Exception as e:
        # Tangkap error tak terduga
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Endpoint untuk mengecek apakah server hidup"""
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    # Jalankan server Flask
    app.run(host='0.0.0.0', port=5000, debug=True)
