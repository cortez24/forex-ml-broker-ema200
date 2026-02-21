<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Forex Analyzer</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #1e1e1e;
            color: #e0e0e0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: auto;
        }
        h1 {
            color: #4CAF50;
            text-align: center;
        }
        .nav {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 20px;
        }
        .nav a {
            color: #e0e0e0;
            text-decoration: none;
            padding: 8px 16px;
            background: #2d2d2d;
            border-radius: 4px;
        }
        .nav a:hover {
            background: #3d3d3d;
        }
        .nav a.active {
            background: #4CAF50;
            color: white;
        }
        .form-group {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            align-items: center;
        }
        label {
            font-weight: bold;
        }
        select, input, button {
            background: #3d3d3d;
            color: #e0e0e0;
            border: 1px solid #555;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 14px;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover {
            background: #45a049;
        }
        .stats {
            background: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .stat-item {
            background: #3d3d3d;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #444;
            padding: 10px;
            text-align: center;
        }
        th {
            background: #333;
        }
        .win { color: #4CAF50; }
        .loss { color: #f44336; }
        .error {
            color: #f44336;
            background: #3d3d3d;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
        }
        .loading {
            display: inline-block;
            width: 30px;
            height: 30px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #4CAF50;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        #apiOptions {
            margin-top: 10px;
            padding: 10px;
            background: #3d3d3d;
            border-radius: 4px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Backtest Forex Analyzer</h1>
        
        <div class="nav">
            <a href="/">Analisis Teknikal</a>
            <a href="/backtest" class="active">Backtest</a>
            <a href="/live">Live Prediction</a>
        </div>
        
        <div class="form-group">
            <label>Pair:</label>
            <select id="pair">
                <option value="EURUSD">EUR/USD</option>
                <option value="GBPUSD">GBP/USD</option>
                <option value="EURJPY">EUR/JPY</option>
                <option value="GBPJPY">GBP/JPY</option>
                <option value="CHFJPY">CHF/JPY</option>
            </select>

            <label>Start Date:</label>
            <input type="date" id="start_date" value="2023-01-01">

            <label>End Date:</label>
            <input type="date" id="end_date" value="2024-01-01">

            <label>Min Confidence (%):</label>
            <input type="number" id="min_confidence" value="65" min="0" max="100" step="1">

            <div style="display: flex; gap: 15px; flex-wrap: wrap; align-items: center;">
                <label>
                    <input type="radio" name="dataSource" value="local" checked> Data Lokal (CSV)
                </label>
                <label>
                    <input type="radio" name="dataSource" value="api"> Data Real-time (Twelve Data API)
                </label>
            </div>

            <div id="apiOptions" style="display: none;">
                <label>Interval:</label>
                <select id="apiInterval">
                    <option value="1h">1 Hour</option>
                    <option value="4h" selected>4 Hours</option>
                    <option value="1day">1 Day</option>
                </select>
                
                <label>Periode (hari):</label>
                <input type="number" id="apiDays" value="90" min="30" max="365">
            </div>

            <button id="runBtn">Jalankan Backtest</button>
            <div id="loading" style="display: none; margin-left: 10px;">
                <div class="loading"></div>
            </div>
        </div>

        <div id="results" style="display: none;"></div>
        <div id="error" class="error" style="display: none;"></div>
    </div>

    <script>
        // Toggle API options
        document.querySelectorAll('input[name="dataSource"]').forEach(radio => {
            radio.addEventListener('change', function() {
                document.getElementById('apiOptions').style.display = 
                    this.value === 'api' ? 'block' : 'none';
            });
        });

        // Fungsi untuk menjalankan backtest
        async function runBacktest() {
            const pair = document.getElementById('pair').value;
            const startDate = document.getElementById('start_date').value;
            const endDate = document.getElementById('end_date').value;
            const minConfidence = document.getElementById('min_confidence').value;
            const dataSource = document.querySelector('input[name="dataSource"]:checked').value;
            
            let url = '/backtest';
            let payload = { pair, min_confidence: minConfidence };
            
            if (dataSource === 'api') {
                url = '/api/backtest-realtime';
                payload.interval = document.getElementById('apiInterval').value;
                payload.days_back = document.getElementById('apiDays').value;
            } else {
                payload.start_date = startDate;
                payload.end_date = endDate;
            }
            
            // Tampilkan loading
            document.getElementById('loading').style.display = 'inline-block';
            document.getElementById('results').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            
            try {
                const response = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('error').textContent = 'Error: ' + data.error;
                    document.getElementById('error').style.display = 'block';
                } else if (data.message) {
                    document.getElementById('error').textContent = data.message;
                    document.getElementById('error').style.display = 'block';
                } else {
                    displayResults(data);
                }
            } catch (err) {
                document.getElementById('error').textContent = 'Gagal terhubung ke server.';
                document.getElementById('error').style.display = 'block';
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        // Fungsi menampilkan hasil
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            
            let html = `
                <div class="stats">
                    <div class="stat-item">
                        <div>Total Trades</div>
                        <div class="stat-value">${data.total_trades}</div>
                    </div>
                    <div class="stat-item">
                        <div>Win Rate</div>
                        <div class="stat-value">${data.win_rate}%</div>
                    </div>
                    <div class="stat-item">
                        <div>Total Profit</div>
                        <div class="stat-value">${data.total_profit_pct}%</div>
                    </div>
                    <div class="stat-item">
                        <div>Avg Profit</div>
                        <div class="stat-value">${data.avg_profit_pct}%</div>
                    </div>
                    <div class="stat-item">
                        <div>Profit Factor</div>
                        <div class="stat-value">${data.profit_factor}</div>
                    </div>
                    <div class="stat-item">
                        <div>Max Drawdown</div>
                        <div class="stat-value">${data.max_drawdown_pct}%</div>
                    </div>
                </div>
                <h3 style="margin-top: 20px;">20 Trades Terakhir</h3>
                <table>
                    <tr>
                        <th>Entry Time</th>
                        <th>Direction</th>
                        <th>Entry</th>
                        <th>SL</th>
                        <th>TP</th>
                        <th>Exit Time</th>
                        <th>Result</th>
                        <th>Profit %</th>
                    </tr>
            `;

            data.trades.forEach(t => {
                html += `
                    <tr>
                        <td>${t.entry_time}</td>
                        <td>${t.direction}</td>
                        <td>${t.entry_price.toFixed(5)}</td>
                        <td>${t.sl.toFixed(5)}</td>
                        <td>${t.tp.toFixed(5)}</td>
                        <td>${t.exit_time}</td>
                        <td class="${t.win ? 'win' : 'loss'}">${t.win ? 'WIN' : 'LOSS'}</td>
                        <td>${t.profit_pct.toFixed(2)}%</td>
                    </tr>
                `;
            });

            html += '</table>';
            
            if (data.data_period) {
                html += `<p><small>Data periode: ${data.data_period}</small></p>`;
            }
            
            resultsDiv.innerHTML = html;
            resultsDiv.style.display = 'block';
        }

        // Event listener tombol
        document.getElementById('runBtn').addEventListener('click', runBacktest);
    </script>
</body>
</html>
