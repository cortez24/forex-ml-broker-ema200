import importlib

REQUIRED = {
    "pandas": "2.2.2",
    "numpy": "1.26.4",
    "matplotlib": "3.9.2",
    "seaborn": "0.13.2",
    "joblib": "1.3.2",
    "scikit-learn": "1.4.2",
    "xgboost": "2.1.1",
    "yfinance": "0.2.40",
    "streamlit": "1.38.0",
    "plotly": "5.24.1",
    "pyngrok": "7.2.0",
}

def check_packages():
    print("🔍 Checking installed packages...\n")
    for pkg, req_version in REQUIRED.items():
        try:
            module = importlib.import_module(pkg)
            installed = getattr(module, "__version__", "unknown")
            if installed == req_version:
                print(f"✅ {pkg} {installed} (OK)")
            else:
                print(f"⚠️ {pkg} {installed} (expected {req_version})")
        except ImportError:
            print(f"❌ {pkg} not installed!")

if __name__ == "__main__":
    check_packages()
