import os
import subprocess
import sys

def main():
    print("🔍 Checking working directory...")
    cwd = os.getcwd()
    print(f"📂 Current directory: {cwd}")

    # Cari requirements.txt
    req_path = os.path.join(cwd, "requirements.txt")

    if not os.path.exists(req_path):
        print("❌ requirements.txt not found in current directory.")
        print("➡️ Please run this first in Colab:")
        print("   %cd forex-ml-broker-ema200")
        sys.exit(1)

    print("✅ Found requirements.txt")
    print("📦 Installing dependencies...")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
        print("🎉 All dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print("❌ Error during pip install")
        print(e)

if __name__ == "__main__":
    main()
