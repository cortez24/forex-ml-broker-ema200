import os
import subprocess
import sys

def find_requirements(start_dir="."):
    """
    Cari file requirements.txt di direktori dan subfolder.
    """
    for root, dirs, files in os.walk(start_dir):
        if "requirements.txt" in files:
            return os.path.join(root, "requirements.txt")
    return None

def main():
    print("🔍 Checking environment...")

    # Cek di direktori saat ini
    req_path = os.path.join(os.getcwd(), "requirements.txt")

    if not os.path.exists(req_path):
        print("⚠️ requirements.txt not found in current directory.")
        print("🔎 Searching in subfolders...")
        req_path = find_requirements(".")
    
    if not req_path:
        print("❌ requirements.txt not found anywhere in repo.")
        print("➡️ Make sure you cloned the full repo and are inside the right folder.")
        sys.exit(1)

    print(f"✅ Found requirements.txt at: {req_path}")
    print("📦 Installing dependencies...")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
        print("🎉 All dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print("❌ Error during pip install")
        print(e)

if __name__ == "__main__":
    main()
