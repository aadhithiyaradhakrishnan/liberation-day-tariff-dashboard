"""
Download large data files from Hugging Face that exceed GitHub's 100MB limit.
Run once after cloning: python download_data.py
"""

import os
import ssl
import urllib.request

# SSL bypass for corporate/university networks with SSL inspection
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

ROOT = os.path.dirname(os.path.abspath(__file__))
HF_REPO = "aditya1240/liberation-day-data"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

# Files to download: (huggingface_path, local_path)
FILES = [
    (
        "data/processed/icio_2022/io_coeff_matrix.npy",
        os.path.join(ROOT, "data", "processed", "icio_2022", "io_coeff_matrix.npy"),
    ),
    (
        "data/processed/icio_2022/io_intermediate_matrix.npy",
        os.path.join(ROOT, "data", "processed", "icio_2022", "io_intermediate_matrix.npy"),
    ),
    (
        "data/code_and_release_data/301 model/D_all_data.zip",
        os.path.join(ROOT, "data", "code_and_release_data", "301 model", "D_all_data.zip"),
    ),
]


def download_file(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading {os.path.basename(dest)}...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(f"\r  {pct:.1f}%", end="", flush=True)

    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl_ctx)
    )
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\r  Done — saved to {dest}")


def main():
    print("Liberation Day — Large Data File Downloader")
    print(f"Source: https://huggingface.co/datasets/{HF_REPO}\n")

    all_present = True
    for hf_path, local_path in FILES:
        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / 1e6
            print(f"  ✅ Already exists ({size_mb:.0f}MB): {os.path.basename(local_path)}")
        else:
            all_present = False
            url = f"{HF_BASE}/{hf_path}"
            try:
                download_file(url, local_path)
            except Exception as e:
                print(f"  ❌ Failed: {e}")

    print()
    if all_present:
        print("All files already present — nothing to download.")
    else:
        print("Download complete. You can now run the dashboard:")
        print("  python -m streamlit run dashboard/app.py --server.headless true")


if __name__ == "__main__":
    main()
