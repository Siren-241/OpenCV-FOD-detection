import sys
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "src" / "data"
SNAPSHOT_URL = "https://192.168.1.6:4444/video/snapshot"


def capture(destination):
    response = requests.get(
        SNAPSHOT_URL,
        timeout=15,
        verify=False,
    )
    response.raise_for_status()

    if "image/jpeg" not in response.headers.get("Content-Type", "").lower():
        raise RuntimeError(
            f"Expected JPEG, received: "
            f"{response.headers.get('Content-Type', 'unknown')}"
        )

    destination.write_bytes(response.content)
    print(f"Image saved: {destination}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"baseline", "after"}:
        print("Usage: python src/capture_images.py baseline|after")
        raise SystemExit(2)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / f"{sys.argv[1]}.png"

    capture(destination)


if __name__ == "__main__":
    main()
