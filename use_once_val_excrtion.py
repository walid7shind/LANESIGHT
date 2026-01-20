import random
from pathlib import Path
from collections import defaultdict

# ---------------- CONFIG ----------------
ROOT = Path(__file__).resolve().parent
LIST_DIR = ROOT / "data" / "list"

SRC_LIST = LIST_DIR / "train_gt - Copy.txt"
TRAIN_OUT = LIST_DIR / "train_clean.txt"
VAL_OUT = LIST_DIR / "val_clean.txt"

VAL_DRIVER_RATIO = 0.15
SEED = 42
# --------------------------------------


def extract_driver(img_path: str) -> str:
    # Robust, OS-independent
    # "/driver_xxx_30frame/..." → "driver_xxx_30frame"
    return img_path.lstrip("/").split("/", 1)[0]


def main():
    assert SRC_LIST.exists(), f"Missing source list: {SRC_LIST}"

    random.seed(SEED)
    by_driver = defaultdict(list)

    with SRC_LIST.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            img_path = line.split()[0]
            driver = extract_driver(img_path)
            by_driver[driver].append(line)

    drivers = sorted(by_driver.keys())
    assert len(drivers) >= 2, "Not enough drivers to split"

    random.shuffle(drivers)

    n_val = max(1, int(len(drivers) * VAL_DRIVER_RATIO))
    val_drivers = set(drivers[:n_val])

    train_lines, val_lines = [], []

    for d in drivers:
        if d in val_drivers:
            val_lines.extend(by_driver[d])
        else:
            train_lines.extend(by_driver[d])

    TRAIN_OUT.write_text("".join(train_lines))
    VAL_OUT.write_text("".join(val_lines))

    print("=== CLEAN SPLIT GENERATED ===")
    print(f"Drivers total : {len(drivers)}")
    print(f"Train drivers : {len(drivers) - len(val_drivers)}")
    print(f"Val drivers   : {len(val_drivers)}")
    print(f"Train frames  : {len(train_lines)}")
    print(f"Val frames    : {len(val_lines)}")
    print("============================")


if __name__ == "__main__":
    main()
