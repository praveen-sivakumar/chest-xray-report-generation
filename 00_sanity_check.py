from pathlib import Path
import pandas as pd

DATA_ROOT = Path("/home/psivakumar/thesis/data/iu_xray")
TRAIN_CSV = DATA_ROOT / "iu_xray_pairs_train.csv"
VAL_CSV   = DATA_ROOT / "iu_xray_pairs_val.csv"
TEST_CSV  = DATA_ROOT / "iu_xray_pairs_test.csv"

def check(csv_path: Path, name: str):
    print(f"\n=== {name} ===")
    print("CSV:", csv_path)
    assert csv_path.exists(), f"Missing: {csv_path}"

    df = pd.read_csv(csv_path)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    assert "image_path" in df.columns, "Expected column: image_path"
    assert "report_text" in df.columns, "Expected column: report_text"

    df = df.dropna(subset=["image_path", "report_text"]).reset_index(drop=True)
    print("Rows after dropna:", len(df))

    # Check file existence for first 5
    for i in range(min(5, len(df))):
        p = Path(df.loc[i, "image_path"])
        print(f"[{i}] image exists={p.exists()}  path={p}")

    # Show one sample text
    print("\nSample report (first 400 chars):")
    print(df.loc[0, "report_text"][:400])

if __name__ == "__main__":
    check(TRAIN_CSV, "TRAIN")
    check(VAL_CSV, "VAL")
    check(TEST_CSV, "TEST")
    print("\n✅ Sanity check passed.")