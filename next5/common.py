from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Competition data is not redistributable: keep it outside the repo.
DATA = ROOT.parent.parent / "data"
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)
