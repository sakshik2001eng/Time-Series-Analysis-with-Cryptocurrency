"""Small helpers shared by the analysis scripts."""
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"


def save_fig(name: str) -> None:
    """Save the current matplotlib figure to reports/figures/<name>.png (used in the README)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
