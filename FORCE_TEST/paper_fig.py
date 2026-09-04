"""IEEE-style figures. Captions live in VISU/README.md — never on the axes."""

from __future__ import annotations

from pathlib import Path

CMD = "#111111"
ACH = "#0072B2"
PLUS = "#0072B2"
MINUS = "#D55E00"
GUIDE = "#4D4D4D"


def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.15,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "axes.formatter.useoffset": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    return plt


def save(fig, out_dir, name: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=300)
    fig.savefig((out / name).with_suffix(".svg"))


def _hz_tick(value: float, _pos=None) -> str:
    if value <= 0:
        return ""
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    return f"{value:g}"


def log_freq_ticks(ax, *, f_lo: float = 0.2, f_hi: float = 10.0, cutoff_hz: float | None = 8.0) -> None:
    """Log frequency axis with written Hertz, not 10^n."""

    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

    marks = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    if cutoff_hz is not None:
        marks.append(float(cutoff_hz))
    ticks = sorted({m for m in marks if f_lo * 0.9 <= m <= f_hi * 1.05})
    if not ticks:
        ticks = [f_lo, f_hi]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(_hz_tick))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9)))
    ax.xaxis.set_minor_formatter(NullFormatter())


def panel_tag(ax, letter: str) -> None:
    ax.text(
        0.02,
        0.96,
        letter,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
