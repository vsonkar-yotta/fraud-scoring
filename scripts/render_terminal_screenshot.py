"""Render terminal-style PNG screenshots from captured command output.

Not part of the production pipeline -- a one-off demo helper for
docs/screenshots/, since this environment has no GUI to capture real
terminal screenshots from.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(title: str, prompt: str, body: str, out_path: str) -> None:
    lines = body.rstrip("\n").split("\n")
    n_lines = len(lines) + 2
    fig_h = max(2.0, 0.28 * n_lines + 0.8)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.015, 0.97, title, transform=ax.transAxes, fontsize=10,
             color="#8ab4f8", family="monospace", va="top", weight="bold")
    y = 0.88
    dy = 1.0 / max(n_lines, 1)
    if prompt:
        ax.text(0.015, y, prompt, transform=ax.transAxes, fontsize=9.5,
                 color="#6ee7b7", family="monospace", va="top")
        y -= dy
    for line in lines:
        ax.text(0.015, y, line, transform=ax.transAxes, fontsize=9.2,
                 color="#e0e0e0", family="monospace", va="top")
        y -= dy

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--body-file", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    render(args.title, args.prompt, Path(args.body_file).read_text(), args.out)
