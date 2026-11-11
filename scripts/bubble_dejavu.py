"""
Bubble Deja Vu: which dot-com stock is today's AI stock "rhyming" with?

For each AI stock, take its last N weeks of normalised price and slide it along
every 1995-2003 dot-com chart. Dynamic Time Warping (DTW) finds the closest
historical window, then we show what that analog did NEXT.

Output: a chart per AI stock with today's path overlaid on its best dot-com
match, plus the analog's "next 26 weeks" as a dashed ghost line.

    pip install yfinance numpy pandas matplotlib
    python scripts/bubble_dejavu.py

Not investment advice. Pattern matching on prices is a conversation starter,
not a forecast. That's the point of the post.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

AI = ["NVDA", "AMD", "AVGO", "MSFT", "META", "ORCL", "TSM", "PLTR"]
DOTCOM = ["CSCO", "INTC", "ORCL", "QCOM", "MSFT", "AMZN", "EBAY", "JNPR", "SUNW", "^IXIC"]
WINDOW = 104   # weeks of recent history to match (2 years)
AHEAD = 26     # weeks of "what happened next" to show
OUT = "dejavu.png"


def weekly(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, interval="1wk", auto_adjust=True, progress=False)["Close"]
    return df.dropna(axis=1, how="all")


def znorm(x):
    x = np.log(np.asarray(x, dtype=float))
    return (x - x.mean()) / (x.std() + 1e-9)


def dtw(a, b, band=8):
    """DTW distance with a Sakoe-Chiba band (keeps the warp honest)."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(m, i + band)
        for j in range(lo, hi + 1):
            c = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return np.sqrt(D[n, m] / n)  # RMS distance per step


def best_analog(recent, history):
    q = znorm(recent)
    best = (np.inf, None, None)
    for tkr in history.columns:
        s = history[tkr].dropna()
        if len(s) < WINDOW + AHEAD:
            continue
        for start in range(0, len(s) - WINDOW - AHEAD, 2):
            w = s.iloc[start:start + WINDOW]
            d = dtw(q, znorm(w))
            if d < best[0]:
                best = (d, tkr, start)
    return best


def main():
    now = weekly(AI, (pd.Timestamp.today() - pd.Timedelta(weeks=WINDOW + 4)).strftime("%Y-%m-%d"), None)
    then = weekly(DOTCOM, "1995-01-01", "2003-12-31")
    cols = [c for c in AI if c in now.columns]
    fig, axes = plt.subplots(2, (len(cols) + 1) // 2, figsize=(4.2 * ((len(cols) + 1) // 2), 7.5), squeeze=False)
    rows = []
    for ax, tkr in zip(axes.flat, cols):
        recent = now[tkr].dropna().iloc[-WINDOW:]
        d, match, start = best_analog(recent, then)
        s = then[match].dropna()
        hist = s.iloc[start:start + WINDOW + AHEAD]
        scale = recent.iloc[-1] / hist.iloc[WINDOW - 1]
        nxt = hist.iloc[WINDOW - 1:] * scale
        ret = nxt.iloc[-1] / nxt.iloc[0] - 1
        rows.append((tkr, match, hist.index[0].date(), hist.index[WINDOW - 1].date(), round(d, 3), f"{ret:+.0%}"))
        x = np.arange(WINDOW)
        ax.plot(x, recent.values, lw=2.2, label=f"{tkr} (last {WINDOW}w)")
        ax.plot(x, (hist.iloc[:WINDOW] * scale).values, lw=1.2, alpha=.7, label=f"{match} {hist.index[0]:%b %Y}")
        ax.plot(np.arange(WINDOW - 1, WINDOW + AHEAD), nxt.values, ls="--", lw=1.4, alpha=.8, label=f"what {match} did next: {ret:+.0%}")
        ax.set_title(f"{tkr} rhymes with {match}", fontsize=11)
        ax.legend(fontsize=7, frameon=False)
        ax.set_xticks([])
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    fig.suptitle("Bubble Deja Vu: today's AI stocks vs their closest dot-com twin (DTW)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(pd.DataFrame(rows, columns=["ai_stock", "dotcom_twin", "twin_from", "twin_to", "dtw_dist", "twin_next_26w"]).to_string(index=False))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
