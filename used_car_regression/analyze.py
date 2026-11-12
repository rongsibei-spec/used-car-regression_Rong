"""Fit resale-price models and backtest the dealer's auction rule."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
FEATURES = ["Mileage_K", "Age_Years", "Horsepower", "Accident", "Num_Owners"]
PREP_COST = 1500
SEED = 42


def design(frame, features):
    """Add an intercept and encode Accident=yes as 1 (no as 0)."""
    predictors = frame[features].copy()
    if "Accident" in features:
        predictors["Accident"] = predictors["Accident"].map({"yes": 1, "no": 0})
    if predictors.isna().any().any():
        raise ValueError("Missing feature or unknown accident status.")
    return np.column_stack([np.ones(len(frame)), predictors.to_numpy(dtype=float)])


def fit_ols(train, features):
    """Return coefficients, two-sided t-test p-values, and adjusted training R²."""
    x = design(train, features)
    y = train["Sale_Price"].to_numpy()
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    n, p = x.shape
    if rank != p or n <= p:
        raise ValueError("OLS requires a full-rank design and residual degrees of freedom.")
    prediction = x @ coefficients
    residual_variance = np.sum((y - prediction) ** 2) / (n - p)
    _, upper = np.linalg.qr(x)
    inverse_upper = np.linalg.solve(upper, np.eye(p))
    standard_errors = np.sqrt(residual_variance * np.sum(inverse_upper ** 2, axis=1))
    p_values = 2 * t.sf(np.abs(coefficients / standard_errors), df=n - p)
    adjusted_r2 = 1 - (1 - r2_score(y, prediction)) * (n - 1) / (n - p)
    return coefficients, p_values, adjusted_r2


def buy_or_pass(car, auction_price, predict_price):
    """Return buy only if estimated retail price exceeds the all-in cost."""
    return "buy" if predict_price(car) > auction_price + PREP_COST else "pass"


def evaluate_auction(lots, predict_price):
    decisions = lots.apply(
        lambda car: buy_or_pass(car, car["Auction_Price"], predict_price), axis=1
    )
    profit = lots["Later_Sale_Price"] - lots["Auction_Price"] - PREP_COST
    bought = decisions.eq("buy")
    summary = {
        "Cars bought": int(bought.sum()),
        "Money-losing buys": int((bought & profit.lt(0)).sum()),
        "Total profit ($)": int(profit[bought].sum()),
    }
    return decisions, profit.where(bought, 0), summary


def plot_predictions(results, output):
    """Plot both models on identical dollar axes and include the perfect-fit line."""
    actual = results["Actual"]
    low = 5000 * np.floor(results.min().min() / 5000)
    high = 5000 * np.ceil(results.max().max() / 5000)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True)
    for ax, model, color in zip(axes, ["Baseline", "v1"], ["#64748b", "#086f72"]):
        ax.plot([low, high], [low, high], "--", color="#8a939e", linewidth=1,
                label="Perfect prediction")
        ax.scatter(actual, results[model], color=color, s=22, alpha=0.75)
        ax.set(xlim=(low, high), ylim=(low, high), title=model,
               xlabel="Actual sale price")
        ax.set_aspect("equal")
        ax.set_xticks(np.linspace(low, high, 4))
        ax.set_yticks(np.linspace(low, high, 4))
        ax.xaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
        ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
        ax.grid(alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Predicted sale price")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "test_predictions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    sales = pd.read_csv(ROOT / "used_car_sales.csv")
    lots = pd.read_csv(ROOT / "auction_lots.csv")
    train, test = train_test_split(sales, test_size=0.30, random_state=SEED)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    print(f"Retail sales: {len(sales)} | train: {len(train)} | test: {len(test)}"
          f" | auction lots: {len(lots)} | split seed: {SEED} | prep: ${PREP_COST:,}")

    full = fit_ols(train, FEATURES)
    # Consider one reduced model by dropping the least significant feature
    # if its training p-value exceeds 0.05. Compare training adjusted R² only.
    worst_index = int(np.argmax(full[1][1:]))
    candidates = [("Full", FEATURES, full)]
    if full[1][worst_index + 1] > 0.05:
        reduced_features = [name for name in FEATURES if name != FEATURES[worst_index]]
        candidates.append(("Reduced", reduced_features, fit_ols(train, reduced_features)))
    for name, features, (coef, pvals, adjusted_r2) in candidates:
        print(f"\n{name} model | training adjusted R²: {adjusted_r2:.6f}")
        print(pd.DataFrame({"Coefficient": coef, "p-value": pvals},
                           index=["Intercept", *features]).to_string(float_format="%.6g"))
    chosen, features, (coef, _, _) = max(candidates, key=lambda item: item[2][2])
    print(f"\nv1 is the {chosen.lower()} model | features: {', '.join(features)}")
    average = train["Sale_Price"].mean()
    print(f"Baseline retail prediction: ${average:,.2f}")
    results = pd.DataFrame({
        "Actual": test["Sale_Price"],
        "Baseline": average,
        "v1": design(test, features) @ coef,
    })
    results.index = results.index + 1
    results.to_csv(output / "test_predictions.csv", index_label="Sales data row")
    for name in ["Baseline", "v1"]:
        r2 = r2_score(results["Actual"], results[name])
        rmse = np.sqrt(mean_squared_error(results["Actual"], results[name]))
        print(f"{name} | test R²: {r2:.6f} | test RMSE: ${rmse:,.2f}")
    plot_predictions(results, output)

    def predict_v1(car):
        return (design(car.to_frame().T, features) @ coef).item()

    auction_results = lots.copy()
    summaries = {}
    for name, predictor in [("Baseline", lambda car: average), ("v1", predict_v1)]:
        decisions, profit, summaries[name] = evaluate_auction(lots, predictor)
        auction_results[f"{name} predicted retail"] = lots.apply(predictor, axis=1)
        auction_results[f"{name} decision"] = decisions
        auction_results[f"{name} profit"] = profit
    auction_results.to_csv(output / "auction_results.csv", index=False)
    print("\nAuction backtest")
    print(pd.DataFrame(summaries).to_string())
    delta = summaries["v1"]["Total profit ($)"] - summaries["Baseline"]["Total profit ($)"]
    print(f"Incremental profit (v1 minus baseline): ${delta:,.0f}")


if __name__ == "__main__":
    main()
