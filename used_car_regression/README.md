# Used-car resale regression

Run with Python 3.11 or newer from this directory.

```sh
python -m pip install -r requirements.txt
python analyze.py
```

The script prints the training fit, coefficient tests, held-out metrics, and auction comparison. It also generates these files in `results/`:

- `test_predictions.png` compares predicted and actual prices for the same 72 test cars.
- `test_predictions.csv` lists the predictions and one-based source row numbers (excluding the header).
- `auction_results.csv` records predicted retail prices, buy/pass decisions, and profit for each of the 40 lots under both rules.

## Method

- The supplied retail file covers 240 sales from the past year, per the assignment. The CSV does not specify calendar dates. Random seed 42 gives 168 training cars and 72 test cars.
- Fit ordinary least squares using only the training cars. The full model uses mileage, age, horsepower, accident history, and previous owners. The reduced model drops the least significant predictor if its training p-value exceeds 0.05. Choose the model with higher training adjusted R². The attached run selected the reduced model.
- Coefficients are changes in predicted price with the other inputs held constant. Mileage is in thousands of miles. Accident is coded as yes = 1, with no = 0 as the reference. The p-values are conventional two-sided OLS t-tests.
- The baseline is the training-set mean. Test R² compares squared errors against the test-set mean, so a baseline based on the training mean can have a slightly negative test R².
- Both rules buy only when predicted retail exceeds the winning auction bid plus $1,500. Later retail prices are used only to score the backtest. Zero-profit buys are not counted as losses.
- The estimated profit difference is for one auction at observed winning bids, compared with the assignment's mean-price baseline. The data do not include the dealer's actual gut-feel bids.

`analysis_results.txt` contains the console output from the supplied data.
