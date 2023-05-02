# -*- coding: utf-8 -*-
"""
Created on Tue May  2 19:02:11 2023

@author: Diego
"""

import sys
import matplotlib
import datetime as dt
import yfinance as yf

from LSPair import *

try:

    df = pd.read_parquet(
        path = "agency_df.parquet", 
        engine = "pyarrow")

    end_date = dt.date(year = 2023, month = 4, day = 5)
    start_date = dt.date(year = end_date.year - 6, month = 1, day = 1)
  
except:

    tickers = ["CMBS", "VMBS", "MBB"]
    end_date = dt.date(year = 2023, month = 4, day = 5)
    start_date = dt.date(year = end_date.year - 6, month = 1, day = 1)

    df = (yf.download(
        tickers = tickers,
        start = start_date,
        end = end_date).
        reset_index().
        melt(id_vars = "Date").
        rename(columns = {
            "variable_0": "ticker_type", 
            "variable_1": "ticker"}))

    df.to_parquet(
      path = "agency_df.parquet",
      engine = "pyarrow")
  

df = (df.query(
    "ticker_type == 'Adj Close'").
    drop(columns = ["ticker_type"]).
    pivot(index = "Date", columns = "ticker", values = "value").
    pct_change().
    dropna())

ls_pair = LSPair(
    long_position = df.VMBS,
    short_position = df.CMBS,
    benchmark = df.MBB)

def try_fail(test, name):
    try:
        test
        print("[INFO] Successfully Completed {} Test".format(name))
    except:
        print("[ALERT] Failed to Complete {} Test".format(name))
        sys.quit()
        
def try_stat_regression_test():
    
    try_fail(ls_pair.in_sample_long_lm_res, "In-Sample Long Regression")
    try_fail(ls_pair.out_sample_long_lm_res, "Out-Sample Long Regression")
    try_fail(ls_pair.full_sample_long_lm_res, "Full-Sample Long Regression")
    try_fail(ls_pair.in_sample_short_lm_res, "In-Sample Short Regression")
    try_fail(ls_pair.out_sample_short_lm_res, "Out-Sample Short Regression")
    try_fail(ls_pair.full_sample_short_lm_res, "Full-Sample Short Regression")
    print("[INFO] Completed All Regression Sample Tests")
    
try_stat_regression_test()

try_fail(ls_pair.plot_regress(), "In-Sample Regression")
try_fail(ls_pair.plot_out_regress(), "Out-of-Sample Regression")
try_fail(ls_pair.plot_full_regress(), "Ful-Sample Regression")

try_fail(ls_pair.plot_cum(), "Cumualative Returns with fill")
try_fail(ls_pair.plot_out_sample_cum(), "Cumualative Returns with fill")
try_fail(ls_pair.plot_full_sample_cum(), "Cumualative Returns with fill")

try_fail(ls_pair.plot_cum(fill = False), "In-Sample Cumualative Returns with fill")
try_fail(ls_pair.plot_out_sample_cum(fill = False), "Out-of-Sample Cumualative Returns with fill")
try_fail(ls_pair.plot_full_sample_cum(fill = False), "Full-Sample Cumualative Returns with fill")

try_fail(ls_pair.generate_even_rebal_risk_premia(plot = True), "Even Rebalance plot with fill")
try_fail(ls_pair.generate_even_rebal_risk_premia(plot = True, fill = False), "Even Rebalance plot without fill")

try_fail(ls_pair.rolling_ols(), "Rolling OLS")
try_fail(ls_pair.plot_single_rolling_ols(window = 30), "Rolling OLS Plot with Confidence Interval")
try_fail(ls_pair.plot_single_rolling_ols(window = 30, fill = False), "Rolling OLS Plot with Confidence Interval")
try_fail(ls_pair.plot_single_rolling_ols_comparison(window = 30), "Rolling OLS Plot comparison without cofidence")
try_fail(ls_pair.plot_single_rolling_ols_comparison(window = 30, conf_int = 0.05), "Rolling OLS Plot comparison with Confidence Interval")

try_fail(ls_pair.plot_single_rolling_ols_parameter_comparison(ols_window = 30, corr_window = 30), "Rolling OLS Parameter Comparison")
try_fail(ls_pair.plot_single_rolling_ols_hist(ols_window = 30), "Rolling OLS Plot Histogram")
try_fail(ls_pair.plot_single_rolling_ols_contour(ols_window = 30), "Rolling OLS Plot Histogram")

print("[INFO] Completed All Tests")
sys.quit()