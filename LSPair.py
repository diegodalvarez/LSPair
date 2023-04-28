import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

class LSPair:

  def _run_lm(self, endog: pd.Series, exog: pd.Series):

    x = sm.add_constant(exog.values)

    lm = (sm.OLS(
        endog = endog.values,
        exog = x).fit())
    lm_res = lm.summary()

    return lm, lm_res

  def __init__(self, long_position: pd.Series, short_position: pd.Series, 
               benchmark: pd.Series, in_sample_ratio = 0.7) -> None:

    # parametrization
    self.long_position = long_position
    self.short_position = short_position
    self.benchmark = benchmark

    self.long_name = self.long_position.name
    self.short_name = self.short_position.name
    self.benchmark_name = self.benchmark.name
    self.in_sample_ratio = in_sample_ratio

    # joining to check for missing values
    self.df_combined = (pd.DataFrame({
      "long_position": self.long_position,
      "short_position": self.short_position,
      "benchmark": self.benchmark}))

    self.df_dropped = self.df_combined.dropna()

    if len(self.df_combined) != len(self.df_dropped):
      print("Some values were dropped")
      self.df_combined = self.df_dropped

    # making in sample and out of sample data
    self.df_count = (self.df_combined.sort_index().assign(
        count = [i + 1 for i in range(len(self.df_combined))]))
    
    self.df_cutoff = int(self.in_sample_ratio * len(self.df_count))

    self.in_sample_df = (self.df_count.query(
        "count <= @self.df_cutoff").
        drop(columns = ["count"]))
    
    self.out_sample_df = (self.df_count.query(
        "count > @self.df_cutoff").
        drop(columns = ["count"]))
    
    self.full_sample_df = self.df_count.drop(columns = ["count"])

    # automatically run the regressions (in_sample, out of sample, full)
    self.in_sample_long_lm, self.in_sample_long_lm_res = self._run_lm(
        self.in_sample_df.long_position, 
        self.in_sample_df.benchmark)
    
    self.out_sample_long_lm, self.out_sample_long_lm_res = self._run_lm(
        self.out_sample_df.long_position,
        self.out_sample_df.benchmark)
    
    self.full_sample_long_lm, self.full_sample_long_lm_res = self._run_lm(
        self.full_sample_df.long_position,
        self.full_sample_df.benchmark)
    
    self.in_sample_short_lm, self.in_sample_short_lm_res = self._run_lm(
        self.in_sample_df.short_position,
        self.in_sample_df.benchmark)
    
    self.out_sample_short_lm, self.out_sample_short_lm_res = self._run_lm(
        self.out_sample_df.short_position,
        self.out_sample_df.benchmark)
    
    self.full_sample_short_lm, self.full_sample_short_lm_res = self._run_lm(
        self.full_sample_df.short_position,
        self.full_sample_df.benchmark)

  # plotting the regressions
  def plot_regress(
      self,
      figsize = (20,6)):

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = figsize)

    # plot full sample
    (self.in_sample_df[
        ["long_position", "benchmark"]].
        assign(
            long_position = lambda x: x.long_position * 1, # need to fix to 100
            benchmark = lambda x: x.benchmark * 1).
        rename(columns = {
            "long_position": self.long_name,
            "benchmark": self.benchmark_name}).
        plot(
            ax = axes[0], kind = "scatter", 
            x = self.benchmark.name, y = self.long_name))

    (self.in_sample_df[
        ["short_position", "benchmark"]].
        assign(
            short_position = lambda x: x.short_position * 1, # fix to 100
            benchmark = lambda x: x.benchmark * 1).
        rename(columns = {
            "short_position": self.short_name,
            "benchmark": self.benchmark_name}).
        plot(
            ax = axes[1], kind = "scatter",
            x = self.benchmark_name, y = self.short_name))
    
    # getting the lines
    long_x_min = self.in_sample_df.benchmark.min()
    long_x_max = self.in_sample_df.benchmark.max()
    x_values = np.linspace(start = long_x_min, stop = long_x_max, num = 100)

    y_long_values = (x_values * self.in_sample_long_lm.params[1]) 
    y_long_values = y_long_values + (self.in_sample_long_lm.params[0]) 
    y_long_r_squared = self.in_sample_long_lm.rsquared
    axes[0].plot(
        x_values, y_long_values, 
        color = "r", label = "R Squared: {}".format(round(y_long_r_squared,2)))
    axes[0].legend()
    axes[0].set_title("Alpha: {} Beta {}".format(
        round(self.in_sample_long_lm.params[0], 2), 
        round(self.in_sample_long_lm.params[1], 2)))

    y_short_values = (x_values * self.in_sample_short_lm.params[1])
    y_short_values = y_short_values + (self.in_sample_short_lm.params[0])
    y_short_r_squared = self.in_sample_short_lm.rsquared
    axes[1].plot(
        x_values, y_short_values, 
        color = "r", label = "R Squared: {}".format(round(y_short_r_squared,2)))
    axes[1].legend()
    axes[1].set_title("Alpha: {} Beta {}".format(
        round(self.in_sample_short_lm.params[0], 2), 
        round(self.in_sample_short_lm.params[1], 2)))
    
    fig.suptitle("In-Sample ({}%) CAPM Regression from {} to {}".format(
        self.in_sample_ratio * 100, 
        self.in_sample_df.index.min().date(), 
        self.in_sample_df.index.max().date()))
    
    plt.tight_layout()

  # plotting the regressions
  def plot_out_regress(
      self,
      figsize = (20,6)):

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = figsize)

    # plot full sample
    (self.out_sample_df[
        ["long_position", "benchmark"]].
        assign(
            long_position = lambda x: x.long_position * 1, # need to fix to 100
            benchmark = lambda x: x.benchmark * 1).
        rename(columns = {
            "long_position": self.long_name,
            "benchmark": self.benchmark_name}).
        plot(
            ax = axes[0], kind = "scatter", 
            x = self.benchmark.name, y = self.long_name))

    (self.out_sample_df[
        ["short_position", "benchmark"]].
        assign(
            short_position = lambda x: x.short_position * 1, # fix to 100
            benchmark = lambda x: x.benchmark * 1).
        rename(columns = {
            "short_position": self.short_name,
            "benchmark": self.benchmark_name}).
        plot(
            ax = axes[1], kind = "scatter",
            x = self.benchmark_name, y = self.short_name))
    
    # getting the lines
    long_x_min = self.out_sample_df.benchmark.min()
    long_x_max = self.out_sample_df.benchmark.max()
    x_values = np.linspace(start = long_x_min, stop = long_x_max, num = 100)

    y_long_values = (x_values * self.out_sample_long_lm.params[1]) 
    y_long_values = y_long_values + (self.out_sample_long_lm.params[0]) 
    y_long_r_squared = self.out_sample_long_lm.rsquared
    axes[0].plot(
        x_values, y_long_values, 
        color = "r", label = "R Squared: {}".format(round(y_long_r_squared,2)))
    axes[0].legend()
    axes[0].set_title("Alpha: {} Beta {}".format(
        round(self.out_sample_long_lm.params[0], 2), 
        round(self.out_sample_long_lm.params[1], 2)))

    y_short_values = (x_values * self.out_sample_short_lm.params[1])
    y_short_values = y_short_values + (self.out_sample_short_lm.params[0])
    y_short_r_squared = self.out_sample_short_lm.rsquared
    axes[1].plot(
        x_values, y_short_values, 
        color = "r", label = "R Squared: {}".format(round(y_short_r_squared,2)))
    axes[1].legend()
    axes[1].set_title("Alpha: {} Beta {}".format(
        round(self.out_sample_short_lm.params[0], 2), 
        round(self.out_sample_short_lm.params[1], 2)))
    
    fig.suptitle("Out-of-Sample ({}%) CAPM Regression from {} to {}".format(
        round((1 - self.in_sample_ratio) * 100, 2), 
        self.out_sample_df.index.min().date(), 
        self.out_sample_df.index.max().date()))
    
    plt.tight_layout()

  # plotting the regressions
  def plot_full_regress(
      self,
      figsize = (20,6)):

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = figsize)

    # plot full sample
    (self.full_sample_df[
        ["long_position", "benchmark"]].
        assign(
            long_position = lambda x: x.long_position * 1, # need to fix to 100
            benchmark = lambda x: x.benchmark * 1).
        rename(columns = {
            "long_position": self.long_name,
            "benchmark": self.benchmark_name}).
        plot(
            ax = axes[0], kind = "scatter", 
            x = self.benchmark.name, y = self.long_name))

    (self.full_sample_df[
        ["short_position", "benchmark"]].
        assign(
            short_position = lambda x: x.short_position * 1, # fix to 100
            benchmark = lambda x: x.benchmark * 1).
        rename(columns = {
            "short_position": self.short_name,
            "benchmark": self.benchmark_name}).
        plot(
            ax = axes[1], kind = "scatter",
            x = self.benchmark_name, y = self.short_name))
    
    # getting the lines
    long_x_min = self.full_sample_df.benchmark.min()
    long_x_max = self.full_sample_df.benchmark.max()
    x_values = np.linspace(start = long_x_min, stop = long_x_max, num = 100)

    y_long_values = (x_values * self.full_sample_long_lm.params[1]) 
    y_long_values = y_long_values + (self.full_sample_long_lm.params[0]) 
    y_long_r_squared = self.full_sample_long_lm.rsquared
    axes[0].plot(
        x_values, y_long_values, 
        color = "r", label = "R Squared: {}".format(round(y_long_r_squared,2)))
    axes[0].legend()
    axes[0].set_title("Alpha: {} Beta {}".format(
        round(self.full_sample_long_lm.params[0], 2), 
        round(self.full_sample_long_lm.params[1], 2)))

    y_short_values = (x_values * self.full_sample_short_lm.params[1])
    y_short_values = y_short_values + (self.full_sample_short_lm.params[0])
    y_short_r_squared = self.full_sample_short_lm.rsquared
    axes[1].plot(
        x_values, y_short_values, 
        color = "r", label = "R Squared: {}".format(round(y_short_r_squared,2)))
    axes[1].legend()
    axes[1].set_title("Alpha: {} Beta {}".format(
        round(self.full_sample_short_lm.params[0], 2), 
        round(self.full_sample_short_lm.params[1], 2)))
    
    fig.suptitle("Full Sample CAPM Regression from {} to {}".format(
        self.full_sample_df.index.min().date(), 
        self.full_sample_df.index.max().date()))
    
    plt.tight_layout()

  # for the groupby in _plot_cum
  def _get_cum_rtns(self, df):

    return(df.sort_values(
        "Date").
        assign(cum_rtns = lambda x: (np.cumprod(1 + x.value) - 1) * 100))

  # underlying function parametrized below
  def _get_cum(self, df: pd.DataFrame):

    id_vars_name = df.index.name

    df_tmp = (df.reset_index().melt(
        id_vars = id_vars_name).
        groupby("variable", group_keys=False).
        apply(self._get_cum_rtns)
        [[id_vars_name, "variable", "cum_rtns"]].
        pivot(index = id_vars_name, columns = "variable", values = "cum_rtns"))
    
    return df_tmp

  # function for parametrization below
  def _plot_cum(self, df, fill: bool, figsize: tuple):
    
    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = figsize)

    df = self._get_cum(df)

    (df[
        ["long_position", "benchmark"]].
        rename(columns = {
            "long_position": "Long: {}".format(self.long_name),
            "benchmark": "Benchmark: {}".format(self.benchmark_name)}).
        plot(
            ax = axes[0,0], ylabel = "Cumulative Return (%)",
            sharex = True,
            title = "Long Position Comparison",
            color = ["mediumblue", "black"]))
    
    (df[
        ["short_position", "benchmark"]].
        rename(columns = {
            "short_position": "Short: {}".format(self.short_name),
            "benchmark": "Benchmark: {}".format(self.benchmark_name)}).
        plot(
            ax = axes[0,1], ylabel = "Cumulative Return (%)",
            sharex = True,
            title = "Short Position Comparison",
            color = ["cornflowerblue", "black"]))
    
    (df[
        ["short_position", "benchmark", "long_position"]].
        rename(columns = {
            "short_position": "Short: {}".format(self.short_name),
            "long_position": "Long: {}".format(self.long_name),
            "benchmark": "Benchmark: {}".format(self.benchmark_name)}).
        plot(
            ax = axes[1,0], ylabel = "Cumulative Return (%)",
            sharex = True,
            title = "Overall Comparison",
            color = ["cornflowerblue", "black", "mediumblue"]))
    
    (df[
        ["short_position", "long_position"]].
        rename(columns = {
            "short_position": "Short: {}".format(self.short_name),
            "long_position": "Long: {}".format(self.long_name)}).
        plot(
            ax = axes[1,1], ylabel = "Cumulative Return (%)",
            sharex = True,
            title = "Long / Short Comparison",
            color = ["cornflowerblue", "mediumblue"]))

    if fill == True:

      axes[0,0].fill_between(
          df.index, df.long_position, df.benchmark,
          where = df.long_position > df.benchmark,
          facecolor = "Green",
          alpha = 0.3)
    
      axes[0,0].fill_between(
          df.index, df.long_position, df.benchmark,
          where = df.long_position < df.benchmark,
          facecolor = "Red",
          alpha = 0.3)
      
      axes[0,1].fill_between(
          df.index, df.short_position, df.benchmark,
          where = df.short_position > df.benchmark,
          facecolor = "Red",
          alpha = 0.3)
      
      axes[0,1].fill_between(
          df.index, df.short_position, df.benchmark,
          where = df.short_position < df.benchmark,
          facecolor = "Green",
          alpha = 0.3)
      
      axes[1,1].fill_between(
          df.index, df.long_position, df.short_position,
          where = df.long_position > df.short_position,
          facecolor = "Green",
          alpha = 0.3)
      
      axes[1,1].fill_between(
          df.index, df.long_position, df.short_position,
          where = df.long_position < df.short_position,
          facecolor = "Red",
          alpha = 0.3)
      
    fig.show()
    return fig

  def plot_cum(self, fill = True, figsize = (20,6)):

    fig = self._plot_cum(self.in_sample_df, fill = True, figsize = (20,6))
    plt.suptitle("Long Short Comparison (In-Sample {}%) from {} to {}".format(
      (self.in_sample_ratio * 100),
      self.in_sample_df.index.min().date(), 
      self.in_sample_df.index.max().date()))
    plt.tight_layout()

  def plot_out_sample_cum(self, fill = True, figsize = (20,6)):

    fig = self._plot_cum(self.out_sample_df, fill = True, figsize = (20,6))
    plt.suptitle("Long Short Comparison (Out-Sample {}%) from {} to {}".format(
        (round(1 - self.in_sample_ratio, 2)) * 100,
        self.out_sample_df.index.min().date(),
        self.out_sample_df.index.max().date()))
    plt.tight_layout()

  def plot_full_sample_cum(self, fill = True, figsize = (20,6)):

    fig = self._plot_cum(self.full_sample_df, fill = True, figsize = (20,6))
    plt.suptitle("Long Short Comparison (Full Dataset) from {} to {}".format(
        self.full_sample_df.index.min().date(),
        self.full_sample_df.index.max().date()))
    plt.tight_layout()

  def _generate_even_rebal_risk_premia(
      self,
      df):
    
    df_ls = (df[
        ["long_position", "short_position"]].
        assign(
            long_position = lambda x: (0.5 * x.long_position),
            short_position = lambda x: (-0.5 * x.short_position),
            port_rtn = lambda x: x.long_position + x.short_position,
            cum_port = lambda x: (np.cumprod(1 + x.port_rtn) - 1) * 100))
    
    df_long_premia = (df[
        ["long_position", "benchmark"]].
        assign(
            long_position = lambda x: (0.5 * x.long_position),
            benchmark = lambda x: (-0.5 * x.benchmark),
            port_rtn = lambda x: x.long_position + x.benchmark,
            cum_port = lambda x: (np.cumprod(1 + x.port_rtn) - 1) * 100))
    
    df_short_premia = (df[
        ["short_position", "benchmark"]].
        assign(
            short_position = lambda x: (-0.5 * x.short_position),
            benchmark = lambda x: (0.5 * x.benchmark),
            port_rtn = lambda x: x.short_position + x.benchmark,
            cum_port = lambda x: (np.cumprod(1 + x.port_rtn) - 1) * 100))
    
    return [df_ls, df_long_premia, df_short_premia]

  def _get_fill(self, axes, df):

    axes.fill_between(
        df.index, df.cum_port, 0,
        where = df.cum_port > 0,
        facecolor = "green",
        alpha = 0.3)
    
    axes.fill_between(
        df.index, df.cum_port, 0,
        where = df.cum_port < 0,
        facecolor = "red",
        alpha = 0.3)

  def generate_even_rebal_risk_premia(
      self, 
      plot = True, 
      fill = True,
      figsize = (20,12)) -> pd.DataFrame:

    fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = figsize)
    id_vars_name = self.in_sample_df.index.name

    # get premias and then repack them
    df_in_sample_premias = self._generate_even_rebal_risk_premia(
        self.in_sample_df)
    
    df_ls_in_sample = df_in_sample_premias[0]
    df_long_premia_in_sample = df_in_sample_premias[1]
    df_short_premia_in_sample = df_in_sample_premias[2]
    
    df_out_sample_premias = self._generate_even_rebal_risk_premia(
        self.out_sample_df)
    
    df_ls_out_sample = df_out_sample_premias[0]
    df_long_premia_out_sample = df_out_sample_premias[1]
    df_short_premia_out_sample = df_out_sample_premias[2]

    df_full_sample_premias = self._generate_even_rebal_risk_premia(
        self.full_sample_df)
    
    df_ls_full_sample = df_full_sample_premias[0]
    df_long_premia_full_sample = df_full_sample_premias[1]
    df_short_premia_full_sample = df_full_sample_premias[2]

    ls_premias = [
        df_ls_in_sample,
        df_ls_out_sample,
        df_ls_full_sample]

    long_premias = [
        df_long_premia_in_sample,
        df_long_premia_out_sample,
        df_long_premia_full_sample]

    short_premias = [
        df_short_premia_in_sample,
        df_short_premia_out_sample,
        df_short_premia_full_sample]

    if plot == True:

      for i, j in enumerate(zip(
          long_premias, short_premias, ls_premias)):
        
        title1 = "Long: {} Short: {} Long Premia".format(
            self.long_name, self.benchmark_name)
        
        title2 = "Long: {} Short: {} Short Premia".format(
            self.benchmark_name, self.short_name)
        
        title3 = "Long: {} Short: {} L/S Premia".format(
            self.long_name, self.short_name)

        (j[0][
            ["cum_port"]].
            plot(
                ax = axes[1,i], legend = False,
                ylabel = "Cumulative Return (%)",
                title = title1))
        
        (j[1][
            ["cum_port"]].
            plot(
                ax = axes[2,i], legend = False,
                ylabel = "Cumulative Return (%)",
                title = title2))
        
        if i == 0:
          title_header = "In Sample ({}%)\n".format(
              self.in_sample_ratio * 100)
          
        if i == 1:
          title_header = "Out of Sample ({}%)\n".format(
              round(1 - self.in_sample_ratio, 2) * 100)
        
        if i == 2:
          title_header = "Full Sample \n"

        (j[2][
            ["cum_port"]].
            plot(
                ax = axes[0,i], legend = False,
                ylabel = "Cumulative Return (%)",
                title = title_header + title3))
        
        if fill == True:

          self._get_fill(axes[1,i], j[0])
          self._get_fill(axes[2,i], j[1])
          self._get_fill(axes[0,i], j[2])
    
    '''
      
      if fill == True:

        axes[0,0].fill_between(
            df_ls.index, df_ls.cum_port, 0, 
            where = df_ls.cum_port < 0,
            facecolor = "red",
            alpha = 0.3)

    df_out = (df_ls[
        ["port_rtn", "cum_port"]].
        reset_index().
        rename(columns = {
            "port_rtn": "long_short_rtn",
            "cum_port": "long_short_cum_rtn"}).
        merge(
            (df_long_premia[
                ["port_rtn", "cum_port"]].
                reset_index().
                rename(columns = {
                    "port_rtn": "long_premia_rtn",
                    "cum_port": "long_premia_cum_rtn"})),
              how = "inner",
              on = [id_vars_name]).
        merge(
            (df_short_premia[
                ["port_rtn", "cum_port"]].
                reset_index().
                rename(columns = {
                    "port_rtn": "short_premia_rtn",
                    "cum_port": "short_premia_cum_port"})),
              how = "inner",
              on = [id_vars_name]).
        melt(id_vars = id_vars_name))
    '''
    plt.tight_layout()

    return pd.DataFrame