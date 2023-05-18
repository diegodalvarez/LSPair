import numpy as np
import pandas as pd
import seaborn as sns
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
    def plot_out_regress(self, figsize = (20,6)):

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
    def plot_full_regress(self, figsize = (20,6)):

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

        fig = self._plot_cum(self.in_sample_df, fill = fill, figsize = (20,6))
        plt.suptitle("Long Short Comparison (In-Sample {}%) from {} to {}".format(
            (self.in_sample_ratio * 100),
            self.in_sample_df.index.min().date(), 
            self.in_sample_df.index.max().date()))
        plt.tight_layout()

    def plot_out_sample_cum(self, fill = True, figsize = (20,6)):

        fig = self._plot_cum(self.out_sample_df, fill = fill, figsize = (20,6))
        plt.suptitle("Long Short Comparison (Out-Sample {}%) from {} to {}".format(
            (round(1 - self.in_sample_ratio, 2)) * 100,
            self.out_sample_df.index.min().date(),
            self.out_sample_df.index.max().date()))
        plt.tight_layout()

    def plot_full_sample_cum(self, fill = True, figsize = (20,6)):

        fig = self._plot_cum(self.full_sample_df, fill = fill, figsize = (20,6))
        plt.suptitle("Long Short Comparison (Full Dataset) from {} to {}".format(
            self.full_sample_df.index.min().date(),
            self.full_sample_df.index.max().date()))
        plt.tight_layout()

    def _generate_even_rebal_risk_premia(self, df):

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

    def generate_even_rebal_risk_premia(self, plot = False, fill = True, figsize = (20,12)) -> pd.DataFrame:
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
            
            fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = figsize)
            id_vars_name = self.in_sample_df.index.name

            for i, j in enumerate(zip(long_premias, short_premias, ls_premias)):
            
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

            
            plt.tight_layout()

        df_out = pd.DataFrame()

        # use the fact that we ordered the dfs above so the counter will keep track
        counter_dict = {
            0: "in_sample",
            1: "out_sample",
            2: "full_sample"}

        for counter, dfs, in enumerate(zip(ls_premias, long_premias, short_premias)):

            name = counter_dict[counter]
            
            # we packed it knowing 0: ls_premia, 1: long_premia, 2: short_premia

            ls_df_tmp = (dfs[0].reset_index().melt(
                id_vars = "Date").
                assign(premia = "ls_premia")) 
            
            long_df_tmp = (dfs[1].reset_index().melt(
                id_vars = "Date").
                assign(premia = "long_premia"))
            
            short_df_tmp = (dfs[2].reset_index().melt(
                id_vars = "Date").
                assign(premia = "short_premia"))
            
            df_tmp_combined = (pd.concat(
                [ls_df_tmp, long_df_tmp, short_df_tmp]).
                rename(columns = {
                    "variable": "position",
                    "value": "rtn"}).
                assign(
                    sample_group = counter_dict[counter],
                    pivot_column = lambda x: x.position + "_" + x.premia + "_" + x.sample_group))
            
            df_out = pd.concat([df_out, df_tmp_combined])

        return df_out

    # function for parametrization 
    def _rolling_ols(
        self, df_endog: pd.Series, df_exog: pd.Series,
        lookback_windows: list, conf_int, verbose) -> pd.DataFrame:

        df_out = pd.DataFrame()
        id_vars_name = df_endog.index.name
        for lookback_window in lookback_windows:

            if verbose == True: print("Working on", lookback_window)
            try:

                rolling_ols = (RollingOLS(
                    endog = df_endog,
                    exog = sm.add_constant(df_exog),
                    window = lookback_window).
                    fit())
                
                rolling_ols_params = (rolling_ols.params.dropna().rename(
                    columns = {
                        "const": "alpha",
                        df_endog.name: "beta"}).
                    reset_index().
                    melt(id_vars = id_vars_name).
                    rename(columns = {
                        "variable": "parameter"}).
                    assign(
                          variable = "value",
                          lookback_window = lookback_window))
                
                rolling_ci = (rolling_ols.conf_int(
                    alpha = conf_int, cols = None).
                    dropna().
                    reset_index().
                    melt(id_vars = id_vars_name).
                    rename(columns = {
                        "variable_0": "parameter",
                        "variable_1": "variable"}).
                    replace({
                        "const": "alpha",
                        df_endog.name: "beta",
                        "upper": "upper_{}_conf".format(str(1 - conf_int/2)),
                        "lower": "lower_{}_conf".format(str(conf_int / 2))}).
                    assign(lookback_window = lookback_window))
                
                df_out = (pd.concat([df_out, rolling_ols_params, rolling_ci]))

            except:
                print("There was a problem with", lookback_window)

        if verbose == True: print(" ")  
        return df_out

    # for all the periods
    def rolling_ols(
        self, lookback_windows = [30, 60, 90, 252, 252 * 2, 252 * 3, 252 * 5],
        conf_int = 0.05,
        verbose = True):

        df_out = pd.DataFrame()

        if verbose == True:
            print("Working on In-Sample Long Rolling OLS")

        in_sample_long_ols = (self._rolling_ols(
            df_endog = self.in_sample_df.long_position,
            df_exog = self.in_sample_df.benchmark,
            lookback_windows = lookback_windows,
            conf_int = conf_int,
            verbose = verbose).
            assign(
                sample_group = "in_sample",
                position = "long",
                ticker = self.long_name,
                benchmark_ticker = self.benchmark_name))
        
        if verbose == True:
            print("Working on Out-of-Sample Long Rolling OLS")

        out_sample_long_ols = (self._rolling_ols(
            df_endog = self.out_sample_df.long_position,
            df_exog = self.out_sample_df.benchmark,
            lookback_windows = lookback_windows,
            conf_int = conf_int,
            verbose = verbose).
            assign(
                sample_group = "out_sample",
                position = "long",
                ticker = self.long_name,
                benchmark_ticker = self.benchmark_name))
        
        if verbose == True:
            print("Working on Full-Sample Long Rolling OLS")

        full_sample_long_ols = (self._rolling_ols(
            df_endog = self.full_sample_df.long_position,
            df_exog = self.full_sample_df.benchmark,
            lookback_windows = lookback_windows,
            conf_int = conf_int,
            verbose = verbose).
            assign(
                sample_group = "full_sample",
                position = "long",
                ticker = self.long_name,
                benchmark_ticker = self.benchmark_name))
        
        if verbose == True:
            print("Working on In-Sample Short Rolling OLS")

        in_sample_short_ols = (self._rolling_ols(
            df_endog = self.in_sample_df.short_position,
            df_exog = self.in_sample_df.benchmark,
            lookback_windows = lookback_windows,
            conf_int = conf_int,
            verbose = verbose).
            assign(
                sample_group = "in_sample",
                position = "short",
                ticker = self.short_name,
                benchmark_ticker = self.benchmark_name))
        
        if verbose == True:
            print("Working on Out-of-Sample short Rolling OLS")

        out_sample_short_ols = (self._rolling_ols(
            df_endog = self.out_sample_df.short_position,
            df_exog = self.out_sample_df.benchmark,
            lookback_windows = lookback_windows,
            conf_int = conf_int,
            verbose = verbose).
            assign(
                sample_group = "out_sample",
                position = "short",
                ticker = self.short_name,
                benchmark_ticker = self.benchmark_name))
        
        if verbose == True:
            print("Working on Full-Sample short Rolling OLS")

        full_sample_short_ols = (self._rolling_ols(
            df_endog = self.full_sample_df.short_position,
            df_exog = self.full_sample_df.benchmark,
            lookback_windows = lookback_windows,
            conf_int = conf_int,
            verbose = verbose).
            assign(
                sample_group = "full_sample",
                position = "short",
                ticker = self.short_name,
                benchmark_ticker = self.benchmark_name))
  
        df_out = pd.concat([
            in_sample_long_ols, out_sample_long_ols, full_sample_long_ols,
            in_sample_short_ols, out_sample_short_ols, full_sample_short_ols])
      
        return df_out

    def plot_single_rolling_ols(self, window: float, fill = True, conf_int = 0.05):

        df_tmp = (self.rolling_ols(
            lookback_windows = [30],
            conf_int = conf_int,
            verbose = False).
            drop(columns = ["lookback_window"]))
        
        fig, axes = plt.subplots(nrows = 4, ncols = 3, figsize = (24,16))

        sample_dict = {
            "in_sample": "In-Sample",
            "out_sample": "Out-of-Sample",
            "full_sample": "Full Sample"}

        conf_out = 1 - conf_int
        counter = 0

        # unfortunately need to unpack the data with for loops
        for k, position in enumerate(df_tmp.position.drop_duplicates().sort_values().to_list()):
            for j, parameter in enumerate(df_tmp.parameter.drop_duplicates().sort_values().to_list()):
                for i, sample in enumerate(["in_sample", "out_sample", "full_sample"]):

                    df_plot_tmp = (df_tmp.query(
                        "position == @position & parameter == @parameter & sample_group == @sample")
                        [["Date", "parameter", "value", "variable", "ticker", "benchmark_ticker"]])
                    
                    ticker, benchmark_ticker = df_plot_tmp.ticker.iloc[0], df_plot_tmp.benchmark_ticker.iloc[0]
                    df_plot_tmp = df_plot_tmp.drop(columns = ["ticker", "benchmark_ticker"])
                    
                    df_value_plot = (df_plot_tmp.query(
                        "variable == 'value'").
                        drop(columns = ["variable"]).
                        pivot(index = "Date", columns = "parameter", values = "value"))
                    
                    df_value_plot.plot(
                        ax = axes[counter,i],
                        color = "black",
                        title = "{} {} {} ({}, Benchmark: {})".format(
                            sample_dict[sample], position, parameter,
                            ticker, benchmark_ticker))
                    
                    if fill == True:

                        df_upper_lower = (df_plot_tmp.query(
                            "variable != 'value'").
                            assign(new_col = lambda x: x.variable.str.split("_").str[0]).
                            drop(columns = ["variable", "parameter"]).
                            pivot(index = "Date", columns = "new_col", values = "value"))
                        
                        axes[counter,i].fill_between(
                            x = df_upper_lower.index,
                            y1 = df_upper_lower.upper,
                            y2 = df_upper_lower.lower,
                            alpha = 0.5,
                            label = "CI: {}%".format(conf_out))
                        
                        axes[counter,i].legend()

                counter += 1

        fig.suptitle("{}-Period Regression".format(window))
        plt.tight_layout()
        
        return df_tmp

    def plot_single_rolling_ols_comparison(self, window: float, conf_int = None, figsize = (16,10)):

        if conf_int == None: 
            conf_int = 0.05
            conf_passed = False

        else: conf_passed = True

        df_tmp = (self.rolling_ols(
            lookback_windows = [window],
            conf_int = conf_int,
            verbose = False).
            drop(columns = ["lookback_window"]))
        
        sample_dict = {
            "in_sample": "In-Sample",
            "out_sample": "Out-of-Sample",
            "full_sample": "Full Sample"}

        if conf_passed == False:

            df_tmp_long = (df_tmp.query(
                "position == 'long' & variable == 'value'").
                drop(columns = ["position", "ticker", "benchmark_ticker"]).
                rename(columns = {"value": "long_value"}))
            
            df_tmp_short = (df_tmp.query(
                "position == 'short' & variable == 'value'").
                drop(columns = ["position", "ticker", "benchmark_ticker"]).
                rename(columns = {"value": "short_value"}))

            df_merge = (df_tmp_long.merge(
                df_tmp_short,
                how = "inner",
                on = ["Date", "parameter", "variable", "sample_group"]).
                drop(columns = ["variable"]).
                melt(id_vars = ["Date", "parameter", "sample_group"]).
                assign(new_col = lambda x: x.variable.str.split("_").str[0] + " " + x.parameter))
            
            fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = figsize)

            for j, parameter in enumerate(df_merge.parameter.drop_duplicates().sort_values().to_list()):
                for i, sample in enumerate(["in_sample", "out_sample", "full_sample"]):
            
                    df_plot_tmp = (df_merge.query(
                        "sample_group == @sample & parameter == @parameter")
                        [["Date", "new_col", "value"]].
                        rename(columns = {"new_col": "parameters"}).
                        pivot(index = "Date", columns = "parameters", values = "value"))
                    
                    df_plot_tmp.plot(
                        ax = axes[i,j],
                        title = sample_dict[sample],
                        color = ["black", "blue"])
                    
                    df_fill = (df_plot_tmp.reset_index().melt(
                        id_vars = "Date").
                        assign(new_col = lambda x: x.parameters.str.split(" ").str[0])
                        [["Date", "new_col", "value"]].
                        pivot(index = "Date", columns = "new_col", values = "value"))
                    
                      
            fig.suptitle("Long: {} Short: {} Benchmark: {} Periods: {}".format(
                self.long_name, self.short_name, self.benchmark_name, window))
            plt.tight_layout()

        if conf_passed == True:

            df_tmp_long = (df_tmp.query(
                "position == 'long'").
                drop(columns = ["position", "ticker", "benchmark_ticker"]).
                rename(columns = {"value": "long_position"}))
            
            df_tmp_short = (df_tmp.query(
                "position == 'short'").
                drop(columns = ["position", "ticker", "benchmark_ticker"]).
                rename(columns = {"value": "short_position"}))
            
            df_merge = (df_tmp_long.merge(
                df_tmp_short,
                how = "inner",
                on = ["Date", "parameter", "variable", "sample_group"]))
            
            df_value = (df_merge.query(
                "variable == 'value'").
                drop(columns = ["variable"]))
            
            fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = figsize)

            for i, parameter in enumerate(df_merge.parameter.drop_duplicates().sort_values().to_list()):
                for j, sample in enumerate(["in_sample", "out_sample", "full_sample"]):

                    df_plot_tmp = (df_value.query(
                        "parameter == @parameter & sample_group == @sample")
                        [["Date", "long_position", "short_position"]].
                        set_index("Date"))
                    
                    df_plot_tmp.plot(
                        ax = axes[j, i],
                        title = sample_dict[sample] + " " + parameter)
                    
                    df_upper_lower = (df_tmp.query(
                        "variable != 'value'").
                        query("parameter == @parameter & sample_group == @sample").
                        drop(columns = ["parameter", "sample_group", "ticker", "benchmark_ticker"]).
                        assign(new_col = lambda x: x.variable.str.split("_").str[0]).
                        drop(columns = ["variable"]))
                    
                    df_upper_lower_long = (df_upper_lower.query(
                        "position == 'long'").
                        drop(columns = ["position"]).
                        pivot(index = "Date", columns = "new_col", values = "value"))
                    
                    df_upper_lower_short = (df_upper_lower.query(
                        "position == 'short'").
                        drop(columns = ["position"]).
                        pivot(index = "Date", columns = "new_col", values = "value"))
                    
                    axes[j,i].fill_between(
                        x = df_upper_lower_long.index, 
                        y1 = df_upper_lower_long.upper,
                        y2 = df_upper_lower_long.lower,
                        alpha = 0.3,
                        label = "Long Position CI: {}%".format((1-conf_int) * 100))
                    
                    
                    axes[j,i].fill_between(
                        x = df_upper_lower_short.index, 
                        y1 = df_upper_lower_short.upper,
                        y2 = df_upper_lower_short.lower,
                        alpha = 0.3,
                        label = "Short Position CI: {}%".format((1 - conf_int) * 100))
                    
                    axes[j,i].legend()
                    
        plt.tight_layout()

    def plot_single_rolling_ols_parameter_comparison(self, ols_window: float, corr_window: float, figsize = (16,10)):

        fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = figsize)

        df_tmp = (self.rolling_ols(
            lookback_windows = [ols_window],
            conf_int = 0.05,
            verbose = False).
            drop(columns = ["lookback_window"]).
            query("variable == 'value'").
            replace({"benchmark": "beta"}))
        
        sample_dict = {
            "in_sample": "In-Sample",
            "out_sample": "Out-of-Sample",
            "full_sample": "Full Sample"}

        for i, sample in enumerate(["in_sample", "out_sample", "full_sample"]):
            for j, parameter in enumerate(["alpha", "beta"]):

                (df_tmp.query(
                    "sample_group == @sample & parameter == @parameter")
                    [["Date", "position", "value"]].
                    pivot(index = "Date", columns = "position", values = "value").
                    assign(corr = lambda x: x.long.rolling(window = corr_window).corr(x.short)).
                    dropna()
                    [["corr"]].
                    plot(
                        ax = axes[i,j], legend = False,
                        title = sample_dict[sample] + " " + parameter,
                        ylabel = "Correlation"))

        plt.tight_layout()

    def plot_single_rolling_ols_hist(self, ols_window: float, figsize = (24,16)):

        df_tmp = (self.rolling_ols(
            lookback_windows = [ols_window],
            conf_int = 0.05,
            verbose = False).
            drop(columns = ["lookback_window"]).
            query("variable == 'value'"))
        
        sample_dict = {
            "in_sample": "In-Sample",
            "out_sample": "Out-of-Sample",
            "full_sample": "Full Sample"}
        
        fig, axes = plt.subplots(nrows = 4, ncols = 3, figsize = figsize)
        counter = 0

        for k, position in enumerate(df_tmp.position.drop_duplicates().sort_values().to_list()):
            for j, parameter in enumerate(df_tmp.parameter.drop_duplicates().sort_values().to_list()):
                for i, sample in enumerate(["in_sample", "out_sample", "full_sample"]):

                    df_tmp_plot = (df_tmp.query(
                        "position == @position & parameter == @parameter & sample_group == @sample")
                        [["value"]])
                    
                    sns.histplot(
                        data = df_tmp_plot.values,
                        ax = axes[counter,i], 
                        bins = int(len(df_tmp_plot) / 10),
                        kde = True,
                        legend = False)
                    
                    axes[counter,i].set_title(sample_dict[sample] + " " + position + " " + parameter)
                    
                counter += 1
        
        fig.suptitle("Long: {}, Short: {}, Benchmark: {}".format(
            self.long_name, self.short_name, self.benchmark_name))
        plt.tight_layout()
        
    def plot_single_rolling_ols_contour(self, ols_window: float, figsize = (20,15)):
        
        df_tmp = (self.rolling_ols(
            lookback_windows = [ols_window],
            conf_int = 0.05,
            verbose = False).
            drop(columns = ["lookback_window"]).
            query("variable == 'value'").
            replace({"benchmark": "beta"}))
        
        sample_dict = {
            "in_sample": "In-Sample",
            "out_sample": "Out-of-Sample",
            "full_sample": "Full Sample"}
        
        df_tmp_long = (df_tmp.query(
                "position == 'long'").
                drop(columns = ["position", "ticker", "benchmark_ticker"]).
                rename(columns = {"value": "long_position"}))
            
        df_tmp_short = (df_tmp.query(
            "position == 'short'").
            drop(columns = ["position", "ticker", "benchmark_ticker"]).
            rename(columns = {"value": "short_position"}))

        df_merge = (df_tmp_long.merge(
            df_tmp_short,
            how = "inner",
            on = ["Date", "parameter", "variable", "sample_group"]))
        
        fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = figsize)
        
        for i, sample in enumerate(["in_sample", "out_sample", "full_sample"]):
                
            sns.kdeplot(
                data = (df_merge.query(
                    "parameter == 'alpha' & sample_group == @sample").
                    rename(columns = {
                        "long_position": "Long Position",
                        "short_position": "Short Position"})),
                ax = axes[i,0],
                x = "Long Position",
                y = "Short Position",
                fill = True,
                cmap = "winter")
            
            axes[i,0].set_title(sample_dict[sample] + " Alpha Contour Map")
            
            sns.kdeplot(
                data = (df_merge.query(
                    "parameter == 'beta' & sample_group == @sample").
                    rename(columns = {
                        "long_position": "Long Position",
                        "short_position": "Short Position"})),
                ax = axes[i,1],
                x = "Long Position",
                y = "Short Position",
                fill = True,
                cmap = "autumn")
            
            axes[i,1].set_title(sample_dict[sample] + " Beta Contour Map")

        fig.suptitle("Long: {}, Short: {}, Benchmark: {}".format(
            self.long_name, self.short_name, self.benchmark_name))    
        plt.tight_layout()