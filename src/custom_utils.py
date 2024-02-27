import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from custom_logging import CustomLogger


class DataPrepUtil:
    def __init__(self):
        self.log_obj = CustomLogger('Customised Logger')

    def add_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log_obj.info('START ...')
        missing_col_ind = [f'{var}_na' for var in df.columns]
        df[missing_col_ind] = df.isna().astype(int)
        self.log_obj.info('... FINISH')
        return df

    def col_groups_by_datatypes(self, df: pd.DataFrame) -> dict:
        self.log_obj.info('START ...')
        grouped = df.columns.to_series().groupby(df.dtypes).groups
        col_groups = {k.name: v for k, v in grouped.items()}
        self.log_obj.info('... FINISH')
        return col_groups

    def plot_variable_distribution(self, variables: list, df: pd.DataFrame, bins: int = 50, color='aqua',
                                   figsize: tuple = (12, 8), dpi: int = 150):
        num_plots = len(variables)

        if num_plots > 9:
            raise Exception('Number of plots exceeds')

        num_cols = math.ceil(math.sqrt(num_plots))
        num_rows = math.ceil(num_plots / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, dpi=dpi)
        self.log_obj.info(f'subplots={num_plots}; in ({num_rows},{num_cols}) format. Axes shape is {axes.shape}.')

        if num_plots == 1:
            axes = [axes]

        for i, predictor in enumerate(variables):
            xcoor = i // num_cols
            yccor = i % num_cols
            self.log_obj.info(f'Axes to be plotted is: {xcoor},{yccor}.')
            ax = axes[xcoor, yccor]
            ax.hist(df[predictor], bins=bins, color=color, alpha=0.75)
            ax.axvline(np.mean(df[predictor]), color='magenta', linestyle='--', linewidth=1.0)
            ax.axvline(np.median(df[predictor]), color='yellow', linestyle='--', linewidth=1.0)
            ax.set_title(f'Distribution of variable : {predictor}')

        for i in range(num_plots, num_cols * num_rows):
            xcoor = i // num_cols
            yccor = i % num_cols
            self.log_obj.info(f'Axes to be suppressed is: {xcoor},{yccor}.')
            ax = axes[xcoor, yccor]
            ax.axis('off')

        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    def plot_residual_distribution(self, variables: list, errors: pd.DataFrame, df: pd.DataFrame, color='grey',
                                   edgecolor='black', figsize: tuple = (12, 8), dpi: int = 150):
        num_plots = len(variables)

        if num_plots > 9:
            raise Exception('Number of plots exceeds')

        num_cols = math.ceil(math.sqrt(num_plots))
        num_rows = math.ceil(num_plots / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, dpi=dpi)
        self.log_obj.info(f'subplots={num_plots}; in ({num_rows},{num_cols}) format. And axes shape={1 if num_plots == 1 else axes.shape}')

        if num_plots == 1:
            axes = [axes]

        for i, predictor in enumerate(variables):
            xcoor = i // num_cols
            yccor = i % num_cols
            self.log_obj.info(f'Axes to be plotted is: {xcoor},{yccor}.')
            ax = axes[i] if num_plots == 1 else axes[xcoor, yccor]
            ax.scatter(y=errors, x=df[variables[i]], s=20, color=color, edgecolor=edgecolor, alpha=0.75)
            ax.axhline(y=0, color='magenta', linestyle='--', linewidth=1.0)
            ax.set_title(f'Residual spread against : {predictor}')
            ax.set_xlabel(variables[i])
            ax.set_ylabel('Residuals')

        for i in range(num_plots, num_cols * num_rows):
            xcoor = i // num_cols
            yccor = i % num_cols
            self.log_obj.info(f'Axes to be suppressed is: {xcoor},{yccor}.')
            ax = axes[i] if num_plots == 1 else axes[xcoor, yccor]
            ax.axis('off')

        plt.suptitle('Visualising scedasticity : residual vs predictor spread')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def remove_missing_values(df: pd.DataFrame):
        """
        Remove missing values from a dataframe
        """
        return df.dropna()

    @staticmethod
    def replace_missing_values(df: pd.DataFrame, value: any):
        """
        Replace missing values in a dataframe with a specified value
        """
        return df.fillna(value)

    @staticmethod
    def encode_categorical_data(df: pd.DataFrame):
        """
        One-hot encode categorical variables in a dataframe
        """
        return pd.get_dummies(df)

    @staticmethod
    def plot_scatter(df: pd.DataFrame, x: str, y: str):
        """
        Generate a scatter plot from a dataframe
        """
        plt.scatter(df[x], df[y])
        plt.title(f'Scatter plot of {y} vs {x}')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    @staticmethod
    def plot_histogram(df: pd.DataFrame, column: str):
        """
        Generate a histogram from a dataframe
        """
        plt.hist(df[column])
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()

    @staticmethod
    def plot_bar_chart(df: pd.DataFrame, column: str):
        """
        Generate a bar chart from a dataframe
        """
        df[column].value_counts().plot(kind='bar')
        plt.title(f'Bar chart of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()
