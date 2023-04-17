import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import pairwise_logrank_test

class SurvivalAnalysis:
"""
This code implements a simple survival analysis model by uisng "lifelines" libraries. It can be used to fit a Kaplan-Meier curve to a dataset and to print the log-rank test results.

The code is divided into three classes:

* `SurvivalAnalysis`: This class represents a survival analysis model. It has the following methods:
    * `__init__`: This method initializes the model with the data, the duration columns, and the dependent variable.
    * `fit`: This method fits the model to the data.
    * `plot`: This method plots the Kaplan-Meier curve for the model.
    * `print_logrank`: This method prints the log-rank test results for the model.
"""
    def __init__(self, data, duration_columns, dependant):
        self.data = data
        self.duration_columns = duration_columns
        self.dependant = dependant
        self.T = self.data[self.duration_columns]
        self.C = self.data[self.dependant]

    def fit(self, covariate):
        self.covariate = covariate

    def plot(self, ax=None, show=False):
        # Plot the Kaplan-Meier curve
        if ax is None:
            ax = plt.gca()
        
        item_no = data[self.covariate].nunique()
        colors = self.__get_colors(item_no)    
        line_styles = ['-', '--', '-.', ':', 'x'] 

        for i, unique_value in enumerate(sorted(data[self.covariate].unique(), reverse=True)):
            idx = data[self.covariate] == unique_value
            self.kmf = KaplanMeierFitter()
            self.kmf.fit(self.T[idx], self.C[idx],label=unique_value)
            self.kmf.plot(ax=ax,
                     color=colors[i],
                     ci_show=show,
                     ls=line_styles[i],
                     label=f'{self.covariate} {unique_value}')
    
    def print_logrank(self):
        # Print the log-rank test results
        log_rank = pairwise_logrank_test(self.data[self.duration_columns],
                                         self.data[self.covariate],
                                         self.data[self.dependant])
        return log_rank.summary

    def __get_colors(self, num_colors):
        cm = plt.get_cmap('gist_gray')
        colors = [cm(1.*i/num_colors) for i in range(0,4*num_colors)]
        return colors
