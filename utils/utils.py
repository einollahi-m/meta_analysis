import pandas as pd
import numpy as np
import scipy.stats as stats

class Describtion:
    def __init__(self, df: pd.DataFrame, 
                 target: [str, int],
                 id_columns: [list, str, int] = None):
        self.df = df
        self.columns = df.columns
        
        if id_columns==None:
            self.id_columns = None
        else:
            self.id_columns = id_columns
            df = df.drop(columns=id_columns)
            
        self.target = target
        self.cat_columns = df.drop(columns=self.target).select_dtypes(['category']).columns
        self.cont_columns = df.select_dtypes(['int' , 'float']).columns
        
    def demographic(self, separator: [str, int] = None, ttest=False):
        '''
        separator: is the name of target column,
            It must be str or Int,
        ttest: if ttest=False it would use Mann-Whitney Test
        '''
        def ttest(df, col, separator, key, prec, ttest):
                df_ = df[df[col].notna()]
                n = len(df_)
                n_1 = len(df_[df_[separator] == key])
                n_2 = len(df_[df_[separator] != key])
                mean = round(df_.loc[:, col].mean(), prec)
                mean_1 = round(df_[df_[separator] == key].loc[:, col].mean(), prec)
                mean_2 = round(df_[df_[separator] != key].loc[:, col].mean(), prec)
                std = round(df_.loc[:, col].std(), prec)
                std_1 = round(df_[df_[separator] == key].loc[:, col].std(), prec)
                std_2 = round(df_[df_[separator] != key].loc[:, col].std(), prec)

                if ttest == False:
                    stat, pval = stats.mannwhitneyu(
                        list(df_[df_[separator] == key].loc[:, col]),
                        list(df_[df_[separator] != key].loc[:, col]))
                else:
                    stat, pval = stats.ttest_ind(
                        list(df_[df_[separator] == key].loc[:, col]),
                        list(df_[df_[separator] != key].loc[:, col]))

                return {'variable':col,
                        f"Total": f"{mean} ± {std} ({n})",
                        f"{separator} == {key}": f"{mean_1} ± {std_1} ({n_1})",
                        f"{separator} != {key}": f"{mean_2} ± {std_2} ({n_2})",
                        "pvalue": round(pval, 4)}
                       
        
        df = self.df.copy()
        if separator == None:
            separator = self.target
        
        if separator not in self.columns:
            raise Exception(f"{separator} does not exist in dataframe")
        else:
            key = df[separator].value_counts().sort_values(ascending=False).head(1).index[0]

        new_df = pd.DataFrame(data=None,
                              columns=['variable',
                                       f"Total",
                                       f"{separator} == {key}",
                                       f"{separator} != {key}",
                                       "pvalue"]
                             )
            
        for col in self.cont_columns:
            
            dd = ttest(df=df, col=col, separator=separator, key=key, prec=2, ttest=ttest)
            
            new_df = pd.concat([new_df, pd.DataFrame(pd.Series(data=dd)).T], axis=0)
        return new_df.reset_index(drop=True)
    
    def chi_square(self, separator: [str, int] = None):
        
        def freqency(df, col, target, key):
            if col != target:
            
                df_new = pd.DataFrame(data=None,
                                      columns=[f"{separator} == {key}",
                                               f"{separator} != {key}",
                                               "Total",
                                              ])

                df_ = df.copy()

                cross = pd.crosstab(df_[col], df_[target])
                cross_1 = pd.crosstab(df_[col], df_[target], margins=True).drop(index="All")
                cross_2 = pd.crosstab(df_[col], df_[target], margins=True, normalize='columns')

                stacked = np.dstack((cross_1, cross_2))
                for row in stacked:
                    dataframe = pd.DataFrame(data=[f"{x[0]:.0f} (%{x[1]:.2f})" for x in row], 
                                        index=[f"{separator} == {key}",
                                               f"{separator} != {key}",
                                               "Total"]).T
                    df_new= pd.concat([df_new, dataframe], axis=0)
                    
                if cross.shape == (2, 2):
                    oddsratio = round(stats.contingency.odds_ratio(cross).statistic, 2)
                else:
                    oddsratio = None

                chi, pvalue, dof, expected = stats.contingency.chi2_contingency(cross)
                
                df_new["variable"] = np.nan
                df_new.loc[:, "variable"] = [f"{col} - {x}" for x in cross.index]
                df_new["chi^2"] = round(chi, 4)
                df_new['oddsRatio'] = oddsratio
                df_new['pvalue'] = round(pvalue, 4)

                return df_new[["variable",
                               f"Total",
                               f"{separator} == {key}",
                               f"{separator} != {key}",
                               "chi^2",
                               "oddsRatio",
                               "pvalue"
                              ]]
            
        
        df_ = self.df.copy()
        if separator == None:
            separator = self.target
        
        if separator not in self.columns:
            raise Exception(f"{separator} does not exist in dataframe")
        else:
            key = df_[separator].value_counts().sort_values(ascending=False).head(1).index[0]
            
        
        new_df = pd.DataFrame(data=None,
                              columns=["variable",
                                       f"Total",
                                       f"{separator} == {key}",
                                       f"{separator} != {key}",
                                       "chi^2",
                                       "oddsRatio",
                                       "pvalue"
                                      ]
                             )
            
        for col in self.cat_columns:
            dd = freqency(df=df_, col=col, target=separator, key=key)
            new_df = pd.concat([new_df, dd], axis=0)
        return new_df.reset_index(drop=True)