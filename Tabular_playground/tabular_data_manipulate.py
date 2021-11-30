import os
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt

class Tabular_manipulate:
    def __init__(self, tabular_path) -> None:
        self.tabular_path = tabular_path
        self.train_df = None
        self.test_df = None
        self.target_column = None
        self.features_column = None
        
    def loading_data(self) -> None:
        train_path = os.path.join(self.tabular_path, "train.csv")
        self.train_df = pd.read_csv(train_path, index_col='id')


        test_path = os.path.join(self.tabular_path, "test.csv")
        self.test_df = pd.read_csv(test_path, index_col='id')

        print("Train: ", self.train_df.shape)
        print("Test: ", self.test_df.shape)
        self.target_column = list(set(self.train_df.columns) - set(self.test_df.columns))[0]
        self.features_column = self.test_df.columns
        print("Target column: ", self.target_column)
        print(self.train_df.describe(include='all').T)

        print(f'\nNumber of Missing Values in Train: {sum(self.train_df.isnull().sum())}')
        print(f'Number of Missing Values in Test: {sum(self.test_df.isnull().sum())}')
        print("\nFeature type count :")

        for data_type, count in zip(np.unique(self.train_df.dtypes, return_counts=True)[0], np.unique(self.train_df.dtypes, return_counts=True)[1]):
            print(data_type, ":", count)

        print("Unique value: ")
        print(self.train_df.nunique().sort_values(ascending=True))

    def apply_feature_engineering(self):
        
        # Handle categorical type
        self.categorical_handle()
        # Handle numerical type
        self.numerical_handle()

        # Normalize data
        self.normalization()
        
        print(self.train_df.describe(include='all').T)

        pass

    def categorical_handle(self):
        """
        Encoding categorical data
        """
        le = preprocessing.LabelEncoder()
        print("Categorical features:")
        for feature in self.train_df:

            if self.train_df[feature].dtypes == 'object':
                print(feature)
                le.fit(np.unique(self.train_df[feature], return_inverse=False))
                self.train_df[feature] = le.transform(self.train_df[feature])
                self.test_df[feature] = le.transform(self.test_df[feature])


    def numerical_handle(self):
        """
        Cap outlier
        """
        for feature in self.features_column:
            upper_lim = self.train_df[feature].quantile(.95)
            self.train_df.loc[(self.train_df[feature] > upper_lim), feature] = upper_lim
            self.test_df.loc[(self.test_df[feature] > upper_lim), feature] = upper_lim

        pass

    def normalization(self, nor_type="z"):
        """
        There are 3 type of normalize:
        "s" : simple feature scaling : x_new = x_old / x_max
        "m" : Min - Max : x_new = (x_old - x_min) / (x_max - x_min)
        "z" : Z-score : x_new = (x_old - mean) / std
        """
        # Get mean and std of Train Dataframe
        mean_n_std = []
        mean, std = self.train_df[self.features_column].mean(), self.train_df[self.features_column].std()
        mean_n_std.append(mean)
        mean_n_std.append(std)

        mean_n_std = np.array(mean_n_std)
        mean_n_std_frame = pd.DataFrame(data= mean_n_std, index=['mean','std'], columns=self.features_column)

        self.train_df = self.get_normalize(self.train_df[self.features_column], mean_n_std_frame)
        self.test_df = self.get_normalize(self.test_df[self.features_column], mean_n_std_frame)


    def get_normalize(self, data_frame, mean_n_std_frame, nor_type="z"):
        """
        There are 3 type of normalize:
        "s" : simple feature scaling : x_new = x_old / x_max
        "m" : Min - Max : x_new = (x_old - x_min) / (x_max - x_min)
        "z" : Z-score : x_new = (x_old - mean) / std
        """
        assert len(data_frame.columns) == len(mean_n_std_frame.columns)

        standarize_df = data_frame.copy()

        if nor_type=="s":
            standarize_df = standarize_df / standarize_df.max()

        elif nor_type == "m":
            standarize_df = (standarize_df - standarize_df.min())/ standarize_df.max() - standarize_df.min()

        elif nor_type == "z":
            standarize_df = (standarize_df - mean_n_std_frame.loc['mean']) / mean_n_std_frame.loc['std']

        else:
            pass

        return standarize_df

    def features_distribution_visualize(self, sub_name='before_preprocess'):
        document_path = self.tabular_path.split("\\")[-1]
        os.makedirs(document_path, exist_ok=True)

        fig, axes = plt.subplots(10,10,figsize=(12, 12))
        axes = axes.flatten()

        for idx, ax in enumerate(axes):
            sns.kdeplot(data=self.train_df, x=self.train_df.columns[idx], 
                        fill=True, 
                        ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.spines['left'].set_visible(False)
            ax.set_title(f'f{idx}', loc='right', weight='bold', fontsize=10)

        fig.supxlabel('Distribution by feature', ha='center', fontweight='bold')

        plt.savefig(os.path.join(document_path, f"features_distribution_{sub_name}.png"))
            

if "__main__" ==  __name__:
    folder_path = r"D:\Coding_practice\_Data\tabular-playground-series-aug-2021"
    tab = Tabular_manipulate(folder_path)
    tab.loading_data()
    tab.features_distribution_visualize()
    tab.apply_feature_engineering()
    tab.features_distribution_visualize("after_preprocess")