import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from scipy.stats import pearsonr 


#Take CSV file and convert it to a dataframe
def make_df(filename):
    df = pd.read_csv(filename)
    return df 






####################################################
df = make_df('C:/Users/msind/Box Sync/code/github_projects/hptb/combined_hptb_data.csv')
print(df)

sns.scatterplot(x = 'winning_percentage_going_in', y = 'confidence_value', data = df)
plt.show()

sns.scatterplot(x = 'opponent_winning_percentage_going_in', y = 'confidence_value', data = df)
plt.show()


correlation = pearsonr(df['opponent_winning_percentage_going_in'], df['confidence_value'])[0]
pval = pearsonr(df['opponent_winning_percentage_going_in'], df['confidence_value'])[1]

print(correlation)
print(pval)

sns.lmplot(data = df, x = 'opponent_winning_percentage_going_in', y = 'confidence_value', line_kws={'color': 'black'}, ci = None)
plt.show()
