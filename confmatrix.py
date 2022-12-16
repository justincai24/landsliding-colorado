import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:/Users/justi/Desktop/College/Junior/SML 312/Figures/confusion_matrix_imp_pt.csv')
df = df.iloc[: , 2:]
print(df)
fig, ax = plt.subplots(figsize=(8,6)) 
sns.heatmap(df, annot=True, fmt='g')
ax.set_xticklabels(['1','2','3','5','8'])
ax.set_yticklabels(['1','2','3','5','8'])
plt.xlabel('Predicted confidence')
plt.ylabel('True confidence')
plt.title('Point-based, important features')
plt.show()