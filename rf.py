import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('df_clean.csv')
data = data.drop(columns=['Unnamed: 0', 'Person ID']).drop_duplicates()

X = data.drop('Sleep Disorder', axis=1)
y = data['Sleep Disorder']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=20)

rf = RandomForestClassifier(n_estimators=100, random_state=20)
rf.fit(X_train, y_train)

feature_importances = rf.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_importances = feature_importances[sorted_indices]
sorted_features = X.columns[sorted_indices]

plt.figure(figsize=(15, 10))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), sorted_feature_importances, align='center')
plt.xticks(range(X_train.shape[1]), sorted_features, rotation=90)
plt.tight_layout()
plt.savefig('RF')
plt.show()
print(sorted_features.tolist())
print(sorted_feature_importances.tolist())

