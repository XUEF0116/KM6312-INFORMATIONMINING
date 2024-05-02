from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('df_clean.csv')
data = data.drop(columns=['Unnamed: 0', 'Person ID']).drop_duplicates()

X = data.drop('Sleep Disorder', axis=1)
y = data['Sleep Disorder']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kf = KFold(n_splits=5, shuffle=True, random_state=20)
k_range = range(1, 50)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
    k_scores.append(scores.mean())
optimal_k = k_range[np.argmax(k_scores)]
optimal_score = max(k_scores)

plt.figure(figsize=(15, 10))
plt.plot(k_range, k_scores, color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('K Value and Accuracy')
plt.xlabel('K')
plt.ylabel('Cross-Validated Accuracy')
plt.grid(True)
plt.axvline(x=optimal_k, color='green', linestyle='--')
plt.axhline(y=optimal_score, color='green', linestyle='--')
plt.scatter(optimal_k, optimal_score, color='red')
plt.savefig('Knn')
plt.show()
print(optimal_k)
