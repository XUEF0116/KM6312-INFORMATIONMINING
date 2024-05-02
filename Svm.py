from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.read_csv('df_clean.csv')
data = data.drop(columns=['Unnamed: 0', 'Person ID']).drop_duplicates()

X = data.drop('Sleep Disorder', axis=1)
y = data['Sleep Disorder']

scaler = StandardScaler()
X_scaled_svm = scaler.fit_transform(X)

kf = KFold(n_splits=5, shuffle=True, random_state=20)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svm_scores = {}

for kernel in kernels:
    svm = SVC(kernel=kernel, gamma='auto')
    scores = cross_val_score(svm, X_scaled_svm, y, cv=kf, scoring='accuracy')
    svm_scores[kernel] = scores.mean()

sorted_svm_scores = sorted(svm_scores.items(), key=lambda item: item[1], reverse=True)
print(sorted_svm_scores)
