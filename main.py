from PCA import PCA
from sklearn.datasets import load_iris
iris = load_iris()
X = iris['data']
y = iris['target']


n_samples, n_features = X.shape

print('Number of samples:', n_samples)
print('Number of features:', n_features)
pca = PCA(2)
print('Number of extracted features/principal components:', pca.n_components)
pca.fit(X)
print(pca.transform())
