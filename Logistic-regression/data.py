from sklearn.datasets import make_classification


def generate_data(n_features=2, n_redundant=0,n_informative=2, random_state=1,n_clusters_per_class=1):
    X, y = make_classification(n_features=n_features, n_redundant=n_redundant,
                           n_informative=n_informative, random_state=random_state,
                           n_clusters_per_class=n_clusters_per_class)
    return X, y 