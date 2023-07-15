import gzip
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from collections import Counter

# Fetch a subset of the 20 newsgroups dataset
categories = ['comp.graphics', 'rec.autos', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

# It's enough to work with 100 training samples and 20 test samples, just to see how the algorithm works
training_set = list(zip(newsgroups_train.data[:100], newsgroups_train.target[:100]))
test_set = list(zip(newsgroups_test.data[:20], newsgroups_test.target[:20]))

k = 5  # Number of neighbors to consider

# Apply the algorithm from the paper: https://arxiv.org/pdf/2212.09410.pdf
for (x1, _) in test_set:
    Cx1 = len(gzip.compress(x1.encode()))
    distance_from_x1 = []
    
    for (x2, y2) in training_set:
        Cx2 = len(gzip.compress(x2.encode()))
        x1x2 = " ".join([x1, x2])
        Cx1x2 = len(gzip.compress(x1x2.encode()))
        ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
        distance_from_x1.append((ncd, y2))
    
    sorted_distances = sorted(distance_from_x1, key=lambda x: x[0])
    top_k_class = [pair[1] for pair in sorted_distances[:k]]
    predict_class = Counter(top_k_class).most_common(1)[0][0]
    
    print(f"Test sample:\n{x1}\n\nPredicted class: {newsgroups_train.target_names[predict_class]}\n{'-'*80}")
