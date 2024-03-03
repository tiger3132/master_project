import numpy as np
import csv

with open("resnet18_cifar100_test.csv", 'r', encoding="utf-8") as f:
    data_file = csv.reader(f)
    temp = next(data_file)
    row_count = sum(1 for row in data_file)
    print(f"row count: {row_count}")
    n_samples = int(row_count)
    n_features = len(temp[:-1])
    print(f"features: {n_features}")
    target_name = temp[-1] 
    data = np.empty((n_samples, n_features))
    target = np.empty((n_samples,))
    f.seek(0)
    data_file = csv.reader(f)
    temp = next(data_file)
    for i, row in enumerate(data_file):
        data[i] = np.asarray(row[:-1], dtype=np.float64)
        target[i] = np.asarray(row[-1], dtype=np.float64)


