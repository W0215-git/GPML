import os
import numpy as np
from skimage.color import rgb2gray

def getData(wandb):
    current_work_dir = os.getcwd()
    dataSetPath = os.path.join(current_work_dir, wandb.config['dataDir'])

    train_files = [f for f in os.listdir(dataSetPath) if '_train.npy' in f and not f.startswith('labels')]
    test_files = [f for f in os.listdir(dataSetPath) if '_test.npy' in f and not f.startswith('labels')]

    modalities = sorted(
        set([f.split('_train.npy')[0] for f in train_files]) & set([f.split('_test.npy')[0] for f in test_files]))

    x_train = {}
    x_test = {}

    for modality in modalities:
        x_train[modality] = np.load(os.path.join(dataSetPath, f"{modality}_train.npy"))
        x_test[modality] = np.load(os.path.join(dataSetPath, f"{modality}_test.npy"))
        print(f"✅ Loaded modality: {modality}")

    if wandb.config['Class'] == 2:
        y_train = np.load(os.path.join(dataSetPath, 'labels_train.npy'))
        y_test = np.load(os.path.join(dataSetPath, 'labels_test.npy'))
        print("loading the 2 class labels..........")
    else:
        raise ValueError(f"Unsupported class type: {wandb.config['Class']}. Please check the configuration.")

    print("\n📊 Data Distribution Description 📊\n")
    print("\n📊 **Data Shape Information** 📊\n")
    print(f"{'Modality':<12} {'Dataset':<10} {'Shape':<25}")
    print("-" * 50)

    for modality in x_train.keys():
        print(f"{modality:<12} {'Training':<10} {str(x_train[modality].shape):<25}")
        print(f"{modality:<12} {'Test':<10} {str(x_test[modality].shape):<25}")

    print("\n")
    print("📊 **Label Distribution** 📊\n")
    print(" - **Training Labels Distribution:**")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for label, count in zip(unique_train, counts_train):
        print(f"   Label {label}: {count} samples")

    print("\n - **Test Labels Distribution:**")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    for label, count in zip(unique_test, counts_test):
        print(f"   Label {label}: {count} samples")

    print("\n📊 **Modality Data Statistics** 📊\n")
    print(f"{'Modality':<12} {'Dataset':<10} {'Mean':<15} {'Std Dev':<15}")
    print("-" * 50)

    for modality in x_train.keys():
        mean_train, std_train = np.mean(x_train[modality]), np.std(x_train[modality])
        mean_test, std_test = np.mean(x_test[modality]), np.std(x_test[modality])

        print(f"{modality:<12} {'Training':<10} {mean_train:<15.2f} {std_train:<15.2f}")
        print(f"{modality:<12} {'Test':<10} {mean_test:<15.2f} {std_test:<15.2f}")

    print("\n✅ Data statistics displayed successfully!")

    return modalities, x_train, x_test, y_train, y_test
