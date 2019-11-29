import os
import pickle
import argparse
import numpy as np
import progressbar
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix

def savePickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def loadPickle(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def storemodel(model, name):
    root = str(Path(__file__).parent.parent)
    modeldir = root + "/models"
    checkandcreatedir(modeldir)
    filepath = modeldir + "/" + name
    savePickle(filepath, model)
    print("saved", filepath)

def loadmodel(filename):
    try:
        model = loadPickle(filename)
        return model
    except:
        raise Exception("Model not found: " + filename )

def checkandcreatedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.PuRd):
    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

test = 1
storemodel(test, "test_model")

ap = argparse.ArgumentParser()
ap.add_argument("-t1", "--train", required=True, help="Path to the train features")
ap.add_argument("-t2", "--test", required=True, help="Path to the test features")
ap.add_argument("-m", "--model", required=False, default="", help="Path to the train features")
args = ap.parse_args()

train_features_path = args.train
test_features_path = args.test
model_dir = args.model if args.model != "" else False

## Load data
if not model_dir:
    train_features = loadmodel(train_features_path)
    train_x = train_features["X"]
    train_y = train_features["Y"]
    train_x = np.reshape(train_x, (len(train_x), 1000))

test_features = loadmodel(test_features_path)
test_x = test_features["X"]
test_y = test_features["Y"]
test_x = np.reshape(test_x, (len(test_x), 1000))
class_names = np.array(list(set(test_y)))
n_classes = len(class_names)

## Train
if not model_dir:
    print("Training")
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(train_x, train_y)
    storemodel(clf, "cifar_svc")
else:
    clf = loadmodel(model_dir)

## Test
print("Predicting")
preds = clf.predict(test_x)

## Get metrics
# Accuracy
accuracy = accuracy_score(test_y, preds)
print("Accuracy:", accuracy)

# Confusion Matrix
# confmat = confusion_matrix(test_y, preds)
# plt.figure()
plot_confusion_matrix(test_y, preds, classes=class_names, normalize=True, title='Normalized confusion matrix')

# ROC Curve
# Compute ROC curve and ROC area for each class
test_y = np.reshape(test_y, (len(test_y), 1))
preds = np.reshape(preds, (len(preds), 1))
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[0], tpr[0], _ = roc_curve(test_y[:, 0], preds[:, 0])
roc_auc[0] = auc(fpr[0], tpr[0])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), preds.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 1
plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
