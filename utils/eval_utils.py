
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import functools
import numbers
from IPython import display
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchvision
from PIL import Image
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
import skimage
from skimage.transform import resize as res1

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import os

import itertools
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)

def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    display.set_matplotlib_formats('svg')


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        self.num_axes = nrows * ncols
        if self.num_axes == 1:
            self.axes = [self.axes, ]
        xlabel = self._repeat_arg_if_not_iterable(xlabel)
        ylabel = self._repeat_arg_if_not_iterable(ylabel)
        legend = self._repeat_arg_if_not_nested_iterable(legend)
        xlim = self._repeat_arg_if_not_nested_iterable(xlim)
        ylim = self._repeat_arg_if_not_nested_iterable(ylim)
        xscale = self._repeat_arg_if_not_nested_iterable(xscale)
        yscale = self._repeat_arg_if_not_nested_iterable(yscale)
        # Bind arguments with functools
        self.config_axes = [functools.partial(set_axes, self.axes[i], xlabel[i], ylabel[i], 
            xlim[i], ylim[i], xscale[i], yscale[i], legend[i]) for i in range(self.num_axes)]
        self.fmts = fmts
        self.X, self.Y = [None] * self.num_axes, [None] * self.num_axes

    def add(self, x, y):
        if self.num_axes == 1:
            x = [x,]
            y = [y,]
        x = self._repeat_arg_if_not_iterable(x)
        y = self._repeat_arg_if_not_iterable(y)
        for ax_idx, (ax, config_ax, ax_x, ax_y) in enumerate(zip(self.axes, self.config_axes, x, y)):
            # Add multiple data points into the figure
            if not hasattr(ax_y, "__len__"):
                ax_y = [ax_y]
            n = len(ax_y)
            if not hasattr(ax_x, "__len__"):
                ax_x = [ax_x] * n
            if not self.X[ax_idx]:
                self.X[ax_idx] = [[] for _ in range(n)]
            if not self.Y[ax_idx]:
                self.Y[ax_idx] = [[] for _ in range(n)]
            for i, (a, b) in enumerate(zip(ax_x, ax_y)):
                if a is not None and b is not None:
                    self.X[ax_idx][i].append(a)
                    self.Y[ax_idx][i].append(b)
            ax.cla()
            for x, y, fmt in zip(self.X[ax_idx], self.Y[ax_idx], self.fmts):
                ax.plot(x, y, fmt)
            config_ax()
        display.display(self.fig)
        display.clear_output(wait=True)

    def _repeat_arg_if_not_iterable(self, arg):
        if arg is None or isinstance(arg, (str, numbers.Number)):
            return [arg] * self.num_axes
        else:
            return arg

    def _repeat_arg_if_not_nested_iterable(self, arg):
        if arg is None or isinstance(arg, str) or isinstance(arg[0], (numbers.Number, str)):
            return [arg] * self.num_axes
        else:
            return arg

def get_dataloader_workers():
    """Use 4 processes to read the data.

    Defined in :numref:`sec_fashion_mnist`"""
    return 4


def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = Accumulator(2)

    with torch.no_grad():
        for inputs in data_iter:
            
            metric.add(accuracy(net(input_ids=inputs["input_ids"].view(1,-1),attention_mask=inputs["attention_mask"]).logits, inputs["labels"]), size(inputs["labels"]))
    return metric[0] / metric[1]


def evaluate_f1_gpu(net, data_iter, device=None):
    """Compute the f1 score for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions

    
    y_tot =[]
    y_hat_tot = []

    with torch.no_grad():
        for inputs in data_iter:
            
            
            
            y_hat = net(input_ids=inputs["input_ids"].view(1,-1),attention_mask=inputs["attention_mask"]).logits
            
            y_tot  += [inputs["labels"].cpu().data.numpy().tolist()]
            
            y_hat_tot += argmax(y_hat.cpu(), axis=1).data.numpy().tolist()
            
    f1_measure = f1_score(y_tot, y_hat_tot, average="macro")
            

    return f1_measure, y_tot, y_hat_tot

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="grey" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

