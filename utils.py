import os
import re
import numpy as np

from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt


from dotenv import load_dotenv
load_dotenv()

def get_LIBSVM(dataset_name: str):
    datasets_path = os.getenv("LIBSVM_DIR")
    
    n_features_dict = {
        "mushrooms": 112,
        "phishing": 68,
    }
    
    n_features = n_features_dict.get(dataset_name)
    
    x = re.search(r"^[a]\d[a](\.t)?$", dataset_name)
    if x is not None:
        n_features: int = 123
    x = re.search(r"^[w]\d[a](\.t)?$", dataset_name)
    if x is not None:
        n_features: int = 300
        
    trainX, trainY = load_svmlight_file(f"{datasets_path}/{dataset_name}", n_features=n_features)
    return trainX, trainY

def make_synthetic_binary_classification(n_samples: int, n_features: int, symmetric: bool = False, seed: int = 0):
    np.random.seed(seed)
    data = np.random.randn(n_samples, n_features)
    
    if symmetric:
        assert n_samples == n_features, f"n_samples must be equal to n_features to get symmetric matrix. " \
            f"Currently n_samples={n_samples}, n_features={n_features}."
        data = (data + data.T) / 2
    w_star = np.random.randn(n_features)

    target = data @ w_star
    target[target <= 0.0] = -1.0
    target[target > 0.0] = 1.0

    return data, target


def map_classes_to(target, new_classes):
    old_classes = np.unique(target)
    new_classes = np.sort(new_classes)
    
    if np.array_equal(old_classes, new_classes):
        return target
    
    assert np.unique(target).size == len(new_classes), \
        f"Old classes must match the number of new classes. " \
        f"Currently ({np.unique(target).size}) classes are being mapped to ({len(new_classes)}) new classes."

    mapping = {v: t for v, t in zip(old_classes, new_classes)}
    target = np.vectorize(mapping.get)(target)
    return target

def plotter(histories, labels, colors=None, linestyles=None, 
            linewidths=None, markers=None, f_star=None, suptitle=None, 
            threshold=1e-10, xlims=None, tight_layout=True, filename=None):
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)
        
    if colors is None:
        colors = list(plt.cm.colors.XKCD_COLORS.keys())
        np.random.shuffle(colors)
        colors = list(plt.cm.colors.BASE_COLORS) + colors
        
    if linestyles is None:
        linestyles = ["-" for x in histories]
    
    if linewidths is None:
        linewidths = [1.5 for x in histories]
        
    if markers is None:
        markers = [" " for x in histories]
        
    if f_star is None:
        f_star = np.min([x["loss"] for x in histories])
    else:
        f_star = np.min([f_star for x in histories])
    
    for history, label, c, ls, lw, m in zip(histories, labels, colors, linestyles, linewidths, markers):
        f_suboptim = (history["loss"] - f_star) / (history["loss"][0] - f_star)
        f_suboptim[f_suboptim < threshold] = 0.0
        
        markevery = [x + np.random.randint(0, 1) for x in range(0, len(history["loss"]), len(history["loss"]) // 10)]
        
        ax[0].semilogy(f_suboptim, linestyle=ls, linewidth=lw, color=c, markevery=markevery, marker=m)
        ax[1].semilogy(history["time"], f_suboptim, linestyle=ls, linewidth=lw, color=c, label=label, markevery=markevery, marker=m)

    if f_star.sum() == 0.0:
        ax[0].set_ylabel(r"$f(x_k)/f(x_0)$")
        ax[1].set_ylabel(r"$f(x_k)/f(x_0)$")
    else:
        ax[0].set_ylabel(r"$(f(x_k) - f^*)/(f(x_0) - f^*)$")
        ax[1].set_ylabel(r"$(f(x_k) - f^*)/(f(x_0) - f^*)$")
        
        
    ax[0].set_xlabel("Steps")
    ax[1].set_xlabel("Time, sec")
    
    if xlims is not None:
        ax[1].set_xlim(right=xlims[1])

    fig.legend()
    ax[0].grid()
    ax[1].grid()
    if tight_layout:
        fig.tight_layout()
        
    if filename is not None:
        fig.savefig(filename)
    else:
        fig.show()    
    
    
def plotter_eig(histories, labels, colors=None, linestyles=None, 
            linewidths=None, markers=None, f_star=None, suptitle=None, 
            threshold=1e-10, tight_layout=True):
    
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)
        
    if colors is None:
        colors = list(plt.cm.colors.XKCD_COLORS.keys())
        np.random.shuffle(colors)
        colors = list(plt.cm.colors.BASE_COLORS) + colors
        
    if linestyles is None:
        linestyles = ["-" for x in histories]
    
    if linewidths is None:
        linewidths = [1.5 for x in histories]
        
    if markers is None:
        markers = [" " for x in histories]
        
    if f_star is None:
        f_star = np.min([x["loss"] for x in histories])
    else:
        f_star = np.min([f_star for x in histories])
    
    for history, label, c, ls, lw, m in zip(histories, labels, colors, linestyles, linewidths, markers):
        f_suboptim = history["loss"] - f_star
        f_suboptim[f_suboptim < threshold] = 0.0
        
        markevery = [x + np.random.randint(0, 3) for x in range(0, len(history["loss"]), len(history["loss"]) // 10)]
        
        ax[0][0].semilogy(f_suboptim, linestyle=ls, linewidth=lw, color=c, markevery=markevery, marker=m)
        ax[0][1].semilogy(history["time"], f_suboptim, linestyle=ls, linewidth=lw, color=c, markevery=markevery, marker=m)
        ax[1][0].semilogy(history["min_eigval"], linestyle=ls, linewidth=lw, color=c, markevery=markevery, marker=m)
        ax[1][1].semilogy(history["max_eigval"], linestyle=ls, linewidth=lw, color=c, markevery=markevery, marker=m,  label=label)
        

    ax[0][0].set_ylabel(r"$f(x_k) - f^*$")
    ax[0][0].set_xlabel("Steps")
    ax[0][1].set_ylabel(r"$f(x_k) - f^*$")
    ax[0][1].set_xlabel("Time, sec")
    ax[1][0].set_ylabel(r"$\lambda_{{min}}$")
    ax[1][0].set_xlabel("Steps")
    ax[1][1].set_ylabel(r"$\lambda_{{max}}$")
    ax[1][1].set_xlabel("Steps")
    

    fig.legend()
    ax[0][0].grid()
    ax[0][1].grid()
    ax[1][0].grid()
    ax[1][1].grid()
    if tight_layout:
        fig.tight_layout()
    fig.show()    