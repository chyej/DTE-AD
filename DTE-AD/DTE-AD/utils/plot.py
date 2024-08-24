import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np

from utils.utils import *

plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 9, 4
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

os.makedirs('plots', exist_ok=True)

experiment = '_ours'


def smooth(y, box_pts=1):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plotter(name, y_true, y_pred, ascore, labels, threshold):
    os.makedirs(os.path.join('plots', experiment, name), exist_ok=True)

    threshold = np.array(threshold)
    threshold = threshold.reshape(1, -1)
    detect = np.stack(detect, axis=1)

    pdf = PdfPages(f'plots/{experiment}/{name}/output_00.pdf')

    for dim in range(y_true.shape[1]):
        y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
        th, pr = threshold[:, dim], detect[:, dim]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.set_ylabel('Value')
        ax1.set_title(f'Dimension = {dim}')
        ax1.plot(smooth(y_t), linewidth=0.4, label='True')
        ax1.plot(smooth(y_p), '-', alpha=0.7, linewidth=0.7, label='Reconstructed')
        #ax1.margins(x = 0.05, y = 0.05)
        ax1.grid(color='lightgray')
        ax3 = ax1.twinx()
        #ax3.plot(l, '--', linewidth=0.3, color='blue', alpha=0.5)
        #ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.5, label='Anomaly')
        ax3.fill_between(np.arange(l.shape[0]), 0, 1, where=l>0.5, color='pink', alpha=0.5, label='Anomaly', transform=ax3.get_xaxis_transform())

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax3.legend(lines1 + lines3, labels1 + labels3, loc='upper right')

        ax2.plot(smooth(a_s), linewidth=0.2, color='black', alpha=0.8, label='Anomaly score')
        ax2.axhline(th, linewidth=0.5, color='red', linestyle='-', alpha=0.7, label='Threshold')
        #ax2.margins(x = 0.05, y = 0.05)
        ax2.grid(color='lightgray')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Anomaly Score')
        ax4 = ax2.twinx()
        #ax4.plot(pr, '--', linewidth=0.3, color='pink', alpha=0.5)
        #ax4.fill_between(np.arange(pr.shape[0]), pr, color='pink', alpha=0.3, label='Predicted')
        ax2.margins(0)


        lines2, labels2 = ax2.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax4.legend(lines2 + lines4, labels2 + labels4, loc='upper right')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    pdf.close()


class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def plot_accuracies(accuracy_list, folder):
    os.makedirs(f'plots/{experiment}/{folder}/', exist_ok=True)
    trainAcc = [i[0] for i in accuracy_list]
    lrs = [i[1] for i in accuracy_list]
    plt.xlabel('Epochs')
    plt.ylabel('Average Training Loss')
    plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
    plt.twinx()
    plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='--', marker='.')
    plt.savefig(f'plots/{experiment}/{folder}/training-graph.pdf')
    plt.clf()


def cut_array(percentage, arr):
    print(f'{color.BOLD}Slicing dataset to {int(percentage * 100)}%{color.ENDC}')
    mid = round(arr.shape[0] / 2)
    window = round(arr.shape[0] * percentage * 0.5)
    return arr[mid - window: mid + window, :]


def getresults2(df, result):
    results2, df1, df2 = {}, df.sum(), df.mean()
    for a in ['FN', 'FP', 'TP', 'TN']:
        results2[a] = df1[a]
    for a in ['precision', 'recall']:
        results2[a] = df2[a]
    results2['f1*'] = 2 * results2['precision'] * results2['recall'] / (results2['precision'] + results2['recall'])
    return results2



def plotter_o(name, y_true, y_pred, ascore, labels, threshold):
    os.makedirs(os.path.join('plots', experiment, name), exist_ok=True)
    pdf = PdfPages(f'plots/{experiment}/{name}/output.pdf')

    threshold = np.array(threshold)
    #print(f'threshold :{threshold.shape}')
    threshold = threshold.reshape(1, -1)
    threshold = np.repeat(threshold, ascore.shape[0], axis=0)
    print(f'threshold :{threshold.shape}')

    for dim in range(y_true.shape[1]):
        y_t, y_p, l, a_s, th = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim], threshold[:, dim]
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.set_ylabel('Value')
        ax1.set_title(f'Dimension = {dim}')

        ax1.plot(smooth(y_t), linewidth=0.2, color='black', label='True')
        ax1.plot(smooth(y_p), '-', alpha=0.6, color='orange', linewidth=0.3, label='Predicted')
        ax1.fill_between(np.arange(l.shape[0]), 0, 1, where=l>0.5, color='blue', alpha=0.3, label='Ground truth', transform=ax1.get_xaxis_transform())
        ax1.margins(0)
        ax1.grid(alpha=0.5, color='lightgray')

        ax1.legend(loc='upper right')

        #lines1, labels1 = ax1.get_legend_handles_labels()
        #lines3, labels3 = ax3.get_legend_handles_labels()
        #ax3.legend(lines1 + lines3, labels1 + labels3, loc='upper right')

        #if dim == 0:
        #    ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))

        ax2.plot(smooth(a_s), linewidth=0.2, color='g', label='Anomaly Score')
        ax2.plot(th, linewidth=3, color='red', alpha=0.7, label='Threshold')
        #ax2.axhline(th, linewidth=3, color='red', linestyle='-', alpha=0.7, label='Threshold')
        #ax2.fill_between(np.arange(smooth(a_s).shape[0]), 0, 1, where=smooth(a_s)>th, color='yellow', alpha=0.6, label='Anomaly', transform=ax2.get_xaxis_transform())
        ax2.margins(0)
        ax2.grid(color='lightgray')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Anomaly Score')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        pdf.savefig(fig)
        plt.close()
    pdf.close()
    
    
    
def plotter_overall(name, y_true, y_pred, ascore, labels, threshold):
    os.makedirs(os.path.join('plots', experiment, name), exist_ok=True)
    pdf = PdfPages(f'plots/{experiment}/{name}/output_overall.pdf')
    
    fig, ax = plt.subplots(nrows=y_true.shape[1], ncols=1)



    for dim in range(y_true.shape[1]):
        y_t, y_p, l, a_s, th = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim], threshold[:, dim]
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.set_ylabel('Value')
        ax1.set_title(f'Dimension = {dim}')

        ax1.plot(smooth(y_t), linewidth=0.2, color='black', label='True')
        ax1.plot(smooth(y_p), '-', alpha=0.6, color='orange', linewidth=0.3, label='Predicted')
        ax1.fill_between(np.arange(l.shape[0]), 0, 1, where=l>0.5, color='blue', alpha=0.3, label='Ground truth', transform=ax1.get_xaxis_transform())
        ax1.margins(0)
        ax1.grid(alpha=0.5, color='lightgray')

        ax1.legend(loc='upper right')

        #lines1, labels1 = ax1.get_legend_handles_labels()
        #lines3, labels3 = ax3.get_legend_handles_labels()
        #ax3.legend(lines1 + lines3, labels1 + labels3, loc='upper right')

        #if dim == 0:
        #    ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))

        ax2.plot(smooth(a_s), linewidth=0.2, color='g', label='Anomaly Score')
        ax2.plot(th, linewidth=3, color='red', alpha=0.7, label='Threshold')
        #ax2.axhline(th, linewidth=3, color='red', linestyle='-', alpha=0.7, label='Threshold')
        #ax2.fill_between(np.arange(smooth(a_s).shape[0]), 0, 1, where=smooth(a_s)>th, color='yellow', alpha=0.6, label='Anomaly', transform=ax2.get_xaxis_transform())
        ax2.margins(0)
        ax2.grid(color='lightgray')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Anomaly Score')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        pdf.savefig(fig)
        plt.close()
    pdf.close()