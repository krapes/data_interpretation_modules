# Functions to analyse and plot results
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-deep')
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.facecolor'] = 'white'

import matplotlib
from matplotlib.dates import DateFormatter

import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


# evaluate number of good predictions
def evaluate(test, predictions):
    predictions["actual"] = test.fraud.values
    predictions.columns = ["prediction", "actual"]
    #predictions["date"] = test.date.values
    #predictions["item"] = test.item.values
    #predictions["store"] = test.store.values
    predictions["residual"] = predictions.actual - predictions.prediction
    predictions["sresidual"] = predictions.residual / np.sqrt(predictions.actual)
    predictions["fit"] = 0
    # if residual is positive there are not enough items in the store
    predictions.loc[predictions.residual > 0, "fit"] = 0
    # if residual is zero or negative there are enough or more items in the store
    predictions.loc[predictions.residual <= 0, "fit"] = 1
    items = predictions.shape[0]
    more_or_perfect = sum(predictions.fit)
    less = items - more_or_perfect
    return (items, less, more_or_perfect)


# print result of evaluation
def print_evaluation(predictions, less, more_or_perfect):
    # set if no figure on output
    # %matplotlib inline

    # create scatter plot to show numbers of the errors
    name = ["False Negatives", "more or perfect"]
    count = [less, more_or_perfect]
    size = [v/2 for v in count]
    rank = [-4, 4]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis([-10, 10, -30000, 300000])
    ax.scatter(rank, count, s=size,  marker='o', c=["#c44e52", "#55a868"])

    for n, c, r in zip(name, count, rank):
        plt.annotate("{}".format(c), xy=(r, c), ha="center", va="center", color="white", weight='bold', size=15)
        plt.annotate(n, xy=(r, c), xytext=(0, 10),
                     textcoords="offset points", ha="center", va="bottom", color="white", weight='bold', size=12)
    plt.title("Ratio between acceptable and nonacceptable predictions", weight='bold', size=20)
    plt.axis('off')
    #plt.show()

    plt.figure(figsize=(20, 10))
    n, bins, patches = plt.hist(x=predictions.sresidual, bins='auto')
    plt.grid(axis='both')
    plt.xlabel('Value of residual', size=20)
    plt.ylabel('Frequency', size=20)
    plt.title('Histogram of standardized residuals \n mean: %f\n variance: %f' % (
    np.mean(predictions.sresidual), np.var(predictions.sresidual)), weight='bold', size=20)
    maxfreq = n.max()
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    for i in range(len(patches)):
        if bins[i] > 0 and round(bins[i + 1]) >= 0:
            patches[i].set_facecolor('#c44e52')
        else:
            patches[i].set_facecolor('#55a868')
    plt.show()
    plt.savefig(os.path.join(dir_path, 'plots/evaluation'))



# residual analysis
def print_residuals(predictions, predictions_custom):
    # histograms
    plt.figure(figsize=(20, 10))
    n, bins, patches = plt.hist(x=[predictions.sresidual, predictions_custom.sresidual],
                                bins='auto', label=['residual', 'residual custom'])
    plt.grid(axis='both')
    plt.xlabel('Value of standardized residual', size=22)
    plt.ylabel('Frequency', size=20)
    plt.title('Histograms of standardized residuals', weight='bold', size=22)
    plt.legend(loc='upper right')
    maxfreq = n[0].max() if n[0].max() > n[1].max() else n[1].max()
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

    # actual vs. predicted

    f = plt.figure(figsize=(30, 10))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    ax.scatter(predictions.prediction, predictions.actual,
               s=10, label='Gaussian', alpha=0.7)
    ax.scatter(predictions_custom.prediction, predictions_custom.actual,
               s=10, label='Custom', alpha=0.7)
    plt.grid(axis='both')
    ax.set_xlabel('Predicted', size=20)
    ax.set_ylabel('Actual', size=20)
    ax.legend(loc='upper right')
    ax.set_title("Predicted vs. Actual", weight='bold', size=22)

    # residual error
    ax2.scatter(predictions.prediction, predictions.sresidual,
                s=10, label='Gaussian', alpha=0.7)
    ax2.scatter(predictions_custom.prediction, predictions_custom.sresidual,
                s=10, label='Custom', alpha=0.7)
    plt.hlines(y=0, xmin=0, xmax=200, linewidth=2)
    plt.grid(axis='both')
    ax2.set_xlabel('Prediction', size=20)
    ax2.set_ylabel('Standardized residual', size=20)
    ax2.legend(loc='upper right')
    ax2.set_title("Standardized residual errors", weight='bold', size=22)
    #plt.show()
    plt.savefig(os.path.join(dir_path, 'plots/residuals'))


# prediction analysis
def print_predictions(predictions, predictions_custom, item, store):
    one_item_data = predictions[(predictions.item == item) & (predictions.store == store)]
    one_item_data_custom = predictions_custom[(predictions_custom.item == item) & (predictions_custom.store == store)]
    fig, ax = plt.subplots(figsize=(30, 10))
    ax.set_title(("Prediction vs. Actual item: %d store: %d") % (item, store), weight='bold', size=22)
    ax.set_xlabel('Date', size=20)
    ax.set_ylabel('Number of sold items', size=20)
    plt.grid(axis='both')
    plt.xticks(rotation=70)
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%d-%m-%Y"))
    ax.xaxis.set_minor_formatter(DateFormatter("%d-%m-%Y"))
    ax.plot_date(one_item_data.date, one_item_data.prediction, "o:", alpha=0.6, ms=10, label="predicted - Gaussian")
    ax.plot_date(one_item_data_custom.date, one_item_data_custom.prediction, "o:", ms=10, alpha=0.6,
                 label="predicted - custom")
    ax.plot_date(one_item_data.date, one_item_data.actual, "o", markersize=10, label="actual")
    ax.legend(loc='upper right')
    plt.savefig(os.path.join(dir_path, 'plots/predictions'))


def plot_scoring_history(history_mm, history_cmm):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Training scoring history", weight='bold', size=22)
    ax.set_xlabel('Number of trees', size=20)
    ax.set_ylabel('Custom metric', size=20)
    plt.grid(axis='both')
    ax.plot(history_mm, "o:", ms=10, label="Gaussian distribution & custom metric")
    ax.plot(history_cmm, "o:", ms=10, label="Custom distribution & custom metric")
    ax.legend(loc='upper right')
    plt.savefig(os.path.join(dir_path, 'plots/scoring_history'))