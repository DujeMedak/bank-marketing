import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

FONTSIZE = 12

# numerical data visualization with seaborn
def plot_distribution_of_numerical_data(column_data, histogram=True, kde=True, rug=False, nbins=None):
    xlabel = column_data.name
    plt.figure(figsize=(16,8))
    plt.xlabel(xlabel, fontsize=FONTSIZE)
    plt.ylabel('Probability Density', fontsize=FONTSIZE) 
    if nbins is not None:
        sns.distplot(column_data, hist=histogram, kde=kde, rug=rug, bins=bins);
    else:       
        sns.distplot(column_data, kde=kde, rug=rug);
        
    plt.show()
    
def plot_distribution_of_numerical_data_with_target(data_frame, column_name, target_column_name='y', xlabel="", histogram=True, kde=True, rug=False, nbins=None):
    class1_value = "no"
    class2_value = "yes"
    target = target_column_name
    fontsize = 20
    
    class1 = data_frame.loc[data_frame[target] == class1_value][column_name]
    class2 = data_frame.loc[data_frame[target] == class2_value][column_name]
    
    plt.figure(figsize=(16,8))
    sns.distplot(class1, kde=kde, hist=histogram, rug=rug, label=class1_value)
    sns.distplot(class2, kde=kde, hist=histogram, rug=rug, label=class2_value)
    
    plt.xlabel(xlabel, fontsize=FONTSIZE) 
    plt.legend()
    plt.ylabel('Probability Density', fontsize=FONTSIZE) 
    plt.show()


# categorical data visualization with seaborn
def plot_categorical_histogram(column_name, data_frame,  xlabel="", ylabel="Frequency", title=""):
    att_count = data_frame[column_name].value_counts()
    plt.figure(figsize=(16,8))
    sns.set(style="darkgrid")
    sns.barplot(att_count.index, att_count.values, alpha=0.9)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
def plot_categorical_histogram_with_target(column_name, data_frame, target_column_name='y', ylabel="Frequency", title="", normalize=True):
    class1_name = "no"
    class2_name = "yes"
    target = target_column_name
    class1_count = 0
    class2_count = 0
    
    att_count = data_frame[[column_name, target]].groupby([target, column_name]).size()
    
    class1 = att_count[class1_name]
    class2 = att_count[class2_name]
    for value1, value2 in zip(class1,class2):
        class1_count += value1
        class2_count += value2
        
    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_ylabel('Frequency')
    if normalize:
        class1 = [int(val/class1_count * 100) for val in class1]
        class2 = [int(val/class2_count * 100) for val in class2]
        ax.set_ylabel('Frequency (normalized)')
        
    columns = data_frame[column_name].unique()

    x = np.arange(len(columns))  # the label locations
    width = 0.35  # the width of the bars

    rects1 = ax.bar(x - width/2, class1, width, label=class1_name)
    rects2 = ax.bar(x + width/2, class2, width, label=class2_name)

    ax.set_xticks(x)
    ax.set_xticklabels(columns)
    ax.legend()
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    ax.title.set_text(title)
    fig.tight_layout()

    plt.show()


# correlation visialization
def plot_correlation(data_frame):
    """
    This is basic implementation that works with numerical data only.
    """
    sns.set(style="white")
    
    # Compute the correlation matrix
    corr = data_frame.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(16, 8))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
    
def plot_correlation_for_categorical_data(data_frame, theilu=True, ):
    """
    This function uses the code from: https://github.com/shakedzy/dython
    To use this part run: pip install dython
    This function can also be used to plot correlation of continuios or mixed
    cases.
    """
    try:
        from dython import nominal
    except ImportError as e:
        raise ModuleNotFoundError("dython module not found")
    
    theilu = nominal.associations(dataset=data_frame, theil_u=True)
    
    
# metric visualization
def plot_confusion_matrix(y_true, y_pred, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.metrics import confusion_matrix
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
