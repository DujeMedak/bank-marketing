def calc_metrics(y_true, y_pred):
    '''
    Prints various evaluation metrics from given predictions (auc, precision, recall, accuracy, f1, average precision). 
    Arguments:
        y_true (numpy 1-D array): true labels (array of 1s and 0s).
        y_pred (numpy 1-D array): predicted labels (array of 1s and 0s).
    Returns:
        None
    '''
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, average_precision_score

    auc = roc_auc_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    print("Auc:%s\nPrec:%s\nRec:%s\nAcc:%s\nF1:%s\nAP:%s" % (auc, prec, rec, acc, f1, average_precision))
    
    
def plot_cumulative_gain(y_true, y_pred_proba, save_path=None):
    '''
    Plots cumulative gain curve 
    Arguments:
        y_true (numpy 1-D array): true labels (array of 1s and 0s).
        y_pred (numpy 2-D array): predicted probabilities (in format obtained from sklearn clf.predict_proba function)
    Returns:
        None
    '''
    # The magic happens here
    import matplotlib.pyplot as plt
    import scikitplot as skplt
    from scikitplot.metrics import cumulative_gain_curve
    import numpy as np
    
    classes = np.unique(y_true)
    perc1, gains1 = cumulative_gain_curve(y_true, y_pred_proba[:, 0], classes[0])
    perc2, gains2 = cumulative_gain_curve(y_true, y_pred_proba[:, 1], classes[1])
    print("area under the curve of class ",classes[0], ":", (np.sum(gains1) / gains1.shape)[0])
    print("area under the curve of class ",classes[1], ":", (np.sum(gains2) / gains2.shape)[0])
    skplt.metrics.plot_cumulative_gain(y_true, y_pred_proba)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    return (perc1, gains1), (perc2, gains2)
    
    
def plot_lift_chart(y_true, y_pred_proba, ax, save_path=None):
    '''
    Plots lift curve 
    Arguments:
        y_true (numpy 1-D array): true labels (array of 1s and 0s).
        y_pred (numpy 2-D array): predicted probabilities (in format obtained from sklearn clf.predict_proba function)
        save_path (str, optional): full path where the plot will be saved. If None the plot will just be displayed but not saved.
    Returns:
        None
    '''
    import matplotlib.pyplot as plt
    import scikitplot as skplt
    
    #dt_y_true = y_true[0]
    #rf_y_true = y_true[1]
    #lgb_y_true = y_true[2]
    
    ax = skplt.metrics.plot_lift_curve(y_true, y_pred_proba, title='Lift Curve', ax=ax, figsize=(10,10), title_fontsize='large',text_fontsize='medium')
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
    return ax
    
    
def plot_lift_curve_custom(y_true, y_probas, title='Lift Curve',
                    ax=None, figsize=None, title_fontsize="large",
                    text_fontsize="medium"):
    """modified version from original: from https://github.com/reiinakano/scikit-plot/blob/26007fbf9f05e915bd0f6acb86850b01b00944cf/scikitplot/metrics.py
    """
    from scikitplot.helpers import cumulative_gain_curve
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate Lift Curve for data with '
                         '{} category/ies'.format(len(classes)))

    # Compute Cumulative Gain Curves
    percentages, gains1 = cumulative_gain_curve(y_true, y_probas[:, 0],
                                                classes[0])
    percentages, gains2 = cumulative_gain_curve(y_true, y_probas[:, 1],
                                                classes[1])

    percentages = percentages[1:]
    gains1 = gains1[1:]
    gains2 = gains2[1:]

    gains1 = gains1 / percentages
    gains2 = gains2 / percentages

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    ax.plot(percentages, gains1, lw=3, label='Class {}'.format(classes[0]))
    ax.plot(percentages, gains2, lw=3, label='Class {}'.format(classes[1]))

    ax.plot([0, 1], [1, 1], 'k--', lw=2, label='Baseline')

    ax.set_xlabel('Percentage of sample', fontsize=text_fontsize)
    ax.set_ylabel('Lift', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid('on')
    ax.legend(loc='lower right', fontsize=text_fontsize)

    return ax
