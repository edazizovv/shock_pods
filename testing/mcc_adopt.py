#


#
import numpy
import pandas
from sklearn.metrics import matthews_corrcoef, confusion_matrix


#


#
def cm_adopted(data, clusters, categories):
    agg = pandas.crosstab(index=data[clusters], columns=data[categories])
    rates = agg / agg.sum(axis=0)

    clusters_re = {}
    categories_re = {}
    n = 0
    while rates.shape[0] > 0:
        n += 1
        ix_mx_clusters = numpy.where(rates.values == rates.values.max())[0][0]
        ix_mx_categories = numpy.where(rates.values == rates.values.max())[1][0]
        mx_cluster = rates.index[ix_mx_clusters]
        mx_category = rates.columns[ix_mx_categories]
        clusters_re[mx_cluster] = n
        categories_re[mx_category] = n

        rates = rates.drop(index=mx_cluster, columns=mx_category)

    def convert_clusters(x):
        return clusters_re[x]

    def convert_categories(x):
        return categories_re[x]

    data[clusters + '_RE'] = data[clusters].apply(func=convert_clusters)
    data[categories + '_RE'] = data[categories].apply(func=convert_categories)

    return data[clusters + '_RE'], data[categories + '_RE']


# by https://en.wikipedia.org/wiki/Phi_coefficient
def mcc_final(c):
    ra = range(c.shape[0])
    num = sum([sum([sum([c[k, k] * c[l, m] - c[k, l] * c[m, k] for m in ra]) for l in ra]) for k in ra])
    den_p1 = sum([sum([c[k, l] for l in ra]) * sum([sum([c[k_, l_] for l_ in ra]) for k_ in ra if k_ != k]) for k in ra]) ** 0.5
    den_p2 = sum([sum([c[l, k] for l in ra]) * sum([sum([c[l_, k_] for l_ in ra]) for k_ in ra if k_ != k]) for k in ra]) ** 0.5
    return num / (den_p1 * den_p2)


def mcc_adopted(y_pred, y_true):
    data = pandas.DataFrame(data={'y_pred': y_pred, 'y_true': y_true})
    y_pred_new, y_true_new = cm_adopted(data=data, clusters='y_pred', categories='y_true')
    score = matthews_corrcoef(y_true=y_true_new, y_pred=y_pred_new)
    return score


a = [2, 1, 2, 2, 2]
b = [1, 2, 2, 1, 1]
data = pandas.DataFrame(data={'a': a, 'b': b})

r1 = mcc_final(confusion_matrix(y_true=a, y_pred=b))
r2 = matthews_corrcoef(y_true=a, y_pred=b)

r = cm_adopted(data=data.copy(), clusters='a', categories='b')

