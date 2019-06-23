import numpy as np
import math as m

def entropy(*p, base=2):
    """
    calculate entropy, [optional]
    example:
        ent = entropy(0.25, 0.1, 0.65, [base=2])

    :param p: probability of occurring each element, or ratio of each element
    :param base: base of log
    :return: entropy (impurity)
    """
    return sum([-1 * pi * m.log(pi, base) for pi in p])

def entropyForD(D, base=2):
    """
    calculate entropy for a data set, given label is in last column.
    example:
       D = [['3' '1' '2' 'A']
            ['3' '5' '3' 'B']
            ['3' '6' '2' 'C']
            ['3' '7' '3' 'C']]
       ent_D = entropyForD(D)

    :param D: data set with label in last column
    :param base: base of log
    :return: entropy (impurity) of data set
    """
    D_labels = D[:, -1]
    _, counts = np.unique(D_labels, return_counts=True)
    ratios = counts / np.sum(counts)
    entropy_Dv = entropy(*ratios, base=base)
    return entropy_Dv

def splitDataWithAttribute(D, attr_idx, base=2):
    """
    split a data set by given attribute. data set is with labels in last column.
    example:
        D = [['3' '1' '2' 'A']
             ['3' '5' '3' 'B']
             ['3' '6' '2' 'C']
             ['3' '7' '3' 'C']]
        Dvs = splitDataWithAttribute(D, 1)

    :param D: data set
    :param attr_idx: index of attribute used to split data set
    :param base: base of log
    :return: a list of dictionary that has 3 key-value pairs.
             keys are: "split_by_attr", one possible value of chosen attribute
                       "data", split sub data set
                       "entropy", entropy of split sub data set
                       "ratio", ratio of number of samples between subset and original data set
    """
    attr_col = D[:, attr_idx]
    attr_uniques = np.unique(attr_col)
    n_D_sample = np.size(D, axis=0)

    Dvs = []
    for e_attr in attr_uniques:
        e_attr_row = np.where(attr_col == e_attr)
        Dv = D[e_attr_row]
        Dv_entropy = entropyForD(Dv, base=base)
        n_Dv_sample = np.size(Dv, axis=0)
        d = {"split_by_attr":e_attr, "data":Dv, "entropy":Dv_entropy, "ratio": n_Dv_sample/n_D_sample}
        Dvs.append(d)

    return Dvs

def infoGain(D, attr_idx, base=2):
    """
    calculate information gain if a data set split by given attribute.
    data set is with labels in last column. attribute is indicated by column index
    example:
              attr0 attr1 attr2 label
        D = [['3'   '1'   '2'   'A']
             ['3'   '5'   '3'   'B']
             ['3'   '6'   '2'   'C']
             ['3'   '7'   '3'   'C']]
        val, Dvs = infoGain(D, 1)

    :param D: data set
    :param attr_idx: index of attribute used to split data set
    :param base: base of log
    :return: information gain value
    :return: a list of dictionary that has 4 key-value pairs.
             keys are: "split_by_attr", one possible value of chosen attribute
                       "data", split sub data set
                       "entropy", entropy of split sub data set
                       "ratio", ratio of number of samples between subset and original data set
    """
    D_ent = entropyForD(D, base=base)
    Dvs = splitDataWithAttribute(D, attr_idx=attr_idx, base=base)

    all_sub_entropy = 0
    for Dv in Dvs:
        n_Dv_sample = np.size(Dv["data"], axis=0)
        all_sub_entropy += (Dv["ratio"] * Dv["entropy"])

    # NOTE: all_sub_entropy can already represent the impurity of a data set, D_ent is
    #       actually redundant
    ig = D_ent - all_sub_entropy
    return ig, Dvs




def gainRatio(D, attr_idx, base=2):
    """
    calculate gain ratio of data set split by attribute.
    example:
          attr0 attr1 attr2 label
    D = [['3'   '1'   '2'   'A']
         ['3'   '5'   '3'   'B']
         ['3'   '6'   '2'   'C']
         ['3'   '7'   '3'   'C']]
    gain_ratio, Dvs = gainRatio(D, 1)
    :param D: data set
    :param attr_idx: index of attribute used to split data set
    :param base: base of log
    :return: gr: gain ratio value
    :return: a list of dictionary that has 4 key-value pairs.
             keys are: "split_by_attr", one possible value of chosen attribute
                       "data", split sub data set
                       "entropy", entropy of split sub data set
                       "ratio", ratio of number of samples between subset and original data set
    """
    ig, Dvs = infoGain(D, attr_idx=attr_idx, base=base)
    ratios = [Dv["ratio"] for Dv in Dvs]
    intrinsicVal = entropy(*ratios, base=base)
    gr = ig / intrinsicVal
    return gr, Dvs


def Gini(D, attr_idx, base=2):
    """
    calculate gini index of a given data set with labels in last column
    example:
          attr0 attr1 attr2 label
    D = [['3'   '1'   '2'   'A']
         ['3'   '5'   '3'   'B']
         ['3'   '6'   '2'   'C']
         ['3'   '7'   '3'   'C']]
    gn, DVs = Gini(D, 1)

    :param D: D: data set
    :param attr_idx: index of attribute used to split data set
    :param base: base of log
    :return: D_Gini: |NEGATIVE| Gini index of data set if split by attribute
                     the higher value, the better split
    :return: a list of dictionary that has 4 key-value pairs.
             keys are: "split_by_attr", one possible value of chosen attribute
                       "data", split sub data set
                       "entropy", entropy of split sub data set
                       "ratio", ratio of number of samples between subset and original data set
    """
    Dvs = splitDataWithAttribute(D, attr_idx=attr_idx, base=base)

    D_Gini = 0
    for Dv in Dvs:
        Dv_label = Dv["data"][:, -1]
        label_uniques = np.unique(Dv_label)

        ratio2_sum = 0
        for e_label in label_uniques:
            e_label_row = np.where(Dv_label == e_label)
            r = np.size(e_label_row) / np.size(Dv["data"], axis=0)
            ratio2_sum += (r*r)

        Dv_Gini = 1 - ratio2_sum
        weighted_Dv_Gini = Dv["ratio"] * Dv_Gini
        D_Gini += weighted_Dv_Gini

    return -D_Gini, Dvs


def count_same_item(a, item):
    return np.size(np.where(a.flat==item))
