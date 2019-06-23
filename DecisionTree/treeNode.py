import numpy as np
import dataset as ds
import utils as u

class DTNode(object):
    def __init__(self, attr, attr_vals, depth):
        self.attr = attr
        self.attr_vals = attr_vals
        self.depth = depth
        self.parent = None
        self.children = {}
        for attr_val in self.attr_vals:
            self.children[attr_val] = None

    def __str__(self):
        s = 'node(attr:{}, depth:{} | split path: '.format(self.attr, self.depth)
        for i, val in enumerate(self.attr_vals):
            s += "[{}], ".format(val, i)
        s = s[:-2] + ')'
        return s

    def set_parent(self, parent):
        self.parent = parent

    def attach_child(self, parent_attr_val, child):
        # child can be a DTNode or a final result label
        self.children[parent_attr_val] = child
        if(isinstance(child, DTNode)):
            child.set_parent(self)

    def predict(self, sample):
        # get sample's attribute value based on attribute selected by this node
        samp_attr_val = sample[self.attr]
        if(not isinstance(self.children[samp_attr_val], DTNode)):
            # leaf node
            return self.children[samp_attr_val]
        else:
            return self.children[samp_attr_val].predict(sample)



class DecisionTree(object):
    def __init__(self, D, criteria="Gini", base=2):
        self.D = D
        self.base = base
        self.n_row = np.size(D, axis=0)
        self.n_col = np.size(D, axis=1)
        self.criteria = criteria
        self.attr_vals = []
        for col in [self.D[:, c] for c in range(self.n_col-1)]:
            self.attr_vals.append(np.unique(col))

    def find_best_split_attr(self, D, used_attr=None):
        # find best attribute in given data set D
        def all_scores(attr_condidates, crt_func):
            best_score = float("-inf")
            for attr_idx in attr_condidates:
                score, Dvs = crt_func(D, attr_idx=attr_idx, base=self.base)
                if(score > best_score):
                    best_score = score
                    best_Dvs = Dvs
                    best_attr = attr_idx
            return best_attr, best_Dvs

        # remove used attributes
        all_attrs = list(range(self.n_col - 1))
        if(used_attr is not None):
            for used in used_attr:
                all_attrs.remove(used)

        # find best attribute and it's sub data set
        if (self.criteria == "Gini"):
            best_attr, Dvs = all_scores(all_attrs, u.Gini)
        elif (self.criteria == "InfoGain"):
            best_attr, Dvs = all_scores(all_attrs, u.infoGain)
        elif (self.criteria == "GainRatio"):
            # find info gain >= info gain average
            igs = []
            for a in all_attrs:
                igs.append(u.infoGain(D, a, base=self.base)[0])
            ig_avg = np.sum(igs) / len(all_attrs)
            all_attrs = [all_attrs[i] for i in np.where(igs >= ig_avg)[0]]
            # find the best attribute to split
            best_attr, Dvs = all_scores(all_attrs, u.gainRatio)

        return best_attr, Dvs

    def create_node(self, parent, parent_attr_val, Dv_of_attr_val, used_attr, depth):
        """
        based on: 1. parent node
                  2. parent node's split paths
                  3. sub data set belongs to parent node's split path
                  3. used attributes in upper level
        to create a sub node in that split path
        :param parent: parent node
        :param parent_attr_val: parent node's split path for next level
        :param Dv_of_attr_val: data set belongs to that split path
        :param used_attr: all attributes that have been used by previous levels
        :param depth: depth of created sub node
        :return: parent node with all sub nodes already attached
        """
        if(len(used_attr)==self.n_col):
            # no attr can be used to create new node
            # in this case, see sample ratio in Dv_of_attr_val
            node = self.best_label_of_impure_D(Dv_of_attr_val)
            parent.attach_child(parent_attr_val, node)
            return parent
        elif(len(np.unique(Dv_of_attr_val[:, -1]))==1):
            # Dv_of_attr_val is pure, just pick a label from sample
            node = Dv_of_attr_val[0, -1]
            parent.attach_child(parent_attr_val, node)
            return parent
        else:
            # need to create a sub node
            best_attr, Dvs = self.find_best_split_attr(Dv_of_attr_val, used_attr)
            node = DTNode(best_attr, self.attr_vals[best_attr], depth)
            parent.attach_child(parent_attr_val, node)

            # back tracking used attribute
            used_attr.append(node.attr)
            for Dv in Dvs:
                self.create_node(node, Dv['split_by_attr'], Dv['data'], used_attr, depth+1)
            used_attr.remove(node.attr)

            return parent

    def best_label_of_impure_D(self, D):
        D_labels = D[:, -1]
        unique, cnt = np.unique(D_labels, return_counts=True)
        max_cnt = np.max(cnt)
        if(u.count_same_item(cnt, max_cnt)> 1):
            # if there are 2 best labels in impure data set
            # go see which label has the most samples in original data set
            best_labels = unique[np.where(cnt==max_cnt)]
            best_labels_cnt = []
            all_labels = self.D[:, -1]
            for bl in best_labels:
                best_labels_cnt.append(u.count_same_item(all_labels, bl))
            return best_labels[np.argmax(best_labels_cnt)]
        else:
            return unique[np.argmax(cnt)]

    def fit(self, D):
        # create root node
        best_attr, Dvs = self.find_best_split_attr(D)
        root = DTNode(best_attr, np.unique(D[:, best_attr]), 0)
        used_attr = []
        # create all sub nodes below root node
        used_attr.append(root.attr)
        for Dv in Dvs:
            root = self.create_node(root, Dv['split_by_attr'], Dv['data'], used_attr, 1)

        return root



def traverse(root):
    """
    traverse and display entire decision tree
    :param root: root node
    :return: None
    """
    if(root is not None):
        if(root.depth==0):
            print("---------------------------------------------")
        print(root)
        for path_val, child in root.children.items():
            print("---------------- path %s(depth:%0d) ----------------" % (path_val, root.depth))
            if(isinstance(child, DTNode)):
                traverse(child)
            else:
                print(child, "   *depth:", root.depth+1)




# dt = DecisionTree(ds.tD, criteria='GainRatio')
# root = dt.fit(ds.tD)

# traverse(root)


# for r in range(np.size(ds.D, axis=0)):
#   samp = ds.D[r, :-1]
#   pred = root.predict(samp)
#   if(pred != ds.D[r, -1]):
#       print("wo cao!")
#   else:
#       print("Niu B!")

