import logging
import sys
import time
from collections import Counter
from datetime import timedelta
from typing import Set, List, Tuple, Optional
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pickle as pkl
import pdb
from utils import * 
import argparse

parser = argparse.ArgumentParser(description='Non-RL')

class ECTS(EarlyClassifier):

    """
    The ECTS algorithm.

    Publications:

    Early classification on time series(2012)
    """

    def __init__(self, timestamps, support: float, relaxed: bool):
        """
        Creates an ECTS instance.

        :param timestamps: a list of timestamps for early predictions
        :param support: minimum support threshold
        :param relaxed: whether we use the Relaxed version or the normal
        """
        self.rnn: Dict[int, Dict[int, List]] = dict()
        self.nn: Dict[int, Dict[int, List]] = dict()
        self.data: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None
        self.mpl: Dict[int, Optional[int]] = dict()
        self.timestamps = timestamps
        self.support = support
        self.clusters: Dict[int, List[int]] = dict()
        self.occur: Dict[int, int] = dict()
        self.relaxed = relaxed
        self.correct: Optional[List[Optional[int]]] = None

    def train(self, train_data: pd.DataFrame, labels: Sequence[int]) -> None:

        """
        Function that trains the model using Agglomerating Hierarchical clustering

        :param train_data: a Dataframe containing-series
        :param labels: a Sequence containing the labels of the data
        """
        self.data = train_data

        self.labels = labels
        if self.relaxed:
            self.__leave_one_out()
        for index, value in self.labels.value_counts().items():
            self.occur[index] = value

        # Finding the RNN of each item
        time_pos = 0
        for e in self.timestamps:
            product = self.__nn_non_cluster(time_pos)  # Changed to timestamps position
            self.rnn[e] = product[1]
            self.nn[e] = product[0]
            time_pos += 1
        import pdb
        pdb.set_trace()
        temp = {}
        finished = {}  # Dictionaries that signifies if an mpl has been found
        for e in reversed(self.timestamps):
            for index, row in self.data.iterrows():
                if index not in temp:
                    self.mpl[index] = e
                    finished[index] = 0  # Still MPL is not found

                else:
                    if finished[index] == 1:  # MPL has been calculated for this time-series so nothing to do here
                        continue

                    if self.rnn[e][index] is not None:
                        self.rnn[e][index].sort()
                    # Sorting it in order to establish that the RNN is in the same order as the value
                    if temp[index] is not None:
                        temp[index].sort()

                    if self.rnn[e][index] == temp[index]:  # Still going back the timestamps
                        self.mpl[index] = e

                    else:  # Found k-1
                        finished[index] = 1  # MPL has been found!
                temp[index] = self.rnn[e][index]
        self.__mpl_clustering()

    def __leave_one_out(self):
        nn = []
        for index, row in self.data.iterrows():  # Comparing each time-series

            data_copy = self.data.copy()
            data_copy = data_copy.drop(data_copy.index[index])
            for index2, row2 in data_copy.iterrows():
                temp_dist = distance.euclidean(row, row2)

                if not nn:
                    nn = [(self.labels[index2], temp_dist)]
                elif temp_dist >= nn[0][1]:
                    nn = [(self.labels[index2], temp_dist)]
            if nn[0][0] == self.labels[index]:
                if not self.correct:
                    self.correct = [index]
                else:
                    self.correct.append(index)
            nn.clear()

    def __nn_non_cluster(self, prefix: int):
        """Finds the NN of each time_series and stores it in a dictionary

        :param prefix: the prefix with which we will conduct the NN

        :return: two dicts holding the NN and RNN"""
        nn = {}
        rnn = {}
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        neigh = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(self.data.iloc[:, 0:prefix + 1])
        def something(row):
            return neigh.kneighbors([row])

        result_data = self.data.iloc[:, 0:prefix + 1].apply(something, axis=1)
        for index, value in result_data.items():
            if index not in nn:
                nn[index] = []
            if index not in rnn:
                rnn[index] = []
            for item in value[1][0]:
                if item != index:
                    nn[index].append(item)
                    if item not in rnn:
                        rnn[item] = [index]
                    else:
                        rnn[item].append(index)
        return nn, rnn

    def __cluster_distance(self, cluster_a: Sequence[int], cluster_b: Sequence[int]):
        """
        Computes the distance between two clusters as the minimum among all
        inter-cluster pair-wise distances.

        :param cluster_a: a cluster
        :param cluster_b: another cluster
        :return: the distance
        """

        min_distance = float("inf")
        for i in cluster_a:
            for j in cluster_b:
                d = distance.euclidean(self.data.iloc[i], self.data.iloc[j])
                if min_distance > d:
                    min_distance = d

        return min_distance

    def nn_cluster(self, cl_key: int, cluster_index: Sequence[int]):
        """Finds the nearest neighbor to a cluster
        :param cluster_index: List of indexes contained in the list
        :param cl_key: The key of the list in the cluster dictionary
        """
        global x
        dist = float("inf")
        candidate = []  # List that stores multiple candidates

        for key, value in self.clusters.items():  # For each other cluster

            if cl_key == key:  # Making sure its a different to our current cluster
                continue
            temp = self.__cluster_distance(cluster_index, value)  # Find their Distance

            if dist > temp:  # If its smaller than the previous, store it
                dist = temp
                candidate = [key]

            elif dist == temp:  # If its the same, store it as well
                candidate.append(key)
        x-=1
        return candidate

    def __rnn_cluster(self, e: int, cluster: List[int]):
        """
        Calculates the RNN of a cluster for a certain prefix.

        :param e: the prefix for which we want to find the RNN
        :param cluster: the cluster that we want to find the RNN
        """

        rnn = set()
        complete = set()
        for item in cluster:
            rnn.union(self.rnn[e][item])
        for item in rnn:
            if item not in cluster:
                complete.add(item)
        return complete

    def __mpl_calculation(self, cluster: List[int]):
        """Finds the MPL of discriminative clusters

        :param cluster: The cluster of which we want to find it's MPL"""
        # Checking if the support condition is met
        index = self.labels[cluster[0]]
        if self.support > len(cluster) / self.occur[index]:
            return
        mpl_rnn = self.timestamps[len(
            self.timestamps) - 1]  # Initializing the  variables that will indicate the minimum timestamp from which each rule applies
        mpl_nn = self.timestamps[len(self.timestamps) - 1]
        """Checking the RNN rule for the clusters"""

        curr_rnn = self.__rnn_cluster(self.timestamps[len(self.timestamps) - 1], cluster)  # Finding the RNN for the L

        if self.relaxed:
            curr_rnn = curr_rnn.intersection(self.correct)

        for e in reversed(self.timestamps):

            temp = self.__rnn_cluster(e, cluster)  # Finding the RNN for the next timestamp
            if self.relaxed:
                temp = temp.intersection(self.correct)

            if not curr_rnn - temp:  # If their division is an empty set, then the RNN is the same so the
                # MPL is e
                mpl_rnn = e
            else:
                break
            curr_rnn = temp

        """Then we check the 1-NN consistency"""
        rule_broken = 0
        for e in reversed(self.timestamps):  # For each timestamp

            for series in cluster:  # For each time-series

                for my_tuple in self.nn[e][series]:  # We check the corresponding NN to the series
                    if my_tuple not in cluster:
                        rule_broken = 1
                        break
                if rule_broken == 1:
                    break
            if rule_broken == 1:
                break
            else:
                mpl_nn = e
        for series in cluster:
            pos = max(mpl_rnn, mpl_nn)  # The value at which at least one rule is in effect
            if self.mpl[series] > pos:
                self.mpl[series] = pos

    def __mpl_clustering(self):
        """Executes the hierarchical clustering"""
        pool = Pool(mp.cpu_count())
        n = self.data.shape[0]
        redirect = {}  # References an old cluster pair candidate to its new place
        discriminative = 0  # Value that stores the number of discriminative values found
        """Initially make as many clusters as there are items"""
        for index, row in self.data.iterrows():
            self.clusters[index] = [index]
            redirect[index] = index
        result = []
        """Clustering loop"""
        while n > 1:  # For each item
            closest = {}
            my_list = list(self.clusters.items())
            res = pool.starmap(self.nn_cluster, my_list)
            for key,p  in zip(self.clusters.keys(),res):
                closest[key] = p
            logger.debug(closest)
            for key, value in closest.items():
                for item in list(value):
                    if key in closest[item]:  # Mutual pair found
                        closest[item].remove(key)
                        if  redirect[item]==redirect[key]:      #If 2 time-series are in the same cluster(in case they had an 3d  neighboor that invited them in the cluster)
                            continue
                        for time_series in self.clusters[redirect[item]]:
                            self.clusters[redirect[key]].append(time_series)  # Commence merging
                        del self.clusters[redirect[item]]
                        n = n - 1
                        redirect[item] = redirect[key]  # The item can now be found in another cluster
                        for element in self.clusters[redirect[key]]:  # Checking if cluster is discriminative
                            result.append(self.labels.loc[element])

                        x = np.array(result)
                        if len(np.unique(x)) == 1:  # If the unique class labels is 1, then the
                            # cluster is discriminative
                            discriminative += 1
                            self.__mpl_calculation(self.clusters[redirect[key]])

                        for neighboors_neigboor in closest:  # The items in the cluster that has been assimilated can
                            # be found in the super-cluster
                            if redirect[neighboors_neigboor] == item:
                                redirect[neighboors_neigboor] = key
                        result.clear()
            if discriminative == 0:  # No discriminative clusters found
                break
            discriminative = 0
        pool.terminate()
        
    def predict(self, test_data: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Prediction phase.
        Finds the 1-NN of the test data and if the MPL oof the closest time-series allows the prediction, then return that prediction
         """
        predictions = []
        nn = []
        candidates = []  # will hold the potential predictions
        cand_min_mpl = []
        #test_data = test_data.rename(columns=lambda x: x - 1)
        for test_index, test_row in test_data.iterrows():
            for e in self.timestamps:
                neigh = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(self.data.iloc[:, 0:e + 1])
                neighbors = neigh.kneighbors([test_row[0:e + 1]])
                candidates.clear()
                cand_min_mpl.clear()
                nn = neighbors[1]
                for i in nn:
                    if e >= self.mpl[i[0]]:
                        candidates.append((self.mpl[i[0]], self.labels[i[0]]))  # Storing candidates by mpl and by label
                if len(candidates) > 1:  # List is not empty so wee found candidates
                    candidates.sort(key=lambda x: x[0])
                    for candidate in candidates:

                        if candidate[0] == candidates[0][0]:
                            cand_min_mpl.append(candidate)  # Keeping the candidates with the minimum mpl
                        else:
                            break  # From here on the mpl is going to get bigger

                    predictions.append(
                        (e, max(set(cand_min_mpl), key=cand_min_mpl.count)))  # The second argument is the max label
                    break
                elif len(candidates) == 1:  # We don't need to to do the above if we have only one nn
                    predictions.append((e, candidates[0][1]))
                    break
            if candidates == 0:
                predictions.append((self.timestamps[-1], 0))
        return predictions

def train_and_test(config: Config, classifier: EarlyClassifier) -> None:
    predictions = []

    start = time.time()
    trip = classifier.train(config.train_data[i], config.train_labels)
    print('Total training time := {}'.format(timedelta(seconds=time.time() - start)),
                file=config.output)

    # Make predictions
    start = time.time()
    votes.append(classifier.predict(config.test_data[i]))
    print('Total testing time := {}'.format(timedelta(seconds=time.time() - start)),
                file=config.output)

    # Make predictions from the votes of each test example
    for i in range(len(votes[0])):
        max_timestamp = max(map(lambda x: x[i][0], votes))
        most_predicted = Counter(map(lambda x: x[i][1], votes)).most_common(1)[0][0]
        predictions.append((max_timestamp, most_predicted))

    accuracy = utils.accuracy(predictions, config.test_labels.tolist())
    earliness = utils.earliness(predictions, config.ts_length - 1)
    harmonic = utils.harmonic_mean(accuracy, earliness)

    print('Accuracy: ' + str(round(accuracy, 4)) + ' Earliness: ' + str(round(earliness * 100, 4)) + '%',
               file=config.output)
    print('Harmonic mean: ' + str(round(harmonic, 4)),
               file=config.output)

def cv(config: Config, classifier: EarlyClassifier) -> None:
    sum_accuracy, sum_earliness, sum_precision, sum_recall, sum_f1 = 0, 0, 0, 0, 0
    all_predictions: List[Tuple[int, int]] = list()
    all_labels: List[int] = list()
    if config.splits:
        ind = []
        for key in config.splits.keys():
            ind.append((config.splits[key][0], config.splits[key][1]))
        indices = zip(ind, range(1, config.folds + 1))
    else:
        print("Folds : {}".format(config.folds))
        indices = zip(StratifiedKFold(config.folds).split(config.cv_data[0], config.cv_labels),
                      range(1, config.folds + 1))
    count = 0
    for ((train_indices, test_indices), i) in indices:
        predictions = []
        count += 1
        print('== Fold ' + str(i), file=config.output)
        

        """In case of a multivariate cv dataset is passed on one of the univariate based approaches"""
        votes = []
        for ii in range(config.variate):
            pdb.set_trace()
            fold_train_data = config.cv_data[ii].iloc[train_indices].reset_index(drop=True) ##(4199,30)
            fold_train_labels = config.cv_labels[train_indices].reset_index(drop=True) ## (4199,)
            fold_test_data = config.cv_data[ii].iloc[test_indices].reset_index(drop=True) ## (1050,30)

            if config.java is True:
                """ For the java approaches"""
                temp = pd.concat([fold_train_labels, fold_train_data], axis=1, sort=False)
                temp.to_csv('train', index=False, header=False, sep=delim_1)

                temp2 = pd.concat([config.cv_labels[test_indices].reset_index(drop=True), fold_test_data], axis=1,
                                    sort=False)
                temp2.to_csv('test', index=False, header=False, sep=delim_2)
                res = classifier.predict(pd.DataFrame())  # The java methods return the tuple (predictions,
                # train time, test time)

                print('Total training time := {}'.format(timedelta(seconds=float(res[1]))),
                            file=config.output)
                print('Total testing time := {}'.format(timedelta(seconds=float(res[2]))),
                            file=config.output)
                votes.append(res[0])

            elif config.cplus is True:

                fold_test_labels = config.cv_labels[test_indices].reset_index(drop=True)
                classifier.train(fold_train_data, fold_train_labels)
                a = fold_train_labels.value_counts()
                a = a.sort_index(ascending=False)

                # The EDSC method returns the tuple (predictions, train time, test time)
                res = classifier.predict2(test_data=fold_test_data, labels=fold_test_labels, numbers=a, types=0)

                print('Total training time := {}'.format(timedelta(seconds=float(res[1]))),
                            file=config.output)
                print('Total testing time := {}'.format(timedelta(seconds=float(res[2]))),
                            file=config.output)
                votes.append(res[0])
            else:
                # Train the classifier
                start = time.time()
                classifier.train(fold_train_data, fold_train_labels)
                print('Total training time := {}'.format(timedelta(seconds=time.time() - start)),
                            file=config.output)

                # Make predictions
                start = time.time()
                votes.append(classifier.predict(fold_test_data))
                print('Total testing time := {}'.format(timedelta(seconds=time.time() - start)),
                            file=config.output)

        # Make predictions from the votes of each test example
        for ii in range(len(votes[0])):
            max_timestamp = max(map(lambda x: x[ii][0], votes))
            most_predicted = Counter(map(lambda x: x[ii][1], votes)).most_common(1)[0][0]
            predictions.append((max_timestamp, most_predicted))
              
        
        all_predictions.extend(predictions)
        all_labels.extend(config.cv_labels[test_indices])

        # Calculate accuracy and earliness
        accuracy = utils.accuracy(predictions, config.cv_labels[test_indices].tolist())
        sum_accuracy += accuracy
        earliness = utils.earliness(predictions, config.ts_length - 1)
        sum_earliness += earliness
        print('Accuracy: ' + str(round(accuracy, 4)) + ' Earliness: ' + str(round(earliness * 100, 4)) + '%',
                   file=config.output)
        harmonic = utils.harmonic_mean(accuracy, earliness)
        print('Harmonic mean: ' + str(round(harmonic, 4)),
                file=config.output)
        
        # Calculate counts, precision, recall and f1-score if a target class is provided
        if config.target_class == -1:
            items = config.cv_labels[train_indices].unique()
            for item in items:
                print('For the class: ' + str(item), file=config.output)
                tp, tn, fp, fn = utils.counts(item, predictions, config.cv_labels[test_indices].tolist())
                print('TP: ' + str(tp) + ' TN: ' + str(tn) + ' FP: ' + str(fp) + ' FN: ' + str(fn),
                           file=config.output)
                precision = utils.precision(tp, fp)
                print('Precision: ' + str(round(precision, 4)), file=config.output)
                recall = utils.recall(tp, fn)
                print('Recall: ' + str(round(recall, 4)), file=config.output)
                f1 = utils.f_measure(tp, fp, fn)
                print('F1-score: ' + str(round(f1, 4)) + "\n", file=config.output)
        elif config.target_class:
            tp, tn, fp, fn = utils.counts(config.target_class, predictions, config.cv_labels[test_indices].tolist())
            print('TP: ' + str(tp) + ' TN: ' + str(tn) + ' FP: ' + str(fp) + ' FN: ' + str(fn), file=config.output)
            precision = utils.precision(tp, fp)
            sum_precision += precision
            print('Precision: ' + str(round(precision, 4)), file=config.output)
            recall = utils.recall(tp, fn)
            sum_recall += recall
            print('Recall: ' + str(round(recall, 4)), file=config.output)
            f1 = utils.f_measure(tp, fp, fn)
            sum_f1 += f1
            print('F1-score: ' + str(round(f1, 4)), file=config.output)
        print('Predictions' + str(predictions), file=config.output)
    print('== Macro-average', file=config.output)
    macro_accuracy = sum_accuracy / config.folds
    macro_earliness = sum_earliness / config.folds
    print('Accuracy: ' + str(round(macro_accuracy, 4)) +
               ' Earliness: ' + str(round(macro_earliness * 100, 4)) + '%',
               file=config.output)

    if config.target_class and config.target_class != -1:
        macro_precision = sum_precision / config.folds
        macro_recall = sum_recall / config.folds
        macro_f1 = sum_f1 / config.folds
        print('Precision: ' + str(round(macro_precision, 4)), file=config.output)
        print('Recall: ' + str(round(macro_recall, 4)), file=config.output)
        print('F1-score: ' + str(round(macro_f1, 4)), file=config.output)

    print('== Micro-average:', file=config.output)
    micro_accuracy = utils.accuracy(all_predictions, all_labels)
    micro_earliness = utils.earliness(all_predictions, config.ts_length - 1)
    print('Accuracy: ' + str(round(micro_accuracy, 4)) +
               ' Earliness: ' + str(round(micro_earliness * 100, 4)) + '%',
               file=config.output)

    # Calculate counts, precision, recall and f1-score if a target class is provided
    if config.target_class and config.target_class != -1:
        tp, tn, fp, fn = utils.counts(config.target_class, all_predictions, all_labels)
        print('TP: ' + str(tp) + ' TN: ' + str(tn) + ' FP: ' + str(fp) + ' FN: ' + str(fn), file=config.output)
        precision = utils.precision(tp, fp)
        print('Precision: ' + str(round(precision, 4)), file=config.output)
        recall = utils.recall(tp, fn)
        print('Recall: ' + str(round(recall, 4)), file=config.output)
        f1 = utils.f_measure(tp, fp, fn)
        print('F1-score: ' + str(round(f1, 4)), file=config.output)


def ects(config: Config, support: float, relaxed: bool) -> None:
    """
     Run 'ECTS' algorithm.
    """

    # logger.info("Running ECTS ...")

    classifier = ECTS(config.timestamps, support, relaxed)

    if config.cv_data is not None:  ## 交叉验证，执行的是这个分支
        # pdb.set_trace()
        cv(config, classifier) 
    else:
        # pdb.set_trace()
        train_and_test(config, classifier)

if __name__ == "__main__":
    ects()
