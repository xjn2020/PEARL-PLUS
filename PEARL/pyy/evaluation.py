import json
import pickle
import sys

import numpy as np
import os
from collections import defaultdict
import argparse
from collections import Counter

import pandas as pd
from sklearn.metrics import f1_score


def load_tlabels(data_dir):
    # if 'profession' in data_dir or 'hobby' in data_dir:
    with open(os.path.join(data_dir, 'labels.txt' if args.screen else 'labels_pearl.txt'), mode='r', encoding='utf-8') as label_file:
        labels = list(map(lambda x: str(x.strip()), label_file.readlines()))
    # else:
    #     with open(os.path.join(data_dir, 'labels.txt'), mode='r', encoding='utf-8') as label_file:
    #         labels = list(map(lambda x: str(x.strip()), label_file.readlines()))
    return labels


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

# def precision_recall_f1(y_true, y_pred):
#     counter_true = Counter(y_true)
#     counter_pred = Counter(y_pred)
#     counter_intersect = Counter([y_t for y_t, y_p in zip(y_true, y_pred) if y_t == y_p])
#
#     precision = {k: counter_intersect[k] / counter_pred[k] for k in counter_pred.keys()}
#     recall = {k: counter_intersect[k] / counter_true[k] for k in counter_true.keys()}
#
#     f1 = {k: 2 * precision[k] * recall[k] / (precision[k] + recall[k]) for k in precision.keys()}
#
#     return precision, recall, f1
#
# def macro_micro_f1(y_true, y_pred):
#     precision, recall, f1 = precision_recall_f1(y_true, y_pred)
#     micro_precision = sum(precision.values()) / len(precision)
#     micro_recall = sum(recall.values()) / len(recall)
#     micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
#     macro_f1 = sum(f1.values()) / len(f1)
#     return macro_f1, micro_f1

def main(dataset, class_num, E, data_form):
    class_num += 1
    gold_labels = load_tlabels('./data/' + dataset)
    # if data_form is None:
    #     s = 'info'
    # else:
    #     s = 'info_' + data_form
    if args.data_form is None:
        s = f'info_{args.niter}_{args.num_keywords}'
    else:
        s = f'info_{args.data_form}'
    print(s)
    for idx in range(1, E + 1):
        print(f'################### {idx}/{E} ###################')
        repr_probility = np.loadtxt('./data/' + dataset + f'/' + s + f'/model/k{class_num}.{idx}.pz_d')[
                         :len(gold_labels)]
        # ACC
        true_num, repr_prediction = 0, np.argmax(repr_probility, axis=1)
        for i in range(len(gold_labels)):
            curr_golds = [int(i) for i in gold_labels[i].split(" ")]
            if repr_prediction[i] in curr_golds:
                true_num = true_num + 1
        print("acc", float(true_num / len(gold_labels)))
        if args.dataset == 'profession' or args.dataset == 'hobby':
            # nDCG
            score, gold_set = 0, set([])
            for i in range(len(gold_labels)):
                index_list = list(np.argsort(-repr_probility[i]))
                curr_golds = [int(i) for i in gold_labels[i].split(" ")]
                ranks = np.zeros(class_num)
                for gold in curr_golds:
                    gold_set.add(gold)
                    gold_index = index_list.index(gold)
                    ranks[gold_index] = 1
                score = score + ndcg_at_k(ranks, 1000)
            print("ndcg", score / len(gold_labels))

            # MRR
            big_count, big_MRR = 0, 0
            prof_dict = defaultdict(lambda: [0.0, 0])
            for i in range(len(gold_labels)):
                index_list = list(np.argsort(-repr_probility[i]))
                curr_golds = [int(i) for i in gold_labels[i].split(" ")]
                for gold in curr_golds:
                    gold_index = index_list.index(gold)
                    imrr = 1.0 / (gold_index + 1)
                    prof_dict[gold][0] += imrr
                    prof_dict[gold][1] += 1
            for prof, stats in prof_dict.items():
                big_count += 1
                big_MRR += float(stats[0] / stats[1])
            print("mrr", big_MRR / big_count)
        else:
            true_labels = [int(i) for i in gold_labels]
            pre_labels = [int(i) for i in repr_prediction]
            print('micro f1', f1_score(true_labels, pre_labels, average='micro'))
            print('macro f1', f1_score(true_labels, pre_labels, average='macro'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='profession')
    parser.add_argument("--class-num", type=int, default=71)
    parser.add_argument("--E", type=int, default=2)  # E in paper
    parser.add_argument("--data-form", type=str, default=None)
    parser.add_argument("--screen", action='store_true')
    parser.add_argument("--num_keywords", type=int, default=60) # how many keywords in one doc
    parser.add_argument("--niter", type=int, default=50)
    args = parser.parse_args()

    # print(vars(args))
    main(args.dataset, args.class_num, args.E, args.data_form)
