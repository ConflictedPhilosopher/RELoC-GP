# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
import os.path
import random

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from config import *


def plot_records(records):
    records_plot = np.zeros([int(MAX_ITERATION / TRACK_FREQ), 3])
    for i in range(records.__len__()):
        record = records[i]
        for j in range(int(MAX_ITERATION / TRACK_FREQ)):
            try:
                records_plot[j] += record[j]
            except IndexError:
                records_plot[j] += record[-1]
    records_plot /= float(records.__len__())

    iterations = range(TRACK_FREQ, MAX_ITERATION + TRACK_FREQ, TRACK_FREQ)
    train_f = records_plot[:, 1]
    test_f = records_plot[:, 2]
    plt.plot(iterations, train_f, label='train f-score')
    plt.plot(iterations, test_f, label='test f-score')
    plt.xlabel('Iteration')
    plt.ylabel('F-score')
    plt.legend()
    fig_name = str(MAX_CLASSIFIER) + '-' + str(PROB_HASH) + '.png'
    plt.savefig(os.path.join(os.path.curdir, REPORT_PATH, DATA_HEADER, fig_name), bbox_inches='tight')
    plt.close()


def plot_bar(value_dict, title):
    fig, ax = plt.subplots(figsize=(16, 9))
    xaxis = list(value_dict.values())
    yaxis = list(value_dict.keys())
    ax.barh(yaxis, xaxis)
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)
    ax.set_title('class-specific ' + title, loc='right', )
    plt.savefig(os.path.join(os.path.curdir, REPORT_PATH, DATA_HEADER, title+'.png'))
    plt.close()


def plot_graph(label_clusters, label_matrix, sim_matrix, label_ref):
    labels = set()
    for cl in label_clusters.values():
        labels = labels.union(cl)
    labels = list(labels)
    graph = nx.Graph()
    edge_list = []

    def same_cluster(l1, l2):
        for cluster in label_clusters.values():
            if l1 in cluster and l2 in cluster:
                return True
        return False

    for c1 in range(labels.__len__()):
        for c2 in range(c1 + 1, labels.__len__()):
            edge_exists = np.dot(label_matrix[:, c1], label_matrix[:, c2]) > 0
            if edge_exists and same_cluster(labels[c1], labels[c2]):
                edge_list.append((labels[c1], labels[c2]))
                w = sim_matrix[c1, c2]
                graph.add_weighted_edges_from([(labels[c1], labels[c2], w)])
            else:
                graph.add_node(labels[c1])
                graph.add_node(labels[c2])

    fig1, ax1 = plt.subplots()
    ax1.set_title('Label similarity graph')
    try:
        pos = nx.planar_layout(graph)
    except nx.NetworkXException:
        pos = nx.spring_layout(graph)
    node_color = ['g', 'm', 'c', 'b', '']
    for k in label_clusters.keys():
        nx.draw_networkx_nodes(graph, pos,
                               node_color=node_color[k],
                               nodelist=list(label_clusters[k]),
                               node_size=300)

    edge_weights = nx.get_edge_attributes(graph, 'weight')
    edge_weights = {k: round(v, 3) for k, v in edge_weights.items()}
    names = {k: label_ref[k] for k in labels}
    nx.draw_networkx_labels(graph, pos, names, font_size=11)
    nx.draw_networkx_edges(graph, pos, edge_list=edge_list, width=1, alpha=0.5)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights)
    plt.show()
    plt.close()


def plot_image(image_id, labels, prediction):
    print('test image with labels')
    print(image_id, labels, prediction)
