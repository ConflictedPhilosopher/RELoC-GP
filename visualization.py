# Shabnam Nazmi.
# Graduate research assistant at electrical and computer engineering department,
# North Carolina A&T State University, Greensboro, NC.
# snazmi@aggies.ncat.edu.
#
# ------------------------------------------------------------------------------
import os.path

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


def plot_graph(label_clusters, sim_matrix):
    labels = set()
    labels = [labels.union(l) for l in label_clusters.values()]
    G = nx.Graph()
    edge_list = []
    for c1 in range(labels.__len__()):
        for c2 in range(c1 + 1, labels.__len__()):
            edge_exists = np.dot(labels[:, c1], labels[:, c2]) > 0
            if edge_exists:
                edge_list.append((c1, c2))
                w = sim_matrix[c1, c2]
                G.add_weighted_edges_from([(c1, c2, w)])
            else:
                G.add_node(c1)
                G.add_node(c2)

    fig1, ax1 = plt.subplots()
    ax1.set_title('Label similarity graph')
    pos = nx.spring_layout(G)
    for k in label_clusters.keys():
        nx.draw_networkx_nodes(G, pos,
                               node_color=np.random.rand(3, ),
                               nodelist=label_clusters[k],
                               )

    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    nx.draw_networkx_edges(G, pos, edge_list=edge_list, width=1, alpha=0.5)
    plt.show()
    # plt.savefig(os.path.join(os.path.curdir, REPORT_PATH, DATA_HEADER, 'similarity-graph.png'))
    # plt.close()


def plot_image(image_id, labels, prediction):
    print('test image with labels')
    print(image_id, labels, prediction)
