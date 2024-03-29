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
from matplotlib import image

from config import *


def plot_records(records):
    records_plot = np.zeros([int(MAX_ITERATION / TRACK_FREQ), 6])
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
    avg_fitness = records_plot[:, 3]
    micro_pop = records_plot[:, 4]
    macro_pop = records_plot[:, 5]
    plt.plot(iterations, train_f, label='train f-score')
    plt.plot(iterations, test_f, label='test f-score')
    plt.plot(iterations, avg_fitness, label='avg-fitness')
    plt.plot(iterations, micro_pop, label='micro-pop(%)')
    plt.plot(iterations, macro_pop, label='macro-pop(%)')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend(loc=2, prop={'size': 8})
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


def plot_heatmap(sim_matrix, label_ref):
    # sim_matrix[sim_matrix <= 0.1] = 0
    annotations = list(label_ref.values())
    fig, ax = plt.subplots()
    im = ax.imshow(sim_matrix, cmap='YlGn', interpolation='nearest')
    # ax.set_title('Pairwise label similarities', alpha=0.7)
    ax.set_xticks(np.arange(annotations.__len__()))
    ax.set_yticks(np.arange(annotations.__len__()))
    ax.set_xticklabels(annotations, alpha=0.9, fontsize=16)
    ax.set_yticklabels(annotations, alpha=0.9, fontsize=16)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.colorbar(im, ax=ax, alpha=1)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    fig.tight_layout()
    plt.savefig(os.path.join(os.path.curdir, REPORT_PATH, DATA_HEADER, 'heat-map.png'))
    plt.close()


def plot_graph(label_clusters, sim_matrix, label_ref):
    fig, axis = plt.subplots()
    for edge, spine in axis.spines.items():
        spine.set_visible(False)
    labels = set()
    for cl in label_clusters.values():
        labels = labels.union(cl)
    labels = sorted(list(labels))
    graph = nx.Graph()
    edge_list = []

    sim_matrix[sim_matrix < 0.1] = 0

    def same_cluster(l1, l2):
        for cluster in label_clusters.values():
            if l1 in cluster and l2 in cluster:
                return True
        return False

    for c1 in range(labels.__len__()):
        for c2 in range(c1 + 1, labels.__len__()):
            w = sim_matrix[c1, c2]
            if w > 0 and same_cluster(labels[c1], labels[c2]):
                edge_list.append((labels[c1], labels[c2]))
                graph.add_weighted_edges_from([(labels[c1], labels[c2], w)])
            else:
                graph.add_node(labels[c1])
                graph.add_node(labels[c2])
    if graph.number_of_nodes() <= 1:
        return
    try:
        pos = nx.planar_layout(graph)
    except nx.NetworkXException:
        pos = nx.spring_layout(graph)
    node_color = ['xkcd:soft green', 'c', 'y' 'g', 'm', 'b', 'k']
    for k in label_clusters.keys():
        nx.draw_networkx_nodes(graph, pos,
                               node_color=node_color[k],
                               nodelist=list(label_clusters[k]),
                               node_size=2000)

    edge_weights = nx.get_edge_attributes(graph, 'weight')
    edge_weights = {k: round(v, 3) for k, v in edge_weights.items()}
    names = {k: label_ref[k] for k in labels}
    nx.draw_networkx_labels(graph, pos, names, font_size=20, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, edge_list=edge_list, width=1, alpha=0.7)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_weights, font_size=16, alpha=0.9)
    # fig.suptitle('Label similarity graph', fontsize=14)
    # axis.set_title('Label similarity graph', fontsize=14)
    plt.show()


def plot_image(image_id, labels, vote, label_ref):
    fig1, axis = plt.subplots()
    axis.set_title('Test image' + image_id)
    for edge, spine in axis.spines.items():
        spine.set_visible(False)
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
    plt.axis('off')
    try:
        pixels = image.imread(os.path.join(DATA_DIR, DATA_HEADER, 'images_dir', image_id + '.png'))
        axis.imshow(pixels)
    except FileNotFoundError:
        pass
    print('Ground truth: ', [label_ref[label] for label in labels])
    print('Ranked prediction: ')
    for k in sorted(vote, key=vote.get, reverse=True):
        if vote[k] > 0:
            print(label_ref[k], round(vote[k], 4))
    plt.show()
