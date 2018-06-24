import numpy as np
import pickle
import pandas as pd
from scipy import stats
import scipy.ndimage
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import community
import difflib
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
import os
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import itertools


def build_graphs_each_team(df_all_stats, curr_year, curr_team):
    """
    For any team on any particular year build a graph where all of the nodes are individual players
    and the edges connecting players is a ratio between player A's  assist rate and player B's
    scoring total (Scoring Potential)
    edges = max{(AST person A) * (PS/G person B), (AST person B) * (PS/G person A)}
    :param df_all_stats: dataFrame with all stats for all players for all years
    :param curr_year: year to analyze
    :param curr_team: team to analyze
    :return: G: graph for that team
    """
    idx = (df_all_stats[curr_year]['Tm'] == curr_team) & (df_all_stats[curr_year]['MP_adv'] > 200)
    Team_df = df_all_stats[curr_year][['Player', 'AST', 'PS/G', 'Pos', 'MP']][idx]
    # can't keep everyone bc some rosters bigger than others so will mess up stats - taking the top 8 players
    Team_df = Team_df.sort_values(['MP'], ascending=False)
    Team_df = Team_df.iloc[0:8]
    curr_team_ast_pts = Team_df[['AST', 'PS/G']].values
    summed_ast_pts = curr_team_ast_pts.sum(axis=0)
    # lets make a matrix of all pairwise comparisons
    all_pairwise_edges = np.ndarray(shape= (np.shape(curr_team_ast_pts)[0], np.shape(curr_team_ast_pts)[0]))
    # build the ratio for edges = (100 * AST for a person / AST total) * (100 * PTS for a person / PTS total)
    for i in range(0, np.shape(curr_team_ast_pts)[0]):
        for j in range(0, np.shape(curr_team_ast_pts)[0]):
#             all_pairwise_edges[i,j] = (100 * curr_team_ast_pts[i][0]/summed_ast_pts[0])
#                       * (100 * curr_team_ast_pts[j][1]/summed_ast_pts[1])
            all_pairwise_edges[i, j] = (curr_team_ast_pts[i][0]) * (curr_team_ast_pts[j][1])
    # for the nodes lets use player names
    pnames = Team_df['Player'].tolist()
    curr_positions = Team_df['Pos'].tolist()
    minutes_played = Team_df['MP'].tolist()
    ast_given = Team_df['AST'].tolist()
    node_size_use = Team_df['PS/G']
    node_size_use = node_size_use.as_matrix()
    G = nx.DiGraph()
    G.add_nodes_from(pnames)
    # to save the PS/G for node size later
    points_per_game = {}
    pos_ = {}
    mp_ = {}
    ast_ = {}
    for i in range(0, len(pnames)):
        points_per_game[pnames[i]] = node_size_use[i]
        pos_[pnames[i]] = curr_positions[i]
        mp_[pnames[i]] = minutes_played[i]
        ast_[pnames[i]] = ast_given[i]
    nx.set_node_attributes(G, points_per_game, 'ppg')
    nx.set_node_attributes(G, pos_, 'pos')
    nx.set_node_attributes(G, mp_, 'mp')
    nx.set_node_attributes(G, ast_, 'ast')
    labels_use = {}
    outward_vs_inward_direction = []
    for i in range(0, np.shape(curr_team_ast_pts)[0]):
        labels_use[pnames[i]] = pnames[i]
#         for j in range(0,np.shape(curr_team_ast_pts)[0]):
#                 G.add_edge(pnames[i],pnames[j], weight= all_pairwise_edges[i,j])
#                 G.add_edge(pnames[j],pnames[i], weight= all_pairwise_edges[j,i])
        for j in range(i+1, np.shape(curr_team_ast_pts)[0]):
            if (all_pairwise_edges[i, j] > all_pairwise_edges[j, i]) & (all_pairwise_edges[i, j] > 5):
                G.add_edge(pnames[i], pnames[j], weight=all_pairwise_edges[i, j])
                outward_vs_inward_direction.append('k')
            elif (all_pairwise_edges[j, i] > all_pairwise_edges[i, j]) & (all_pairwise_edges[j, i] > 5):
                G.add_edge(pnames[j], pnames[i], weight=all_pairwise_edges[j, i])
                outward_vs_inward_direction.append('k')
    return G


def visualize_graph_for_team_year(G, save_tag=False, save_path='Graph_Test.png'):
    """
    Visualize the graph with the each node being a player and each edge being the
    Scoring Potential (thicker lines = higher scoring potential).
    Each position will have a unique color with darker colors being bigger players
    (i.e. yellow = PG and dark purple = Center)
    The size of each node corresponds to how much that node scores per game
    :param G: the graph for a team
    :param save_tag: Whether to save or not
    :param save_path: Where and what name to save as
    :return:
    """
    pos = nx.kamada_kawai_layout(G)
#     pos=nx.spring_layout(G, iterations=500)
    # pos=nx.circular_layout(G)
    cm = plt.get_cmap('inferno_r', 7) # 7 bc of the 5 positions and no black
    unique_pos = ['PG', 'SG', 'SF', 'PF', 'C']
    curr_positions = nx.get_node_attributes(G, 'pos')
    nodecolors = [cm.colors[j] for i in curr_positions for j in range(0,len(unique_pos))
                  if curr_positions[i] == unique_pos[j]]
    edge_labels=dict([((u, v, ), round(d['weight']))
                     for u, v, d in G.edges(data=True)])
    # for the node size
    node_size_use = nx.get_node_attributes(G, 'ppg')
    node_array = [node_size_use[i]*25 for i in list(node_size_use)]
    labels_use = {}
    for i in G.nodes():
        labels_use[i] = i
    # print(edge_labels)
    edgewidth = [d['weight']/25 for (u, v, d) in G.edges(data=True)]
    plt.figure(figsize=(10, 4))
    # # we can now added edge thickness and edge color
    nx.draw_networkx_nodes(G, pos=pos, node_size=node_array, node_color=nodecolors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edgewidth, alpha=0.15, arrowstyle='simple',
                           arrowsize=5, arrows=False, edge_cmap=plt.cm.Greys)
    nx.draw_networkx_labels(G, pos, labels_use, font_size=8)
    plt.axis('off')
    plt.tight_layout()
    if save_tag == True:
        # find the root above
        save_root = save_path[0:save_path.rfind('/')+1]
        print(save_root)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        plt.savefig(save_path, bbox_inches='tight', dpi = 150)
    plt.show()


def get_graph_metric(G, metric, extra_weighting_values="None"):
    """
    Function to get many of the graph theory metrics we are interested in
    using Networkx. Current metrics that can be obtained include:
    eigenvector centrality, weighted clustering, katz centrality,
    current flow closeness centrality, pagerank, closeness vitality
    :param G: graph for a team
    :param metric: what metric to analyze
    :param extra_weighting_values: if you want to additionally weight the output
    by some additional metric enter that metric here and will weight each by player
    by that players weighted value/ sum all weighted values
    :return: all_ec, output
    all_ec: a list for all players without labels for each player
    output: dict with each player as a key
    """
    if metric == "eigenvector_centrality":
        output = nx.eigenvector_centrality(G, weight='weight', max_iter=500)
    elif metric == "weighted_clustering":
        output = nx.algorithms.cluster.clustering(G, weight='weight')
    elif metric == "katz_centrality":
        output = nx.katz_centrality(G, max_iter=1000)
    elif metric == "current_flow_closeness_centrality":
        output = nx.current_flow_closeness_centrality(G, weight='weight')
    elif metric == "pagerank":
        output = nx.pagerank_numpy(G, weight='weight')
    elif metric == "closeness_vitality":
        output = nx.closeness_vitality(G, weight='weight')
    else:
        print('Need to give valid metric')
    if extra_weighting_values != "None":
        from_node = nx.get_node_attributes(G,extra_weighting_values)
        weighted_val = [from_node[i] for i in list(from_node)]
        weighted_val = weighted_val/np.sum(weighted_val)
    all_ec = []
    for j, i in enumerate(output.keys()):
        if extra_weighting_values != "None":
            all_ec.append(output[i] * weighted_val[j])
        else:
            all_ec.append(output[i])
    return all_ec, output


def plot_team_scatter(df, xstat, ystat, title='None'):
    """
    Plot a scatter plot where each dot is a team with the team's 3 letter label
    written right next to the dot
    :param df: dataFrame
    :param xstat: stat to go on x axis
    :param ystat: stat to go on y axis
    :param title: If you want a title
    :return:
    """
    fig, ax = plt.subplots()

    ax.scatter(df[xstat], df[ystat])

    for i, txt in enumerate(df['Tm']):
        ax.annotate(txt, (df[xstat][i], df[ystat][i]))
    if title != "None":
        plt.title(title)
    plt.xlabel(xstat)
    plt.ylabel(ystat)
    plt.show()


def normalize(a):
    """
    Normalize to a unit vector
    :param a: vector to normalize
    :return: normalized vector
    """
    a = np.array(a)
    return a / np.linalg.norm(a)


def load_images_from_folder(folder):
    """
    Load images from a folder
    :param folder: folder to load images from
    :return: all images
    """
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
        images.append(img)
    return images


def plot_tsne_with_colors_sorted(tsne_result, vector, cmap_use, save_tag=False, save_name='Test.pdf'):
    """
    Plot the tsne output but color each dot based on the relative value within vector
    with colors according to cmap_use
    Choose to save or not
    :param tsne_result: the output from tsne dimensionality reduction
    :param vector: the sorting vector for the colormap - needs to be same size as tsne_result
    :param cmap_use: colormap to use
    :param save_tag: should you save or not
    :param save_name: name and path for saving
    :return:
    """
    vector = np.asarray(vector)
    new_vector = [i[0] for i in vector]
    idx = np.argsort(new_vector)
    plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], c=cmap_use)
    plt.axis('off')
    if save_tag == True:
        plt.savefig(
            "/Users/rohanramesh/Documents/SportsData/NBA/Figures/tsne_%s.png" % save_name,
            bbox_inches='tight', dpi=100)
    plt.show()


def barplot_tsne_clust(agg_clust_labels, vector,
                       save_tag=False, save_path='Bar_Test.png', plot_n_in_cluster=False):
    """
    Plot the mean and standard error of some vector based on the clusters defined in
    agg_clust_labels (output from agglomerative hierarchical clustering where a value in cluster
    2 will have the label 2)
    :param agg_clust_labels: label output from agglomerative hierarchical clustering
    :param vector: whatever values to look at across clusters
    :param save_tag: whether to save or not
    :param save_path: name and path for saving
    :param plot_n_in_cluster: plot the number of teams/units in that cluster
    :return:
    """
    vector = np.asarray(vector)
    curr_bar = {}
    curr_means = []
    curr_SEM = []
    cluster_n = []
    n_in_cluster = []
    for curr_clust in range(0, np.max(agg_clust_labels)+1):
        curr_bar[str(curr_clust)] = vector[agg_clust_labels == curr_clust]
        n_in_cluster.append(len(vector[agg_clust_labels == curr_clust]))
        cluster_n.append(curr_clust+1)
        curr_means.append(np.mean(vector[agg_clust_labels == curr_clust]))
        curr_SEM.append(stats.sem(vector[agg_clust_labels == curr_clust]))
    bar_cmap = plt.get_cmap('plasma_r', np.max(agg_clust_labels)+1)
    fig, ax = plt.subplots()
    rects1 = ax.bar(cluster_n, curr_means, color=bar_cmap.colors, yerr=curr_SEM)
    # get y range so can scale where text is
    ymin, ymax = plt.ylim()
    scaled_val = (ymax - ymin)/18
    if plot_n_in_cluster == True:
        for i in range(0, len(rects1)):
            rect = rects1[i]
            height = rect.get_height()
            if height > 0:
                plt.text(rect.get_x() + rect.get_width()/2.0, -1*scaled_val, '%d' % int(n_in_cluster[i]),
                         ha='center', va='bottom')
            elif height <= 0:
                plt.text(rect.get_x() + rect.get_width()/2.0, 0, '%d' % int(n_in_cluster[i]),
                         ha='center', va='bottom')
    # plt.ylabel(curr_stat)
    plt.xlabel('Cluster')
    if save_tag == True:
        save_root = save_path[0:save_path.rfind('/')+1]
        print(save_root)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        plt.savefig(save_path, bbox_inches='tight', dpi = 150)
    plt.show()


def plot_scatter_across_years(xstat, ystat, df):
    """
    Plot the scatter plot across all years for two stats
    :param xstat: stat for x axis
    :param ystat: stat for y axis
    :param df: dict where each key is a year and df['year'] = dataFrame
    :return:
    """
    years_to_use = list(df)
    for curr_year in years_to_use[1:]:
        a = df[str(curr_year)][xstat].tolist()
        b = df[str(curr_year)][ystat].tolist()
        plt.scatter(a, b)
    plt.xlabel(xstat)
    plt.ylabel(ystat)
    plt.legend(years_to_use[1:])
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues, save_tag=False):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param cm: confusion matrix
    :param classes: labels for x and y axis
    :param normalize: normalize to be from 0-1 or raw numbers
    :param title: title of plot
    :param cmap: colormap to use
    :param save_tag: save or not
    :return:
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_tag == True:
        plt.savefig(
            "/Users/rohanramesh/Documents/SportsData/NBA/Figures/RF_Classification/%s_cm_" % title,
            bbox_inches='tight', dpi=100)

