import collections
import numpy as np
import networkx as nx
import hypernetx as hnx


def convert_to_line_graph(hgraph_dict):
    # Line-graph is a NetworkX graph
    line_graph = nx.Graph()

    node_list = list(hgraph_dict.keys())
    node_list.sort() # sort the node by id
    # Add nodes
    [line_graph.add_node(edge) for edge in node_list]


    # For all pairs of edges (e1, e2), add edges such that
    # intersection(e1, e2) is not empty
    s = 1
    for node_idx_1, node1 in enumerate(node_list):
        for node_idx_2, node2 in enumerate(node_list[node_idx_1 + 1:]):
            vertices1 = hgraph_dict[node1]
            vertices2 = hgraph_dict[node2]
            if len(vertices1) > 0 or len(vertices2) > 0:
                # Compute the intersection size
                intersection_size = len(set(vertices1) & set(vertices2))
                union_size = len(set(vertices1) | set(vertices2))
                jaccard_index = intersection_size / union_size
                if intersection_size >= s:
                    line_graph.add_edge(node1, node2, intersection_size=1/intersection_size, jaccard_index=1/jaccard_index)
    # line_graph = nx.readwrite.json_graph.node_link_data(line_graph)
    return line_graph

def get_hgraph_dual(hgraph):
    return hgraph.dual()

def get_v(hgraph_dict, hgraph_dual_dict):
    edge_list = sorted(list(hgraph_dict.keys()))
    vlist = [0 for i in edge_list]
    for eidx,edge in enumerate(edge_list):
        sum_of_degrees = 0
        node_list = list(hgraph_dict[edge])
        for node in node_list:
            sum_of_degrees += len(hgraph_dual_dict[node])
        vlist[eidx] = sum_of_degrees
        
    return np.array(vlist)/sum(vlist)

def get_all_edges(hgraph, node):
    """
    Get all edges containing the given node.
    """
    hgraph_dict = hgraph.incidence_dict
    edges = []
    for h in hgraph_dict:
        node_list = hgraph_dict[h]
        if node in node_list:
            edges.append(h)
    return edges


def get_omega(hgraph,hgraph_dual,lgraph,weight_type=None):
    """
    Node-hyperedge distances. Uses single Floyd-Warshall call
    on line graph
    
    Parameter:
    hgraph      : hnx.Hypergraph, nodes in form [str(v) for v in range(numNodes)] 
    hgraph_dual : hnx.Hypergraph
    lgraph      : nx.Graph
    weight_type : str, optional
        Options: None, 'intersection_size', 'jaccard_index'. The default is None.
                 If None, then each edge has weight 1.
                 
    Returns:
    w : np.ndarray
    
    """
    num_nodes, num_edges = hgraph.shape
    nodes = sorted(list(hgraph.nodes))
    
    try:
        weights = nx.adjacency_matrix(lgraph, weight=weight_type).A
    except:
        return "weight type doesn't exist!"
    
    ldist = nx.floyd_warshall_numpy(lgraph,weight=weight_type) # May have inf 
    
    w = np.zeros((num_nodes, num_edges))
    for i,node in enumerate(nodes):
        # node = str(i) # Important that hypergraph nodes were formatted this way
        # all hyperedges containing the node
        idxs = list(hgraph_dual.incidence_dict[node])
        shortest_target_dists = ldist[idxs,:].min(axis=0)
        w[i,:] = shortest_target_dists
        
    return w