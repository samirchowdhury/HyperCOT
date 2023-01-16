import collections
import numpy as np
import networkx as nx
import hypernetx as hnx
from scipy import sparse
from scipy.sparse.linalg import eigs, eigsh
import plotly.graph_objects as go
import ot
import time
from cot import *


"""
Hypernetwork constructions
"""
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

"""
Multiscale matching constructions

Function descriptions

get_graph_plotters  : Get plotly elements needed for plotting a geometric graph
get_nerve           : Get nerve graph from a geometric graph
multiscale_nerve    : Iterate get_nerve to get multiple reductions of an input graph
multiscale_cot      : Multiscale matching
"""


def get_graph_plotters(G,v):
    """
    Get elements needed for plotting a geometric graph using plotly

    Parameters:
    G : nx.Graph
    v : (len(G),3) ndarray, vertex coordinates
    """
    n = len(G)
    
    #we  need to create lists that contain the starting and ending coordinates of each edge.
    x_edges=[]
    y_edges=[]
    z_edges=[]

    #need to fill these with all of the coordiates
    edgeList = list(G.edges)
    for edge in edgeList:
        #format: [beginning,ending,None]
        x_coords = [v[edge[0],0],v[edge[1],0],None]
        x_edges += x_coords

        y_coords = [v[edge[0],1],v[edge[1],1],None]
        y_edges += y_coords

        z_coords = [v[edge[0],2],v[edge[1],2],None]
        z_edges += z_coords
        
    #create a trace for the edges
    trace_edges = go.Scatter3d(x=x_edges,
                            y=y_edges,
                            z=z_edges,
                            mode='lines',
                            line=dict(color='black', width=0.5),
                            hoverinfo='none')
    
    #create a trace for the nodes
    trace_nodes = go.Scatter3d(x=v[:,0],
                             y=v[:,1],
                            z=v[:,2],
                            mode='markers',
                            marker=dict(symbol='circle',
                                        size=2,
                                        color='red',
                                        line=dict(color='black', width=0.5)),
                            text=list(range(n)),
                            hoverinfo='text')
    
    #we need to set the axis for the plot 
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    
    #also need to create the layout for our plot
    layout = go.Layout(title="Nerve",
                    width=650,
                    height=625,
                    showlegend=False,
                    scene=dict(xaxis=dict(axis),
                            yaxis=dict(axis),
                            zaxis=dict(axis),
                            ),
                    margin=dict(t=100),
                    hovermode='closest',
                    paper_bgcolor = 'rgba(0,0,0,0)',
                    plot_bgcolor = 'rgba(0,0,0,0)')

    plot_data = [trace_edges, trace_nodes]
    return plot_data, layout



def get_nerve(G,K,v):
    """
    Get iterated nerve graph
    
    Parameters:
    G : nx.Graph
    K : heat kernel
    v : (len(G),3) vertex coordinates
    
    Returns:
    G : nx.Graph, reduced graph
    v : (len(G),3) ndarray, vertex coordinates of reduced graph
    memberMat : scipy.sparse.coo_matrix, binary membership matrix representing cover elements
    
    
    """       
    # Get nerve
    unCov = set(G.nodes)
    memberMat = []
    pts = []
    n = len(G)

    while len(unCov):
        pt = next(iter(unCov)) 
        pts.append(pt)
        h = np.zeros(n)
        h[pt] = 1
        h_ = K @ h
        halfMax = h_.max()/4 # FUTURE: rename halfMax to fracMax, as we want to take fractions
        halfVec = h_>= halfMax/2
        memberMat.append(sparse.coo_matrix(halfVec).astype(int))
        halfN = set(np.where(h_ >= halfMax)[0]) #half-max neighbors
        halfN.add(pt) # Ensure source is covered when using approximation for heat kernel
        unCov -= halfN

    memberMat = sparse.vstack(memberMat).tocsr()
    # Restrict to new graph
    G = nx.Graph(memberMat @ memberMat.T) # Nerve graph
    v = v[pts]
    K = K[pts][:,pts]
    plot_data, layout = get_graph_plotters(G,v)
    return G, v, memberMat, plot_data, layout


def multiscale_nerve(G,v,plot_data,t_init,params):
    """
    Get multiscale nerve of an input geometric graph.
    
    Parameters
    ----------
    G : networkX graph
    v : numpy array of 3D vertex coordinates
    plot_data : list[Scatter3d] containing plotting data
    t_init : initial t value for heat kernel
    params : dictionary of reduction parameters
    
    Returns
    -------
    G_list : list[nx.Graph] of nerves, beginning with first reduction
    v_list : list[np.ndarray] of vertex coordinates of nerves
    plot_data_list : list[list[Scatter3d]] of plotting data for nerves
    """
    G_list = [G]
    v_list = [v]
    plot_data_list = [plot_data]
    memberMat_list = [None]
    t = t_init
    
    while len(G) > params['min_graph_size']:
        print(f"Graph size {len(G)}, using t={t} and reducing...")
        start = time.time()
        # Obtain heat kernel, possibly through approximation
        L = nx.normalized_laplacian_matrix(G)
        if (L.shape[0] / params['low_rank_approx_k']) > params['low_rank_factor']:
            #using shift-invert to get smallest eigenvalues
            lam,phi = eigsh(L,k=params['low_rank_approx_k'],sigma=0,which='LM')
            print(f"Graph size {L.shape[0]}, using low-rank approx with k={params['low_rank_approx_k']}")
        else:
            lam,phi = np.linalg.eigh(L.A)
            print(f"Graph size {L.shape[0]}, using full kernel")
            
        K = phi @ np.diag(np.exp(-t * lam)) @ phi.T
        print(f"Kernel computed in {time.time() - start:3.3g} seconds")
        start = time.time()
        # Compute cover and nerve
        G, v,memberMat,plot_data,_  = get_nerve(G,K,v)
        print(f"Nerve of size {len(G)} computed in {time.time() - start:3.3g} seconds")
        
        # Save results
        if (len(G_list[-1]) / len(G)) < 1.1:
            break # point of diminishing returns
        G_list.append(G)
        v_list.append(v)
        memberMat_list.append(memberMat)
        plot_data_list.append(plot_data)
        t = np.log(len(G))/np.log(10)
        
        
    return G_list, v_list, plot_data_list, memberMat_list


def multiscale_cot(X1_list, X2_list, w1_list, w2_list, v1_list, v2_list,
                   niter=10, verbose=True, log=False):

    """ Returns multiscale HyperCOT between two lists of datasets

    Parameters
    ----------
    X1_list : list[numpy array], shape (n,). Source dataset.
    X2_list : list[numpy array], shape (n,). Target dataset
      Each entry of X1_list, X2_list is a rectangular matrix (possibly different sizes)
         
    w1_list : list[numpy array], shape (n,)
              Weight (histogram) on the samples (rows) of X1s. If None uniform distribution is considered.
    w2_list : numpy array, shape (n',)
              Weight (histogram) on the samples (rows) of X2s. If None uniform distribution is considered.
    v1_list : numpy array, shape (d,)
              Weight (histogram) on the features (columns) of X1s. If None uniform distribution is considered.
    v2_list : numpy array, shape (d',)
              Weight (histogram) on the features (columns) of X2s. If None uniform distribution is considered.
      Weights on columns of one X1 matrix = weights on rows of next X1 matrix. However, no error checking
      is asserted currently.
        
    """  

    Ts_list = []
    Tv_list = []
    constC_s_list, hC1_s_list, hC2_s_list = [], [], []
    constC_v_list, hC1_v_list, hC2_v_list = [], [], []
    
    for idx in range(len(X1_list)):
        X1 = X1_list[idx]
        X2 = X2_list[idx]
        w1 = w1_list[idx]
        w2 = w2_list[idx]
        v1 = v1_list[idx]
        v2 = v2_list[idx]
        
        # Initialize coupling matrices
        Ts = np.ones((X1.shape[0], X2.shape[0])) / (X1.shape[0] * X2.shape[0])  # is (n,n')
        Tv = np.ones((X1.shape[1], X2.shape[1])) / (X1.shape[1] * X2.shape[1])  # is (d,d')
        
        # Initialize constant matrices
        constC_s, hC1_s, hC2_s = init_matrix_np(X1, X2, v1, v2)
        constC_v, hC1_v, hC2_v = init_matrix_np(X1.T, X2.T, w1, w2)
        
        # Append to list
        Ts_list.append(Ts)
        Tv_list.append(Tv)
        constC_s_list.append(constC_s)
        hC1_s_list.append(hC1_s)
        hC2_s_list.append(hC2_s)
        constC_v_list.append(constC_v)
        hC1_v_list.append(hC1_v)
        hC2_v_list.append(hC2_v)
        

    log_out ={}
    log_out['cost'] = []
    
    for i in range(niter):
        # Left-to-right sweep 
        for idx in range(len(X1_list)):
            M = constC_v_list[idx] - np.dot(hC1_v_list[idx], Ts_list[idx]).dot(hC2_v_list[idx].T)
            Tv = ot.emd(v1_list[idx], v2_list[idx], M, numItermax=1e7)
            Tv_list[idx] = Tv
            if idx < len(X1_list)-1: #Initialize next Ts value
                Ts_list[idx+1] = Tv
         

        # Right-to-left sweep
        for idx in range(len(X1_list)-1,-1,-1):
            M = constC_s_list[idx] - np.dot(hC1_s_list[idx], Tv_list[idx]).dot(hC2_s_list[idx].T)
            Ts = ot.emd(w1_list[idx], w2_list[idx], M, numItermax=1e7)
            Ts_list[idx] = Ts
            if idx > 0:
                Tv_list[idx-1] = Ts
                
        cost = np.sum(M * Ts)
        log_out['cost'].append(cost)
        
        if (i > int(0.75 * niter)) and (cost < np.percentile(log_out['cost'],20)):
            print(f'breaking at iteration {i}')
            break
    
    return Ts_list, Tv_list, log_out