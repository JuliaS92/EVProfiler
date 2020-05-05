#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import networkx as nx
from sklearn.metrics import pairwise as pw
pd.set_option('mode.chained_assignment',None)
import panel as pn
import hvplot.pandas
import hvplot.networkx as hvnx
import holoviews as hv
from holoviews import opts
import matplotlib.pyplot as plt
import time


# In[2]:


df = pd.read_csv("PredictorData.txt", sep='\t', usecols=range(4,31))
add_cols = pd.read_csv("PredictorData.txt", sep='\t', usecols=["Delta F3", "Min correl", "id"])
df = df.merge(add_cols, on="id")
df.index = df["id"]

df_core = df.iloc[:, 18:33]
meta_dict = {}
meta_dict["gene_id"] = {}
for g, i in zip(df["Lead gene name"], df.index):
    if g not in meta_dict["gene_id"].keys():
        meta_dict["gene_id"][g] = i
    else:
        if type(meta_dict["gene_id"][g]) == list:
            meta_dict["gene_id"][g].append(i)
        else:
            meta_dict["gene_id"][g] = [meta_dict["gene_id"][g], i]
meta_dict["id_meta"] = df.iloc[:, 1:18].to_dict("index")


# In[3]:


#df_pca = pd.read_csv("PCA cooridnates Jurkat Exosomes 13 April 2020t.txt", sep='\t')
#df_pca.columns
#df_pca.columns = ['Primary ID', 'ClassID', 'Lead gene name',
#       'Protein names', 'Gene names',
#       'Fasta headers', 'Sec. ID',
#       'Organelle', 'Min Ratio count', 'Class',
#       'PCA 1', 'PCA2']


# In[4]:


## Backend functions

## single query dataset
def hv_query_df(df_core, meta_dict, gene, dist="Manhattan",
                min_enr=-1, min_corr=0, size=50, rank_tolerance=25, perc_area=250):
    size += 1
    if gene.upper() not in meta_dict["gene_id"].keys():
        return "Gene {} not in dataset".format(gene)
    
    # Filter data
    df = df_core.loc[[enr >= min_enr and corr >= min_corr for enr, corr
                      in zip(df_core["Delta F3"], df_core["Min correl"])]].copy()
    df.drop(["Delta F3", "Min correl"], axis=1, inplace=True)
    
    # Retrieve query
    query_id = meta_dict["gene_id"][gene]
    msg_append = ""
    if type(query_id) == list:
        counts = [meta_dict["id_meta"][qid1]["Min Ratio count"] for qid1 in query_id]
        max_count = [qid2 for qid2, count in zip(query_id, counts) if count == max(counts)][0]
        msg_append = "Multiple ids found for gene {}: {}. Using id with the highest number of counts: {}".format(
            gene, str(query_id), max_count)
        query_id = max_count
    if query_id not in df.index:
        return "The query protein {} was filtered out.".format(gene)
    query_profile = df.loc[query_id,:]
    
    # Calculate difference profiles and sum within replicates
    df_diff = df-query_profile
    df_diff = df_diff.transform(np.abs)
    df_diff_rep = df_diff.apply(lambda x: pd.Series([sum(x[i:i+3]) for i in [0,3,6]]), axis=1)
    df_diff_rep.columns=["Replicate 1", "Replicate 2", "Replicate 3"]
    
    # Score distances
    if dist == "Manhattan":
        dists = df_diff.apply(sum, axis=1)
    dists.name = "Distance measure"
    dists = dists.sort_values()
    
    # Calculate rank ranges across replicates
    df_diff_rep_rank = df_diff_rep.rank()
    df_diff_rep_rank.columns = [el + " rank" for el in df_diff_rep_rank.columns]
    df_diff_rep_rank["Rank range"] = df_diff_rep_rank.apply(lambda x: max(x)-min(x), axis=1)
    df_diff_rep_rank["Common lists"] = df_diff_rep_rank[["Replicate 1 rank", "Replicate 2 rank", "Replicate 3 rank"]].apply(
        lambda x: sum([el <= size+rank_tolerance for el in x]), axis=1)
    
    # Calculate percentiles in local neighborhood
    area_members = dists.index[0:perc_area]
    area_members = df_core.loc[area_members,[col for col in df_core.columns if col not in ["Delta F3", "Min correl"]]]
    area_dists = pw.distance.pdist(np.array(area_members), "cityblock")
    qsteps = [0,0.001,0.005,0.01,0.025,0.05,0.1,0.25,0.5,1]
    q = pd.Series(area_dists).quantile(q=qsteps)
    
    # Calculate z-scoring of distances to query
    z_dists_in = dists[1:perc_area]
    median = np.median(z_dists_in)
    dists_from_median = np.abs(z_dists_in-median)
    mad = np.median(dists_from_median)
    rob_sd = 1.4826*mad
    z_dists = -(z_dists_in-median)/rob_sd
    z_dists = pd.DataFrame(z_dists)
    z_dists.columns = ["z-scored closeness"]
    z_dists["z-scoring based category"] = pd.cut(z_dists["z-scored closeness"],
                                          bins=[-1000, 1.28, 1.64, 2.33, 3.09, 1000],
                                          labels=["-", "B", "*", "**", "***"])
    
    #assemble output table
    out_ids = dists.index[0:size]
    out = pd.DataFrame(dists[out_ids])
    out = out.join(df_diff_rep, how="left")
    out = out.join(df_diff_rep_rank, how="left")
    out = out.join(pd.DataFrame.from_dict({el:meta_dict["id_meta"][el] for el in out_ids}, orient="index"), how="left")
    out = out.join(df_core, how="left")
    out["Local distance percentile"] = pd.cut(out["Distance measure"], bins=[-1]+list(q.values)[1:],
                                              labels=qsteps[1:])
    out = out.join(z_dists, how="left")
    out.reset_index(inplace=True)
    
    msg = "Neighborhood recalculated for {}.\n".format(gene)+msg_append
    
    return out, q, msg

## query multiple genes
def multi_query(df_core, meta_dict, genes, dist="Manhattan",
                min_enr=-1, min_corr=0, size=50, rank_tolerance=25, perc_area=250, pb=None):
    hits = []
    qs = []
    msgs = ""
    for gene in genes:
        output = hv_query_df(df_core, meta_dict, gene=gene, size=size, rank_tolerance=rank_tolerance,
                             min_enr=min_enr, min_corr=min_corr, perc_area=perc_area)
        if len(output) == 3:
            (h, q, m) = output
            h["Query"] = np.repeat(gene, len(h))
            hits.append(h)
            qs.append(q)
        else:
            m = output
        msgs= msgs+m+"\n"
        if pb:
            pb.value = pb.value+1
    
    query_result = pd.concat(hits)
    q = pd.DataFrame(qs).apply(np.mean, axis=0)
    
    return query_result.reset_index(drop=True), q, msgs

## barplot function single query
def sq_barplot(df, var):
    q_m = df[["Distance measure", "Rank range", "Delta F3", "Min correl", "Local distance percentile",
              "Lead gene name", "Common lists", "z-scoring based category"]]
    q_m["Lead gene name"] = [str(n)+", rank "+str(r) for n, r in zip(q_m["Lead gene name"], q_m.index)]
    label_var_dict = {
        "Distance": "Distance measure",
        "z-scoring based category": "z-scoring based category",
        "Rank range between replicates": "Rank range",
        "Percentile change F3 - full proteome": "Delta F3",
        "Worst replicate correlation": "Min correl",
        "Local distance percentile": "Local distance percentile"
    }
    q_m = q_m.iloc[::-1,:]
    var = label_var_dict[var]
    if var == "z-scoring based category":
        z_dict = {"-":5, "B":4, "*":3, "**":2, "***":1, np.nan:0, None:0}
        q_m["z-scoring based category"] = [z_dict[k] for k in q_m["z-scoring based category"]]
        z_dict = {"-":5, "B":4, "*":3, "**":2, "***":1, "query":0}
        p = q_m.hvplot.barh(x="Lead gene name", y=var, color="Common lists",
                               height=60+11*len(q_m), xticks = [(z_dict[k], k) for k in z_dict.keys()])
    else:
        p = q_m.hvplot.barh(x="Lead gene name", y=var, color="Common lists",
                               height=60+11*len(q_m))
    p.opts(cmap=["orange", "darkgrey", "white"][0:len(np.unique(q_m["Common lists"]))][::-1])
    p.opts(legend_position="top")
    return p

## Network construction functions
def filter_nwk_members(query_result, min_rep, max_z):
    nwk_members_1 = query_result[query_result["Common lists"] >= min_rep]["id"].values
    z_dict = {"-":5, "B":4, "*":3, "**":2, "***":1, np.nan:0, None:0}
    query_result["z-scoring based category"] = [z_dict[k] for k in query_result["z-scoring based category"]]
    nwk_members_2 = query_result[query_result["z-scoring based category"] <= max_z]["id"].values
    nwk_members = [el for el in nwk_members_1 if el in nwk_members_2]
    nwk_members = np.unique(nwk_members)
    if len(nwk_members) == 1:
        return None
    else:
        return nwk_members

def layout_network(df_core, query_result, nwk_members, max_q, q, layout_method="Spring"):
    nwk_profiles = df_core.loc[nwk_members,[col for col in df_core.columns if col not in ["Delta F3", "Min correl"]]]
    dists = pw.distance.pdist(np.array(nwk_profiles), "cityblock")
    dists_pd = pd.DataFrame([[meta_dict["id_meta"][nwk_members[i1]]["Lead gene name"],
                              meta_dict["id_meta"][nwk_members[i2]]["Lead gene name"]]
                             for i1 in range(len(nwk_members)) for i2 in range(len(nwk_members)) if i1<i2])
    dists_pd.columns = ["Gene 1", "Gene 2"]
    dists_pd["dist"] = [1/el for el in dists]
    dists_pd["quantile"] = pd.cut(pd.Series(dists), bins=[-1]+list(q.values)[1:], labels=[float(el) for el in q.index[1:]])
    dists_pd = dists_pd[dists_pd["quantile"] <= float(max_q)]
    if len(dists_pd) == 0:
        return "No data at this quantile."
    
    # Genrate network graph
    G = nx.Graph()
    for idx in range(len(dists_pd)):
        G.add_edge(dists_pd.iloc[idx,0], dists_pd.iloc[idx,1], weight=dists_pd.iloc[idx,2],
                   quantile=dists_pd.iloc[idx,3])
    if layout_method == "Spectral":
        GP = nx.spectral_layout(G)
    elif layout_method == "Spring":
        GP = nx.spring_layout(G, k=5/np.sqrt(len(dists_pd)))
    
    return dists_pd, GP

## Functions for drawing a network
def draw_network_figure(dists_pd, GP, query_result, highlight, q):
    # Regerate network graph from distances
    G = nx.Graph()
    for idx in range(len(dists_pd)):
        G.add_edge(dists_pd.iloc[idx,0], dists_pd.iloc[idx,1], weight=dists_pd.iloc[idx,2],
                   quantile=dists_pd.iloc[idx,3])

    fig = plt.figure(figsize=(7,7))
    fig.set_dpi(300)

    # edges
    edge_styles = [{"width": 5, "edge_color": "darkred", "style": "solid", "alpha": 1}, # 0.1%
                   {"width": 5, "edge_color": "darkred", "style": "solid", "alpha": 1}, # 0.5%
                   {"width": 5, "edge_color": "darkred", "style": "solid", "alpha": 1}, # 1%
                   {"width": 4, "edge_color": "red", "style": "solid", "alpha": 1}, # 2.5%
                   {"width": 4, "edge_color": "red", "style": "solid", "alpha": 1}, # 5%
                   {"width": 3, "edge_color": "darkorange", "style": "solid", "alpha": 1}, # 10%
                   {"width": 2, "edge_color": "#ababab", "style": "solid", "alpha": 1}, # 25%
                   {"width": 2, "edge_color": "lightgrey", "style": "solid", "alpha": 1}, # 50%
                   {"width": 2, "edge_color": "lightgrey", "style": "dotted", "alpha": 1}] # 100%
    edges = []
    for q_cat, style in zip(q.index[-2::-1], edge_styles[-2::-1]):
        edgelist = [(u, v) for (u, v, d) in G.edges(data=True) if d['quantile']==float(q_cat)]
        if len(edgelist) == 0:
            continue
        Gi = nx.Graph(edgelist)
        nx.draw_networkx_edges(Gi, GP, **style)
    
    # nodes
    # highlighted gene
    if highlight is not None:
        Gi = nx.Graph([(hit,hit) for hit in [highlight] if hit in G.nodes()])
        nx.draw_networkx_nodes(Gi, GP, node_color="blue", node_size=800)
    # replicate 1
    Gi = nx.Graph([(hit,hit) for hit, rep in zip(query_result["Lead gene name"], query_result["Common lists"]) if rep==1 and hit in G.nodes()])
    nx.draw_networkx_nodes(Gi, GP, node_color="white", node_size=600)
    nx.draw_networkx_labels(Gi, GP, font_size=7)
    # replicate 2
    Gi = nx.Graph([(hit,hit) for hit, rep in zip(query_result["Lead gene name"], query_result["Common lists"]) if rep==2 and hit in G.nodes()])
    nx.draw_networkx_nodes(Gi, GP, node_color="lightgrey", node_size=600)
    nx.draw_networkx_labels(Gi, GP, font_size=7)
    # replicate 3
    Gi = nx.Graph([(hit,hit) for hit, rep in zip(query_result["Lead gene name"], query_result["Common lists"]) if rep==3 and hit in G.nodes()])
    nx.draw_networkx_nodes(Gi, GP, node_color="orange", node_size=600)
    nx.draw_networkx_labels(Gi, GP, font_size=7)
    # query
    if "Query" in query_result.columns:
        Gi = nx.Graph([(hit,hit) for hit in np.unique(query_result["Query"]) if hit in G.nodes()])
        nx.draw_networkx_nodes(Gi, GP, node_color="red", node_size=600)
    else:
        Gi = nx.Graph([(hit,hit) for hit in [query_result["Lead gene name"][0]] if hit in G.nodes()])
        nx.draw_networkx_nodes(Gi, GP, node_color="red", node_size=600)
    plt.close()

    return fig

def draw_network_interactive(dists_pd, GP, query_result, highlight, q):
    G = nx.Graph()
    for idx in range(len(dists_pd)):
        G.add_edge(dists_pd.iloc[idx,0], dists_pd.iloc[idx,1], weight=dists_pd.iloc[idx,2],
                   quantile=dists_pd.iloc[idx,3])
    
    width=min(900, int(np.sqrt(len(G.nodes)))*150)
    if width < 400:
        width=400
    
    Gi = nx.Graph([(hit,hit) for hit in [highlight] if hit in G.nodes()])
    highlight = hvnx.draw(Gi, GP, node_color="blue", node_size=3000, with_labels=False,
                        width=width, height=width)
    Gi = nx.Graph([(hit,hit) for hit, rep in zip(query_result["Lead gene name"], query_result["Common lists"]) if rep==1 and hit in G.nodes()])
    labels0 = hvnx.draw(Gi, GP, node_color="white", node_size=2500, with_labels=True,
                        width=width, height=width)
    Gi = nx.Graph([(hit,hit) for hit, rep in zip(query_result["Lead gene name"], query_result["Common lists"]) if rep==2 and hit in G.nodes()])
    labels1 = hvnx.draw(Gi, GP, node_color="lightgrey", node_size=2500, with_labels=True,
                        width=width, height=width)
    Gi = nx.Graph([(hit,hit) for hit, rep in zip(query_result["Lead gene name"], query_result["Common lists"]) if rep==3 and hit in G.nodes()])
    labels2 = hvnx.draw(Gi, GP, node_color="orange", node_size=2500, with_labels=True,
                        width=width, height=width)
    if "Query" in query_result.columns:
        Gi = nx.Graph([(hit,hit) for hit in np.unique(query_result["Query"]) if hit in G.nodes()])
        labels3 = hvnx.draw(Gi, GP, node_color="red", node_size=2500, with_labels=True,
                            width=width, height=width)
    else:
        Gi = nx.Graph([(hit,hit) for hit in [query_result["Lead gene name"][0]] if hit in G.nodes()])
        labels3 = hvnx.draw(Gi, GP, node_color="red", node_size=2500, with_labels=True,
                            width=width, height=width)
    # .opts(tools=[HoverTool(tooltips=[('index', '@index_hover')])])

    # edges
    edge_styles = [{"edge_width": 8, "edge_color": "darkred", "style": "solid", "alpha": 1}, # 0.1%
                   {"edge_width": 8, "edge_color": "darkred", "style": "solid", "alpha": 1}, # 0.5%
                   {"edge_width": 8, "edge_color": "darkred", "style": "solid", "alpha": 1}, # 1%
                   {"edge_width": 6, "edge_color": "red", "style": "solid", "alpha": 1}, # 2.5%
                   {"edge_width": 6, "edge_color": "red", "style": "solid", "alpha": 1}, # 5%
                   {"edge_width": 4, "edge_color": "darkorange", "style": "solid", "alpha": 1}, # 10%
                   {"edge_width": 3, "edge_color": "#ababab", "style": "solid", "alpha": 1}, # 25%
                   {"edge_width": 2, "edge_color": "lightgrey", "style": "solid", "alpha": 1}, # 50%
                   {"edge_width": 2, "edge_color": "lightgrey", "style": "dotted", "alpha": 1}] # 100%
    edges = []
    for q_cat, style in zip(q.index[1:], edge_styles):
        edgelist = [(u, v) for (u, v, d) in G.edges(data=True) if d['quantile']==float(q_cat)]
        if len(edgelist) == 0:
            continue
        Gi = nx.Graph(edgelist)
        edge = hvnx.draw(Gi, GP, node_size=2000, with_labels=False,
                         width=width, height=width, **style)
        edges.append(edge)
    
    nw = hv.Overlay(edges[::-1]) * highlight * labels0 * labels1 * labels2 * labels3
    return nw


# In[5]:


## Interface elements for single query output

# Panes to cache query calculation as json
cache_sq_neighborhood = pn.pane.Str("")  # neighborhood query result
cache_sq_q = pn.pane.Str("")  # neighborhood quantile steps
cache_sq_nwdists = pn.pane.Str("")  # network edge distances
cache_sq_gp = pn.pane.Str("")  # network layout positions

# Main options
input_sq_gene = pn.widgets.AutocompleteInput(options=[el for el in meta_dict["gene_id"].keys() if type(el) == str],
                                             name="Select gene (capital letters):", value="CD63")
input_sq_topn = pn.widgets.IntSlider(start=20, end=100, step=5, value=50, value_throttled=50,
                                     name="Neighborhood size")
input_sq_button = pn.widgets.Button(name='Calculate neighborhood', button_type='primary')
output_sq_status = pn.pane.Markdown("", width_policy="max", sizing_mode="stretch_width")

# Advanced options
input_sq_tolerance = pn.widgets.IntSlider(start=5, end=50, step=5, value=25, value_throttled=25,
                                          name="Replicate neighborhood tolerance")
input_sq_minr = pn.widgets.FloatSlider(start=-1, end=1, step=0.05, value=0, value_throttled=0,
                                       name="Minimal profile correlation")
input_sq_minenr = pn.widgets.FloatSlider(start=-1, end=1, step=0.05, value=-1, value_throttled=-1,
                                         name="Minimal percentile shift in F3 vs. full proteome")
input_sq_nhsize = pn.widgets.IntSlider(start=50, end=500, step=50, value=250, value_throttled=250,
                                       name="Neighborhood size for scoring")

# Tabcolumn containing options
options_single = pn.Tabs()
options_single.append(("Neighborhood selection", pn.Column(input_sq_gene, input_sq_topn,
                                                           input_sq_button, output_sq_status)))
options_single.append(("Advanced options", pn.Column(input_sq_tolerance, input_sq_minr,
                                                     input_sq_minenr, input_sq_nhsize)))

# Options for network display
input_sq_minz = pn.widgets.Select(options=["***", "**", "*", "B", "all"],
                                  name="Minimum z scoring for nodes:", value="**")
input_sq_minrep = pn.widgets.Select(options=[1, 2, 3], name="Minimum shared replicates:", value=2)
input_sq_maxq = pn.widgets.Select(options=["1%", "5%", "10%", "25%", "50%", "100%"],
                                  name="Maximum quantile for edges:", value="50%")
input_sq_highlight = pn.widgets.AutocompleteInput(options=[el for el in meta_dict["gene_id"].keys() if type(el) == str]+
                                                  ["none", "None"], name="Highlight gene:", value="None")
input_sq_disablehvnx = pn.widgets.Checkbox(value=False, name="figure style")

# Options for barplot
input_sq_bary = pn.widgets.Select(options=["Distance", "z-scoring based category", "Rank range between replicates",
                                           "Percentile change F3 - full proteome", "Worst replicate correlation",
                                           "Local distance percentile"], name="Select y-axis to display:")


# In[6]:


## Functions for single query data display

# Get the single query data
def store_gene_data(event):
    input_sq_button.disabled=True
    output_sq_status.object = "Calculating neighborhood ..."
    output = hv_query_df(df_core, meta_dict, gene=input_sq_gene.value, size=input_sq_topn.value_throttled,
                         rank_tolerance=input_sq_tolerance.value_throttled, min_corr=input_sq_minr.value_throttled,
                         min_enr=input_sq_minenr.value_throttled, perc_area=input_sq_nhsize.value_throttled)
    if len(output) == 3:
        (df, q, msg) = output
        output_sq_status.object = "Done calculating neighborhood"
        cache_sq_q.object = q.to_json()
        cache_sq_neighborhood.object = df.to_json()
    else:
        output_sq_status.object = output
        cache_sq_q.object = ""
        cache_sq_neighborhood.object = ""
    input_sq_button.disabled=False
input_sq_button.on_click(store_gene_data)

# Barplot
@pn.depends(cache_sq_neighborhood.param.object, input_sq_bary.param.value)
def get_gene_data(df, var):
    try:
        df = pd.read_json(df).sort_values("Distance measure")
    except:
        return ""
    p = sq_barplot(df, var)
    return p

# Networkplot
@pn.depends(cache_sq_neighborhood.param.object,
            input_sq_minz.param.value, input_sq_maxq.param.value, input_sq_minrep.param.value)
def layout_single_network(query, min_z, max_q, min_rep):
    # Retrieve neighborhood information
    try:
        query_result = pd.read_json(query).sort_values("Distance measure")
        q = pd.read_json(cache_sq_q.object, typ="series", convert_dates=False, convert_axes=False)
    except:
        cache_sq_nwdists.object = ""
        cache_sq_gp.object = ""
        return "Failed to read neighborhood (see settings panel for details)."
    
    # Define node list
    z_dict = {"all":5, "B":4, "*":3, "**":2, "***":1}
    min_z = z_dict[min_z]
    try:
        nwk_members = filter_nwk_members(query_result, min_rep, min_z)
    except:
        cache_sq_nwdists.object = ""
        cache_sq_gp.object = ""
        return "Failed to generate node list."
    
    # Layout network
    q_dict = {"1%":0.01, "5%":0.05, "10%":0.1, "25%":0.25, "50%":0.5, "100%":1}
    try:
        dists_pd, GP = layout_network(df_core, query_result, nwk_members, q_dict[max_q], q, layout_method="Spring")
    except:
        cache_sq_nwdists.object = ""
        cache_sq_gp.object = ""
        return "Failed to layout network."
    
    # Store distances and layout in caching panels
    cache_sq_nwdists.object = dists_pd.to_json()
    cache_sq_gp.object = pd.DataFrame(GP).to_json()
    
    return ""

@pn.depends(cache_sq_gp.param.object, input_sq_highlight.param.value, input_sq_disablehvnx.param.value)
def draw_single_network(GP, highlight, figure_style):
    # Read query result and quantile cutoffs
    try:
        query_result = pd.read_json(cache_sq_neighborhood.object).sort_values("Distance measure")
        q = pd.read_json(cache_sq_q.object, typ="series", convert_dates=False, convert_axes=False)
    except:
        return ""
    
    # Read distances and network layout from cache
    try:
        dists_pd = pd.read_json(cache_sq_nwdists.object)
        GP = pd.read_json(GP).to_dict()
        GP = {k: np.array(list(GP[k].values())) for k in GP.keys()}
    except:
        return ""
    
    # switch between figure and interactive style
    if figure_style:
        nwk = draw_network_figure(dists_pd, GP, query_result, highlight, q)
    else:
        nwk = draw_network_interactive(dists_pd, GP, query_result, highlight, q)
    
    return nwk


# In[7]:


## Assemble output tabs for single query
output_bar = pn.Column(input_sq_bary, pn.pane.Markdown("colors indicate occurence across replicate lists"), get_gene_data)
output_nwk = pn.Column(pn.Row(input_sq_minz, input_sq_maxq, input_sq_minrep),
                       pn.Row(input_sq_highlight, input_sq_disablehvnx),
                       pn.Row(draw_single_network, pn.pane.SVG("./LegendNetworksFull.svg")),
                       layout_single_network)

output_sq_tabs = pn.Tabs()
output_sq_tabs.append(("Network plot", output_nwk))
output_sq_tabs.append(("Barplot", output_bar))


# In[8]:


## Interface elements for multiple network query

# Caching panels for multiqueries
cache_mq_query = pn.pane.Str("")
cache_mq_q = pn.pane.Str("")
cache_mq_nwdists = pn.pane.Str("")
cache_mq_gp = pn.pane.Str("")
cache_mq_settings = pn.pane.Str("")

# Main options
input_mq_preselected = pn.widgets.Select(options=["CD63, CD81, CD3G", "Tetraspanins", "EV Markers from PCA"],
                                         name="Predefined \n gene lists")
input_mq_list = pn.widgets.TextAreaInput(value="CD63, CD81, CD3G", sizing_mode="fixed")
input_mq_topn = pn.widgets.IntSlider(start=20, end=100, step=5, value=30, value_throttled=30,
                                     name="Neighborhood size")
input_mq_button = pn.widgets.Button(name='Calculate neighborhoods', button_type='primary')
output_mq_status = pn.pane.Markdown("", width_policy="max", sizing_mode="stretch_width")

# Advanced options
input_mq_tolerance = pn.widgets.IntSlider(start=5, end=50, step=5, value=45, value_throttled=45,
                                          name="Replicate neighborhood tolerance")
input_mq_minr = pn.widgets.FloatSlider(start=-1, end=1, step=0.05, value=0, value_throttled=0,
                                       name="Minimal profile correlation")
input_mq_minenr = pn.widgets.FloatSlider(start=-1, end=1, step=0.05, value=-1, value_throttled=-1,
                                         name="Minimal percentile shift in F3 vs. full proteome")
input_mq_nhsize = pn.widgets.IntSlider(start=50, end=500, step=50, value=250, value_throttled=250,
                                       name="Neighborhood size for scoring")

# Tabcolumn containing options
options_multi = pn.Tabs()
options_multi.append(("Neighborhood selection", pn.Column(input_mq_preselected, 
                                                          pn.pane.Markdown("List of gene symbols to compare  \n\
                                                          (select from above for examples)"),
                                                          input_mq_list, input_mq_topn,
                                                          input_mq_button, output_mq_status)))
options_multi.append(("Advanced options", pn.Column(input_mq_tolerance, input_mq_minr,
                                                    input_mq_minenr, input_mq_nhsize)))

# Options for multi query network
input_mq_minz = pn.widgets.Select(options=["***", "**", "*", "B", "all"],
                                  name="Minimum z scoring for nodes:", value="B")
input_mq_minrep = pn.widgets.Select(options=[1, 2, 3], name="Minimum shared replicates:", value=2)
input_mq_maxq = pn.widgets.Select(options=["1%", "5%", "10%", "25%", "50%", "100%"],
                                  name="Maximum quantile for edges:", value="50%")
input_mq_highlight = pn.widgets.AutocompleteInput(options=[el for el in meta_dict["gene_id"].keys() if type(el) == str]+
                                                  ["none", "None"], name="Highlight gene:", value="None")
input_mq_disablehvnx = pn.widgets.Checkbox(value=False, name="figure style")


# In[9]:


## Functions for multi query data display

# Put in predefined gene sets
def update_genelist(event):
    l = event.new
    lg_dict = {"CD63, CD81, CD3G": "CD63, CD81, CD3G",
               "Tetraspanins": "CD81, CD9, CD63, CD151, TSPAN33, CD82, CD53, TSPAN5, TSPAN7",
               "EV Markers from PCA": "TTYH3, PLSCR3, MINK1, SELPLG, IGSF8, CD9, SLC44A1, CD53, CD82, TNFAIP3, \
                ARRDC1, ALCAM, CD63, TAOK1, MAP4K4, VPS28, PPP1R18, CD81, CHMP2A, PIP4K2A, PDCD6, SH3GL1, VPS4B, \
                CHMP2B, CHMP1A, VPS4A, ANXA7, GDI2, TRIM25, LDHA, KARS"}
    if l in lg_dict.keys():
        input_mq_list.value = lg_dict[l]
input_mq_preselected.param.watch(update_genelist, 'value')
    

# Get the multi query data -> change this to a button triggered callback and create a second callback to
# validate the input, compare the settings and enable/disable the run button accordingly
# @pn.depends(input_mq_list.param.value, input_mq_topn.param.value_throttled,
#             input_mq_tolerance.param.value_throttled, input_mq_minr.param.value_throttled,
#             input_mq_minenr.param.value_throttled, input_mq_nhsize.param.value_throttled)
@pn.depends(input_mq_list.param.value, input_mq_topn.param.value_throttled,
            input_mq_tolerance.param.value_throttled, input_mq_minr.param.value_throttled,
            input_mq_minenr.param.value_throttled, input_mq_nhsize.param.value_throttled)
def validate_mq_param(genes, size, tolerance, minr, minenr, nhsize):
    wdgts = [input_mq_preselected, input_mq_list, input_mq_topn, input_mq_tolerance,
             input_mq_minr, input_mq_minenr, input_mq_nhsize]
    for wdgt in wdgts:
        wdgt.disabled = True
    genes = [el.strip().upper() for el in input_mq_list.value.split(",")]
    genes.sort()
    for el in genes:
        if el not in meta_dict['gene_id'].keys():
            output_mq_status.object = "{} not found in the dataset. The single gene query tab has                                        an autocomplete function for all quantified genes. Use this to check,                                        which genes can be put into the mutli gene query.".format(el)
            for wdgt in wdgts:
                wdgt.disabled = False
            input_mq_button.disabled = True
            return
    oldparams = cache_mq_settings.object
    newparams = {"gene": genes, "size": size, "tolerance": tolerance, "minr": minr, "minenr": minenr, "nhsize": nhsize}
    newparams = pd.DataFrame(pd.Series(newparams)).to_json()
    if oldparams == newparams:
        output_mq_status.object = "No need to recalculate, the result for these settings is what you are looking at."
        for wdgt in wdgts:
            wdgt.disabled = False
        input_mq_button.disabled = True
        return
    else:
        output_mq_status.object = "Hit the button to start calculating the selected neighborhoods.         This can take ~0.5 min per gene."
        for wdgt in wdgts:
            wdgt.disabled = False
        input_mq_button.disabled = False
        return


def store_mq_data(event):
    wdgts = [input_mq_preselected, input_mq_list, input_mq_topn, input_mq_tolerance,
             input_mq_minr, input_mq_minenr, input_mq_nhsize]
    for wdgt in wdgts:
        wdgt.disabled = True
    input_mq_button.disabled = True
    output_mq_status.object = "Started neighborhood calculation ..."
    genes = [el.strip().upper() for el in input_mq_list.value.split(",")]
    genes.sort()
    params = {"gene": genes, "size": input_mq_topn.value_throttled, "tolerance": input_mq_tolerance.value_throttled,
              "minr": input_mq_minr.value_throttled, "minenr": input_mq_minenr.value_throttled,
              "nhsize": input_mq_nhsize.value_throttled}
    cache_mq_settings.object = pd.DataFrame(pd.Series(params)).to_json()
    pb = pn.widgets.Progress(max=len(genes), value=0, width_policy="max")
    options_multi[0].append(pb)
    output = multi_query(df_core, meta_dict, genes=genes,
                         size=params["size"], rank_tolerance=params["tolerance"], min_corr=params["minr"],
                         min_enr=params["minenr"], perc_area=params["nhsize"], pb=pb)
    options_multi[0].remove(pb)
    if len(output) == 3:
        (df, q, msg) = output
        output_mq_status.object = "Done calculating neighborhoods."
        cache_mq_q.object = q.to_json()
        cache_mq_query.object = df.to_json()
    else:
        output_mq_status.object = output
        cache_mq_q.object = ""
        cache_mq_query.object = ""
    for wdgt in wdgts:
        wdgt.disabled = False
input_mq_button.on_click(store_mq_data)


# Networkplot
@pn.depends(cache_mq_query.param.object, input_mq_minz.param.value, input_mq_maxq.param.value, input_mq_minrep.param.value)
def layout_multi_network(query, min_z, max_q, min_rep):
    # Retrieve neighborhood information
    try:
        query_result = pd.read_json(query).sort_values("Distance measure")
        q = pd.read_json(cache_mq_q.object, typ="series", convert_dates=False, convert_axes=False)
    except:
        cache_mq_nwdists.object = ""
        cache_mq_gp.object = ""
        return "Failed to read neighborhood (see settings panel for details)."
    
    # Define node list
    z_dict = {"all":5, "B":4, "*":3, "**":2, "***":1}
    min_z = z_dict[min_z]
    try:
        nwk_members = filter_nwk_members(query_result, min_rep, min_z)
    except:
        cache_mq_nwdists.object = ""
        cache_mq_gp.object = ""
        return "Failed to generate node list."
    
    # Layout network
    q_dict = {"1%":0.01, "5%":0.05, "10%":0.1, "25%":0.25, "50%":0.5, "100%":1}
    try:
        dists_pd, GP = layout_network(df_core, query_result, nwk_members, q_dict[max_q], q, layout_method="Spring")
    except:
        cache_mq_nwdists.object = ""
        cache_mq_gp.object = ""
        return "Failed to layout network."
    
    # Store distances and layout in caching panels
    cache_mq_nwdists.object = dists_pd.to_json()
    cache_mq_gp.object = pd.DataFrame(GP).to_json()
    
    return ""

@pn.depends(cache_mq_gp.param.object, input_mq_highlight.param.value, input_mq_disablehvnx.param.value)
def draw_multi_network(GP, highlight, figure_style):
    # Read query result and quantile cutoffs
    try:
        query_result = pd.read_json(cache_mq_query.object).sort_values("Distance measure")
        q = pd.read_json(cache_mq_q.object, typ="series", convert_dates=False, convert_axes=False)
    except:
        return ""
    
    # Read distances and network layout from cache
    try:
        dists_pd = pd.read_json(cache_mq_nwdists.object)
        GP = pd.read_json(GP).to_dict()
        GP = {k: np.array(list(GP[k].values())) for k in GP.keys()}
    except:
        return ""
    
    # switch between figure and interactive style
    if figure_style:
        nwk = draw_network_figure(dists_pd, GP, query_result, highlight, q)
    else:
        nwk = draw_network_interactive(dists_pd, GP, query_result, highlight, q)
    
    return nwk


# In[10]:


## Assemble output tabs for multi query
output_mq_nwk = pn.Column(pn.Row(input_mq_minz, input_mq_maxq, input_mq_minrep),
                          pn.Row(input_mq_highlight, input_mq_disablehvnx),
                          pn.Row(draw_multi_network, pn.pane.SVG("./LegendNetworksFull.svg")),
                          layout_multi_network)

output_mq_tabs = pn.Tabs()
output_mq_tabs.append(("Network plot", output_mq_nwk))


# In[11]:


## Aboutstab
about = '''
## Quick Guide:
1. Enter a gene symbol in UPPER CASE in the single query tab or several in the multi query tab.
2. Start the neighborhood calculation by clicking the button underneath.
3. The network shows the distance based relationships between the close neighbours, ideally
revealing tight clusters (try e.g. PSMA1) and peripheral proteins.
4. The barplot shows various quantitative measures for all neighbors (only in single query mode).
5. In order to understand what the advanced settings and additional parameters for the network
selection, please refer to the theoretical section below. All settings have reasonable default values.

### Useful hints
- The checkbox underneath the settings for the network layout lets you switch between the interactive network, 
which includes a toolbar to the right as provided by hvplot, and the figure style (7pt font-size) network, 
which can be copied for future reference (Download option pending).
- If network nodes are too crowded, reduce the stringency on the quantile to add back more edges that pull 
distant nodes closer together and thereby tight clusters slightly apart.
- The inclusion of neighbours found in only one neighbourhood replicate is not recommended, but can be tried 
for an low stringency explorative analysis of the data. 
- If you expect a protein to appear in the neighborhood of a query but can’t see it in the network, try 
increasing the neighborhood size and tolerance, relief the z-score stringency and use the highlight option 
on the network to find the protein.
- The single gene input autocompletes. If a protein is not found, check on Uniprot if it has a 
different primary gene name. If it is still not on the list it was not quantified sufficiently.

## Theoretical framework and interpretation of results

### Neighborhood calculation
Profile similarity between proteins is evaluated by calculating the pairwise 
Manhattan distance (absolute summed difference at every point of the two profiles). The query  
is retrieved as the top hit (with a distance of 0), and other proteins (see single query barplot) are 
shown in order of increasing distance to the query. The neighbourhood size defines how many proteins 
will be displayed in the barplot, and used for building a neighbourhood network. In the advanced 
settings quality filters can be applied to adjust the stringency and sensitivity of the analysis:
- minimum number of SILAC quantification events (to be added)
- minimum cosine correlation of profiles across repeats
- abundance percentile shift in the F3 extracellular vesicle fraction, which contains the smallest vesicles 
in the highest proportion, relative to the full proteome
Proteins not matching these criteria are removed from the dataset before any distances are calculated.

### Neighborhood evaluation
There are three measures provided to gauge which proteins are reproducibly in close proximity to the 
query, in the context of the local data density.

1. Replicate ranking  
The barplot shows the difference between the best and the worst proximity rank of a protein 
across the three replicates. If the top scoring neighbor (most likely rank 1 in at least one of the 
replicates) has a replicate range of 10 this means that is was at least on rank 11 in one of the 
other replicates. A protein is considered common to all three replicates if its overall rank is within 
the specified neighborhood size and each replicate rank is within the set neighbourhood size + ‘tolerance’, 
which can be set in the advanced settings.
2. Z-scoring of proximity  
To define the close protein neighbourhood of a query, in the context of the local dataspace, 
a simple estimate of the local profile density and distribution is performed. From the nearest 250 proteins 
(neighborhood size for scoring as set in the advanced settings), the median distance to the query 
(termed MDQ) is calculated. Next, for every one of these 250 proteins the absolute distance to the MDQ is 
determined. The median of these values corresponds to the MAD (median absolute distance to the MDQ). Each 
positive distance to the MDQ (ie the right tail of the distribution) is then transformed to a pseudo Z 
score, using a robust estimate of the standard deviation: pZ = ((distance to MDQ) – MDQ) / (1.483 x MAD). 
Profiles are categorized by these pZ scores: very close neighbours (Z>3.09, ***), close (Z>2.33, **), 
fairly close (Z>1.64, *), borderline (Z>1.28, B). Note that these categories provide 
guidance to evaluate relative profile proximity in the context of other mapped proteins nearby. However, they 
neither reflect exact boundaries for local clusters, nor exact probabilities of association with the query 
protein. While the actual distances between two proteins are identical regardless 
which is submitted as the query, the proximity guide classifier may be different because the set 
of proteins defining the distribution changes with the query. For example, an isolated protein near the 
edge of a dense cluster of profiles will retrieve some of these proteins as relatively close neighbours, but 
not vice versa. Thus, it may also be informative to consider also the absolute distances of proteins to the 
query, as indicated in the predictor output, and the distance quantile.
3. Distance quantiles  
To achieve an additional non-directional classification of distances, which is essential for network 
construction, all pairwise Manhattan-distances between the query’s 250 closest neighbours (same setting 
as for z-scoring) were calculated. The distances between the network nodes are then categorized by the 
quantile they occupy in this distribution. If a neighbor has a distance quantile of <1% it means it 
is at least as close to the query as the closest 1% of all protein pairs in that area are to each other (i.e. 
a very close neighbor). This score is non-directional, but is affected by other data structures in the 
vicinity. Again considering an isolated protein close to a protein cluster, the nearest neighbours will 
occupy a higher quantile because the pairwise distances within the protein cluster will shift the overall 
distribution towards smaller distances.

### Network generation
To visualize the complex structure of local neighbourhoods, we utilized network analysis. This can reveal 
for example if the closest neighbours of a protein form a very tight network among themselves, indicating 
a functional cluster; if the neighbourhood contains one or more tight subclusters, which may correspond to 
protein complexes; or if the local neighbourhood is very evenly distributed. These insights cannot be judged 
from a PCA plot alone, as the dimensional flattening loses information contained in the original 
nine-dimensional dataset. Similarly, the network architecture is not apparent from the linear list of nearest 
neighbours (the bar plot), as the relative proximity of the neighbours is not considered; hence, two proteins 
that are both close to the query might be quite distant from each other. The network analysis was carried 
out in python using the networkx library 
[(Hagberg et al, 2008)](https://networkx.github.io/documentation/stable/index.html). Network nodes are 
selected based on the filter criteria described above, from the neighborhood as calculated based on the 
selected settings. Nodes have to be close neighbours (by rank) in at least a defined number of replicates 
(nodes are colour coded accordingly), and have to achieve a minimal z-score, as set by the user. The distances 
between the remaining network nodes are then categorized by the quantile scoring, calculated by scoring all 
pairwise connections within this set (not just the ones to the query). All edges that fall into a higher 
quantile than the selected cut off are discarded. If a node is completely disconnected by this it will not be 
displayed in the network. All remaining distances are inverted and used as edgeweights for a force-directed 
layouting algorithm (spring network layout (Fruchterman & Reingold, 1991)), which pulls proteins with short 
distances closer together. Edges are colored by their distance percentile to visually reveal tight clusters.

### Multi query networks
To visualize the segregation/overlap of adjacent neighbourhoods, networks can be constructed from more 
than one query protein. For each of the individual queries, nodes are selected as above. Here a slight 
adjustment of parameters is useful: Fewer neighbors should be considered, to clearly separate distant queries, 
but borderline distances should be included, to allow for more connections between adjacent neighbourhoods to 
shape the network layout. The quantile boundaries are calculated for each individual set of 250 neighbors and 
then averaged across the queries. Thus, if two queries from areas of dataspace with different densities are 
selected, this will become apparent in the number and color of the edges (the sparser neighbourhood will have 
fewer/thinner connections).

## Implementation notes
This site was implemented in python using the holoviz framework and its bindings to several other python 
libraries (<https://holoviz.org>). The source code for this website and the corresponding analysis is stored 
at <https://github.com/JuliaS92/EVProfiler>, which is a private interim repository, until final acceptance of 
the manuscript. For access to the code prior to publication, please get in touch with 
Julia Schessner (schessner@biochem.mpg.de) directly or through the Journal editor, if you are a reviewer.

## Reference
Added upon publication.
'''


# In[12]:


## Assemble full app
content = pn.Tabs()
content.append(("Single gene query", pn.Row(pn.Column(options_single), output_sq_tabs)))
content.append(("Multi gene query", pn.Row(pn.Column(options_multi, validate_mq_param), output_mq_tabs)))
content.append(("About", pn.pane.Markdown(about)))

app = pn.Column("# Jurkat EV Neighbour Network Predictor Tool", content)


pwd = pn.widgets.PasswordInput(name="Please enter password for access.")
app_container = pn.Column(pwd)

def check_pwd(event, app=app):
    pwd = event.new
    if pwd == "Jurkat":
        app_container[0]=app
pwd.param.watch(check_pwd, 'value')


# In[13]:


app_container.servable()

