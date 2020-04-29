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


df_pca = pd.read_csv("PCA cooridnates Jurkat Exosomes 13 April 2020t.txt", sep='\t')
df_pca.columns
df_pca.columns = ['Primary ID', 'ClassID', 'Lead gene name',
       'Protein names', 'Gene names',
       'Fasta headers', 'Sec. ID',
       'Organelle', 'Min Ratio count', 'Class',
       'PCA 1', 'PCA2']


# In[4]:


## Backend functions
def hv_query_df(df_core, meta_dict, gene, dist="Manhattan",
                min_enr=-1, min_corr=0, size=50, rank_tolerance=25, perc_area=250):
    size += 1
    if gene.upper() not in meta_dict["gene_id"].keys():
        return "Gene not in dataset"
    
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
        return "The query protein was filtered out."
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


# Plotting functions
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
    var = label_var_dict[var]
    if var == "z-scoring based category":
        z_dict = {"-":5, "B":4, "*":3, "**":2, "***":1, np.nan:0, None:0}
        q_m["z-scoring based category"] = [z_dict[k] for k in q_m["z-scoring based category"]]
        z_dict = {"-":5, "B":4, "*":3, "**":2, "***":1, "query":0}
        p = q_m.hvplot.bar(x="Lead gene name", y=var, color="Common lists",
                               height=500, yticks = [(z_dict[k], k) for k in z_dict.keys()])
    else:
        p = q_m.hvplot.bar(x="Lead gene name", y=var, color="Common lists",
                               height=500)
    p.opts(xrotation=60)
    p.opts(legend_position="top")
    return p

def filter_nwk_members(query_result, min_rep, max_z):
    nwk_members_1 = query_result[query_result["Common lists"] >= min_rep]["id"].values
    z_dict = {"-":5, "B":4, "*":3, "**":2, "***":1, np.nan:0, None:0}
    query_result["z-scoring based category"] = [z_dict[k] for k in query_result["z-scoring based category"]]
    nwk_members_2 = query_result[query_result["z-scoring based category"] <= max_z]["id"].values
    nwk_members = [el for el in nwk_members_1 if el in nwk_members_2]
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

def draw_network_figure(dists_pd, GP, query_result, highlight, q):
    # Regerate network graph from distances
    G = nx.Graph()
    for idx in range(len(dists_pd)):
        G.add_edge(dists_pd.iloc[idx,0], dists_pd.iloc[idx,1], weight=dists_pd.iloc[idx,2],
                   quantile=dists_pd.iloc[idx,3])

    fig = plt.figure(figsize=(7,7))

    # edges
    edge_styles = [{"width": 5, "edge_color": "darkred", "style": "solid", "alpha": 1}, # 0.1%
                   {"width": 5, "edge_color": "darkred", "style": "solid", "alpha": 1}, # 0.5%
                   {"width": 5, "edge_color": "darkred", "style": "solid", "alpha": 1}, # 1%
                   {"width": 4, "edge_color": "red", "style": "solid", "alpha": 1}, # 2.5%
                   {"width": 4, "edge_color": "red", "style": "solid", "alpha": 1}, # 5%
                   {"width": 3, "edge_color": "darkorange", "style": "solid", "alpha": 1}, # 10%
                   {"width": 2, "edge_color": "#ababab", "style": "solid", "alpha": 1}, # 25%
                   {"width": 2, "edge_color": "lightgrey", "style": "solid", "alpha": 1}, # 50%
                   {"width": 2, "edge_color": "lightgrey", "style": "solid", "alpha": 1}] # 100%
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
    
    Gi = nx.Graph([(hit,hit) for hit, rep in zip(query_result["Lead gene name"], query_result["Common lists"]) if rep==1 and hit in G.nodes()])
    labels0 = hvnx.draw(Gi, GP, node_color="white", node_size=2500, with_labels=True,
                        width=width, height=width)
    Gi = nx.Graph([(hit,hit) for hit, rep in zip(query_result["Lead gene name"], query_result["Common lists"]) if rep==2 and hit in G.nodes()])
    labels1 = hvnx.draw(Gi, GP, node_color="lightgrey", node_size=2500, with_labels=True,
                        width=width, height=width)
    Gi = nx.Graph([(hit,hit) for hit, rep in zip(query_result["Lead gene name"], query_result["Common lists"]) if rep==3 and hit in G.nodes()])
    labels2 = hvnx.draw(Gi, GP, node_color="orange", node_size=2500, with_labels=True,
                        width=width, height=width)
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
    
    nw = hv.Overlay(edges[::-1]) * labels0 * labels1 * labels2 * labels3
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
                                             name="Select gene:", value="CD63")
input_sq_topn = pn.widgets.IntSlider(start=20, end=100, step=5, value=50, value_throttled=50,
                                     name="Neighborhood size")

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
options_single.append(("Neighborhood selection", pn.Column(input_sq_gene, input_sq_topn)))
options_single.append(("Advanced options", pn.Column(input_sq_tolerance, input_sq_minr,
                                                     input_sq_minenr, input_sq_nhsize)))

# Options for network display
input_sq_minz = pn.widgets.Select(options=["***", "**", "*", "B", "all"],
                                  name="Minimum z scoring for nodes:", value="**")
input_sq_minrep = pn.widgets.Select(options=[1, 2, 3], name="Minimum shared replicates:", value=2)
input_sq_maxq = pn.widgets.Select(options=["1%", "5%", "10%", "25%", "50%", "100%"],
                                  name="Minimum quantile for edges:", value="50%")
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
@pn.depends(input_sq_gene.param.value, input_sq_topn.param.value_throttled,
            input_sq_tolerance.param.value_throttled, input_sq_minr.param.value_throttled,
            input_sq_minenr.param.value_throttled, input_sq_nhsize.param.value_throttled)
def store_gene_data(gene, topn, tolerance, min_corr, min_enr, nhsize):
    output = hv_query_df(df_core, meta_dict, gene=gene, size=topn, rank_tolerance=tolerance,
                        min_corr=min_corr, min_enr=min_enr, perc_area=nhsize)
    if len(output) == 3:
        (df, q, msg) = output
        cache_sq_neighborhood.object = df.to_json()
        cache_sq_q.object = q.to_json()
    else:
        msg = output
        cache_sq_neighborhood.object = ""
        cache_sq_q.object = ""
    return msg

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
@pn.depends(cache_sq_q.param.object,
            input_sq_minz.param.value, input_sq_maxq.param.value, input_sq_minrep.param.value)
def layout_single_network(q, min_z, max_q, min_rep):
    # Retrieve neighborhood information
    try:
        query_result = pd.read_json(cache_sq_neighborhood.object).sort_values("Distance measure")
        q = pd.read_json(q, typ="series", convert_dates=False, convert_axes=False)
    except:
        cache_sq_nwdists.object = ""
        cache_sq_gp.object = ""
        return "Failed to read neighborhood (see setings panel for details)."
    
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

#@pn.depends(cache_sq_gp.param.object, cache_sq_q.param.object
#def draw_single_network()


# In[7]:


## Assemble output tabs for single query
output_bar = pn.Column(input_sq_bary, get_gene_data)
output_nwk = pn.Column(pn.Row(input_sq_minz, input_sq_maxq, input_sq_minrep),
                       pn.Row(input_sq_disablehvnx, input_sq_highlight),
                       draw_single_network, layout_single_network)

output_sq_tabs = pn.Tabs()
output_sq_tabs.append(("Network plot", output_nwk))
output_sq_tabs.append(("Barplot", output_bar))


# In[8]:


## Assemble full app
content = pn.Tabs()
content.append(("Single gene query", pn.Row(pn.Column(options_single, store_gene_data), output_sq_tabs)))

app = pn.Column("# Exosome neighborhood analysis", content)


pwd = pn.widgets.PasswordInput(name="Please enter password for access.")
app_container = pn.Column(pwd)

def check_pwd(event, app=app):
    pwd = event.new
    if pwd == "Jurkat":
        app_container[0]=app
pwd.param.watch(check_pwd, 'value')


# In[9]:


app_container.servable()

