import pandas as pd
import numpy as np
import helpers.file_merger as fm
import seaborn as sns
import matplotlib.pyplot as plt
import statannot
import math
import itertools
from sklearn.cluster import KMeans
from matplotlib.pyplot import gcf
import matplotlib as mpl
import math
mpl.rcParams.update({'font.size': 8})

def display_heatmap(df,value,index=['normalization','batch_reduction'],ax=False,annot=False,pivot=True,n_cluster=False,figsize=(5,5), fontsize=8):
    if pivot:
        heatmap_df = df.pivot_table(index = index,columns=['model'],values=value)
    else: 
        heatmap_df = df

    if n_cluster:
        kmeans = KMeans(n_clusters=n_cluster, random_state=26).fit(heatmap_df.drop(['Mean', 'SD'], axis=1).apply(lambda row: row.fillna(row.mean()), axis=1))
        labels = kmeans.labels_ + 1
        unique = set(labels)
        pal = sns.color_palette('Set2',n_colors=len(set(kmeans.labels_)))
        lut = dict(zip(map(int, sorted(unique)), pal))

        colors = pd.Series(labels).map(lut)
        colors.index = heatmap_df.index

        g = sns.clustermap(heatmap_df, row_cluster=False, col_cluster=False, figsize=figsize, row_colors=colors, cmap=sns.color_palette("Blues", as_cmap=True),
                                       vmin=0,vmax=100,xticklabels=True, yticklabels=True,annot=True, annot_kws={"size": 6})
        g.ax_heatmap.axvline(heatmap_df.shape[1] - 2, color='w', lw=2)
        # plt.tight_layout()
        for label in set(labels):
            g.ax_col_dendrogram.bar(0, 0, color=lut[label], label=label, linewidth=0)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation='horizontal', fontsize = 8)
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation='45', ha='right')
        # plt.setp(g.ax_heatmap.set_xticklabels(), g.ax_heatmap.get_xmajorticklabels(), )
        plt.subplots_adjust(bottom=0.5)
        g.cax.set_position([.1, .46, .03, .45])
        l1 = g.ax_col_dendrogram.legend(title='Cluster', loc="center", ncol=5, bbox_to_anchor=(0.5, .92), bbox_transform=gcf().transFigure)

    else:
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['xtick.labelsize'] = 8
        fontsize_pt = plt.rcParams['ytick.labelsize']
        fontsize_pt = plt.rcParams['xtick.labelsize']
        
        g=sns.heatmap(heatmap_df, cmap="Blues",vmin=0,vmax=100,xticklabels=True, yticklabels=True,annot=annot, ax=ax, annot_kws={"size": 6},cbar=False)
        ax.axvline(heatmap_df.shape[1] - 2, color='w', lw=1)
        ax.set_xticklabels(rotation=45, ha='right',labels=heatmap_df.columns)
        # ax_heatmap.axvline(heatmap_df.shape[1] - 2, color='w', lw=4)
        if isinstance(heatmap_df.index, pd.MultiIndex):
            y_labels = ['-'.join(col).strip() for col in df.index.values]
        else:
            y_labels = df.index.values
        ax.set_yticklabels(rotation='horizontal', labels=y_labels)
        ax.set_ylabel(None)
        ax.set_xlabel(None)
        ax.set_title("{}".format(value))


#Stats annotation using https://github.com/webermarcolivier/statannot/tree/master/statannot

def display_boxplot(df,feature,metric,order,palette,ax,base='',label=True,median=False,fontsize=8,verbose=0):
    
    if base:
        feat = list(set(df[feature].values.tolist()))
        feat.remove(base)
        combos = list(zip(feat, itertools.cycle([base])))
    else:
        combos = list(itertools.combinations(set(df[feature].values.tolist()), 2))
    
    plot = sns.boxplot(data=df,x=feature,y=metric,order=order,palette=palette,ax=ax,fliersize=0.5,linewidth=0.5)
    
    if median:
        medians = round(df.groupby([feature])[metric].median(),2)

        medians = medians.reindex(order)

        for xtick in plot.get_xticks():        
            plot.text(xtick,medians[xtick]*1.01,medians[xtick], 
                horizontalalignment='center',size='x-small',color='black',weight='semibold')
    
    if label:
        statannot.add_stat_annotation(plot, data=df, x=feature, y=metric, order=order, box_pairs=combos,
                        test='Mann-Whitney', text_format='star', loc='outside', verbose=verbose)
    
    if metric == 'MCC':
        x=np.min(df[metric].values.tolist())
        lower = int(math.floor(x / 10.0)) * 10
        plot.set_ylim([lower,100])
    else:
        plot.set_ylim([0,100])           
    plot.set_xlabel(None)
    plot.set_xticklabels(rotation=45,labels=['Zero-Centering', 'No Batch \n Reduction', 'MMUPHin #1', 'MMUPHin #2'], ha='right',fontsize = fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)



def get_generalizable_datapreps(df, metric,index=[], columns=['model'], percent = 0.1, top_only = False):
#This function gets the top 10% data preparations, using the mean and variance of the metric score selected to rank them
    
    df_pivoted = df.pivot_table(index = index,columns=columns,values=metric)
   
    df2 = pd.DataFrame()
    df2['Mean'] = df_pivoted.mean(axis=1)
    df2['SD'] = df_pivoted.std(axis=1)
    df_pivoted['Mean'] = np.round(df2['Mean'], 0)
    df_pivoted['SD'] = df2['SD']        
        
    sorted_df =  df_pivoted.sort_values(by = ['Mean', 'SD'], ascending=[False,True])
    
    #To get top 10% data preparations, we get the size of our newly indexed df
    df_height = df_pivoted.shape[0]
    
    ten_perc = math.floor(percent*df_height)

    if top_only:
        generalizers_df = sorted_df.copy().head(ten_perc)
    else:
        top_generalizers_df = sorted_df.copy().head(ten_perc)
        bottom_generalizers_df = sorted_df.copy().tail(ten_perc)

        generalizers_df = pd.concat([top_generalizers_df, bottom_generalizers_df])
        generalizers_df = generalizers_df.drop_duplicates()
    
    return generalizers_df

def sort_and_filter(sorter_column,df,column_name):
    #Sorts a dataframe using the order of a column from another dataframe
    sorterIndex = dict(zip(sorter_column, range(len(sorter_column))))
    
    df['ranking'] = df[column_name].map(sorterIndex)
    df.sort_values(by=column_name, ascending = True, inplace = True)
    sorted_df = df.dropna(subset=['ranking'])
    sorted_df.drop('ranking',axis=1,inplace=True)
    return sorted_df