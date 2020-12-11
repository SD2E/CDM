import time
import warnings
import operator
import numpy as np
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from harness.utils.parsing_results import *
from cdm_src.utils.names import Names as N
from cdm_src.cdm_base_class import CombinatorialDesignModel


class HostResponseModel(CombinatorialDesignModel):
    def __init__(self, initial_data=None, output_path=".", leaderboard_query=None,
                 exp_condition_cols=None, target_col="logFC",
                 custom_future_conditions: Optional[pd.DataFrame] = None,
                 gene_col="gene"):
        """

        :param initial_data:
        :param output_path:
        :param leaderboard_query:
        :param exp_condition_cols:
        :param target_col:
        :param custom_future_conditions: None, or a DataFrame with exp_condition_cols as its columns.
                                         Each row should represent a condition. Rows do not have to be unique,
                                         as the code will ignore duplicates. This variable is used by the user
                                         to give a custom set of conditions to predict on when they don't want
                                         the default of all possible conditions to be predicted.
        :param gene_col:
        """
        self.per_condition_index_col = gene_col
        self.gene_network_df = None
        super().__init__(initial_data, output_path, leaderboard_query,
                         exp_condition_cols, target_col, custom_future_conditions)

    def add_index_per_existing_condition(self, initial_data):
        """
        This intentionally doesn't do anything because HostResponseModel data already has genes in it.
        """
        return initial_data

    def add_index_per_future_condition(self, future_conditions):
        """
        In this HostResponseModel, this method will add all the genes that exist in the self.existing_data to
        the self.future_data DataFrame. For each condition in future_conditions, all genes are combined with
        that condition, making each row a combination of conditions and gene.
        :param future_conditions: a list of future conditions returned from the self.retrieve_future_conditions method
        :type future_conditions: list
        :return: DataFrame with all the combinations of experimental conditions and genes
        :rtype: Pandas DataFrame
        """
        future_conditions_df = pd.DataFrame(future_conditions, columns=self.exp_condition_cols)
        unique_genes = self.existing_data[self.per_condition_index_col].unique().tolist()
        future_conditions_df[self.per_condition_index_col] = [unique_genes for i in range(len(future_conditions_df))]
        future_conditions_df = future_conditions_df.explode(self.per_condition_index_col).reset_index(drop=True)
        return future_conditions_df

    def set_impact_col(self, FDR_threshold: float = 0.05, logFC_threshold: float = 1.1):
        """
        Adds (or updates) the impact column of both self.existing_data and self.combined_df (if they exist)
        :param nlogFDR_threshold: Threshold for nlogFDR to consider as impacted.
                                  Will look for nlogFDR values less than this threshold.
        :param logFC_threshold: Threshold for logFC to consider as impacted.
                                Will look for logFC values with absolute value greater than this threshold.
        :return:
        """
        if hasattr(self, "existing_data"):
            df = self.existing_data
            if isinstance(df, pd.DataFrame):
                df[N.impact_col] = False
                df.loc[(df["FDR"] < FDR_threshold) &
                       (np.abs(df["logFC"]) > logFC_threshold), N.impact_col] = True
        if hasattr(self, "combined_df"):
            df = self.combined_df
            if isinstance(df, pd.DataFrame):
                print("combined_df")
                df[N.impact_col] = False
                df.loc[(df["FDR"] < FDR_threshold) &
                       (np.abs(df["logFC"]) > logFC_threshold), N.impact_col] = True

    def _align_predictions_with_new_data(self, predictions_df, new_data_df):
        new_data_cols = list(new_data_df.columns.values)
        first_cols = self.feature_and_index_cols + [self.target_col]
        last_cols = list(set(new_data_cols).difference(set(first_cols)))
        new_data_df = new_data_df[first_cols + last_cols]
        merged_df = pd.merge(new_data_df, predictions_df, how="inner",
                             on=self.feature_and_index_cols)
        len_preds = len(predictions_df)
        len_new_data = len(new_data_df)
        len_merged = len(merged_df)
        print("Use the following print lines to ensure the inner merge is working correctly...")
        print("len(predictions_df): {}\n"
              "len(new_data_df): {}\n"
              "len(merged_df): {}\n".format(len_preds, len_new_data, len_merged))
        return merged_df

    def score(self, x, y):
        return r2_score(x, y)

    def embed_prior_network(self, df_network=None, src_node='Source', tgt_node='Target', attrs=['Weight'],
                            emb_dim=32, workers=4, debug=False, write_out=False):
        '''
        Provide a dataframe in the form of an edge list and embed the network
        :param df_network: pd.Dataframe, dataframe of network. if nothing is passed in
        :param src_node: str, source column name in dataframe
        :param tgt_node: str, target column name in dataframe
        :param weight: list of attributes to attach to edges
        :param emb_dim: int, embedding dimension
        :param workers: number of workers to use (TACC/MAC can go up to 16)
        :param debug: boolean, set to True if you want to run in Debug mode
        :return:
        '''

        if df_network is not None:
            from node2vec import Node2Vec
            import networkx as nx

            G = nx.convert_matrix.from_pandas_edgelist(df_network, src_node, tgt_node, attrs)
            if not debug:
                print("Building model...")
                node2vec = Node2Vec(G, dimensions=emb_dim, walk_length=30, num_walks=200, workers=workers)
                print()
                print("Fitting model...")
                model = node2vec.fit(window=10, min_count=1, batch_words=4)
            else:
                print("-" * 20, 'Entering DEBUG Mode', '-' * 20)
                print("Building model...")
                node2vec = Node2Vec(G, dimensions=emb_dim, walk_length=5, num_walks=10, workers=workers)  # For debugging purposes
                print("Fitting model...")
                model = node2vec.fit(window=2, min_count=1, batch_words=2)  # For debugging purposes

            df_emb = pd.DataFrame(np.asarray(model.wv.vectors), columns=['embcol_' + str(i) for i in range(emb_dim)], index=G.nodes)
            df_emb.reset_index(inplace=True)
            df_emb.rename({df_emb.columns[0]: self.per_condition_index_col}, axis=1, inplace=True)
            df_emb['emb_present'] = 1

            if write_out:
                fname = os.path.join(self.output_path, 'network_embedding.csv')
                df_emb.to_csv(fname, index=False)
        else:
            try:
                fname = os.path.join(self.output_path, 'network_embedding.csv')
                df_emb = pd.read_csv(fname)
            except:
                raise ValueError('No previous network embedding found. You must provide a df_network.')

        self.existing_data = pd.merge(self.existing_data, df_emb, how='left', on=self.per_condition_index_col)
        self.existing_data['emb_present'].fillna(0, inplace=True)
        self.future_data = pd.merge(self.future_data, df_emb, how='left', on=self.per_condition_index_col)
        self.future_data['emb_present'].fillna(0, inplace=True)
        self.feature_and_index_cols = self.feature_and_index_cols + ['embcol_' + str(i) for i in range(emb_dim)]
        print('Embedding added to dataframe')
        print()

    def generate_gene_network_df(self, network_df: pd.DataFrame, num_gene_2: int = 5,
                                 edge_exists_ratio: float = 1.0, min_rows_with_no_edge=5):
        """
        Generates a DataFrame with all genes in one column, and a subsample of those genes (determined by num_gene_2) in
        the gene_2 column. The edge_present column indicates if there was an edge in the network_df between gene and gene_2.
        :param network_df: DataFrame containing the edge network information.
        :param num_gene_2: The number of genes to randomly subsample from the gene column and use in the gene_2 column.
        :param edge_exists_ratio: For each experimental condition and gene, the ratio you want between rows with edge_present = 0
                                  and rows with edge_present = 1. Since the number of rows with edge_present = 0 is much greater
                                  than the number of rows with edge_present = 1, The code will randomly down-sample the rows
                                  with edge_present = 0 in order to meet the desired ratio (also see min_rows_with_no_edge).
                                  Note that the ratio can be greater than 1.0 too.
        :param min_rows_with_no_edge: For each experimental condition and gene, this is the minimum number of rows that will be
                                      sampled from rows with edge_present = 5. Only kicks in if the amount derived from
                                      edge_exists_ratio is less than min_rows_with_no_edge.
        """
        unique_genes = np.unique(self.existing_data[self.per_condition_index_col])
        if num_gene_2 >= len(unique_genes):
            warnings.warn("The value you chose for num_gene_2 is greater than or equal to the number of unique genes.\n"
                          "All unique genes will be used instead of a random subset.")
            random_subset_of_genes = unique_genes.copy()
        else:
            random_subset_of_genes = list(np.random.choice(unique_genes, size=num_gene_2, replace=False))

        start_time = time.time()
        final_df = pd.concat([self.existing_data.assign(gene_2=g) for g in random_subset_of_genes], ignore_index=True)
        print("\nloop 1 took {} seconds.".format(round(time.time() - start_time, 2)))

        network_df = network_df[["Source", "Target"]]
        edges = list(zip(network_df["Source"], network_df["Target"])) + list(zip(network_df["Target"], network_df["Source"]))

        final_df["gene_pair"] = list(zip(final_df["gene"], final_df["gene_2"]))
        final_df["edge_present"] = 0

        start_time = time.time()
        final_df.loc[final_df["gene_pair"].isin(edges), "edge_present"] = 1
        print("loop 2 took {} seconds.".format(round(time.time() - start_time, 2)))
        final_df.drop(columns=["gene_pair"], inplace=True)
        final_df["(logFC, edge_present)"] = list(zip(final_df["logFC"], final_df["edge_present"]))

        col_order_beginning = ["gene", "FDR", "nlogFDR", "logFC", "gene_2", "edge_present", "(logFC, edge_present)"]
        col_order = col_order_beginning + [c for c in list(final_df.columns) if c not in col_order_beginning]
        final_df = final_df[col_order]

        # apply edge_exists_ratio here for each experimental condition
        ones = final_df.loc[final_df["edge_present"] == 1].reset_index(drop=True)

        start_time = time.time()
        g = final_df.groupby(self.exp_condition_cols + [self.per_condition_index_col])

        def apply_me(x):
            num_present_edges = sum(x["edge_present"])
            num_samples = max(int(edge_exists_ratio * num_present_edges), min_rows_with_no_edge)
            # print(int(edge_exists_ratio * num_present_edges), min_rows_with_no_edge, num_samples)
            x_new = x.loc[x["edge_present"] == 0].sample(num_samples)
            return x_new

        zeros = g.apply(apply_me).reset_index(drop=True)
        print("loop 3 took {} seconds.\n".format(round(time.time() - start_time, 2)))

        final_df = pd.concat([ones, zeros])
        # uncomment these prints to validate that you're getting what you expect
        # print(final_df["edge_present"].value_counts(), "\n")
        # print(final_df.groupby(self.exp_condition_cols).count(), "\n")
        # print(final_df.groupby(self.exp_condition_cols + [self.per_condition_index_col]).count(), "\n")

        self.gene_network_df = final_df.copy()
