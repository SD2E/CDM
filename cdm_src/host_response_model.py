import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from harness.utils.parsing_results import *
from cdm_src.cdm_base_class import CombinatorialDesignModel


class HostResponseModel(CombinatorialDesignModel):
    def __init__(self, initial_data=None, output_path=".", leaderboard_query=None,
                 exp_condition_cols=None, target_col="logFC", gene_col="gene"):
        self.per_condition_index_col = gene_col
        super().__init__(initial_data, output_path, leaderboard_query,
                         exp_condition_cols, target_col)

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

    def align_predictions_with_new_data(self, predictions_df, new_data_df):
        new_data_df = new_data_df[self.feature_and_index_cols + [self.target_col]]
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

    def embed_prior_network(self, df_network, src_node='Source', tgt_node='Target', attrs=['Weight'],
                            emb_dim=32):
        '''
        Provide a dataframe in the form of an edge list and embed the network
        :param df_network:
        :param src_node:
        :param tgt_node:
        :param weight:
        :return:
        '''

        from node2vec import Node2Vec
        import networkx as nx

        G = nx.convert_matrix.from_pandas_edgelist(df_network, src_node, tgt_node, attrs)
        node2vec = Node2Vec(G, dimensions=emb_dim, walk_length=30, num_walks=200, workers=4)
        print("Fitting model...")
        print()
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        df_emb = pd.DataFrame(np.asarray(model.wv.vectors), columns=['embcol_' + str(i) for i in range(emb_dim)], index=G.nodes)
        df_emb.reset_index(inplace=True)
        df_emb.rename({df_emb.columns[0]: self.per_condition_index_col}, axis=1, inplace=True)
        # TODO needd to join to all other datasets --
        return df_emb
