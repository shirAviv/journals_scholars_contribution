import pandas as pd
import itertools
#from scipy import misc
import math
import numpy as np
# from pybliometrics.scopus import AuthorRetrieval
from itertools import chain,combinations
from datetime import date,datetime,timedelta
import random



class Fractional_And_Full_value:

    def get_single_author_fractional_value(self,author, author_data):
        author_contrib = 0
        author_num_papers=len(author_data['papers'].values[0])
        for idx, paper_citations in enumerate(author_data['Num citations'].values[0]):
            # count_coauthors_in_current_subset=0
            coauthors_set = set()
            coauthors = author_data['Coauthors'].iloc[0][idx].split(';')
            coauthors.remove('')
            num_coauthors = len(coauthors)
            author_contrib += paper_citations / (num_coauthors)
        # author_contrib=author_contrib/author_num_papers
        return author_contrib

    def get_authors_fractional_values(self,authors_df):
        authors = set(authors_df.loc[:, 'Author Id'])
        for author in authors:
            author_data = authors_df[authors_df['Author Id'] == author]

            author_fractional_value=self.get_single_author_fractional_value(author,author_data)
            authors_df.loc[authors_df['Author Id'] == author, 'Fractional'] = author_fractional_value


    def get_single_author_full_value(self,author_data):
        author_contrib = 0
        author_num_papers=len(author_data['papers'].values[0])
        for idx, paper_citations in enumerate(author_data['Num citations'].values[0]):
            author_contrib += paper_citations
        # author_contrib = author_contrib / author_num_papers
        author_contrib=author_contrib
        return author_contrib

    def get_authors_full_values(self,authors_df):
        authors = set(authors_df.loc[:, 'Author Id'])
        for author in authors:
            author_data = authors_df[authors_df['Author Id'] == author]

            author_fractional_value = self.get_single_author_full_value(author_data)
            authors_df.loc[authors_df['Author Id'] == author, 'Full'] = author_fractional_value


if __name__ == '__main__':
    start_time = datetime.now()
    print(start_time)
    file_path=''
    ffv=Fractional_And_Full_value()

    end_time=datetime.now()
    print(end_time)



