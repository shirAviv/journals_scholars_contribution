import pandas as pd
import math
import random
from datetime import date,datetime,timedelta
from fractional_and_full_value import Fractional_And_Full_value
from journals import Journals


class Journal_shapley_value_full_star:
    eps = 0.05
    delta = 0.05
    all_permutations = dict()
    duplicate_permutations_counter = 0
    all_coalitions = dict()
    duplicate_coalitions_counter = 0
    num_pcs=15

    def init_permutations(self):
        self.duplicate_permutations_counter = 0
        for idx in range(0, 500):
            self.all_permutations[idx] = list()
        self.duplicate_coalitions_counter = 0
        for idx in range(0, 100):
            self.all_coalitions[idx] = set()

    def extract_data(self, file):
        df=pd.read_csv(file)
        df=df.loc[df['Document Type'].isin(['Article','Review','Conference Paper'])]
        df['Cited by']=df['Cited by'].fillna(0)
        print(df.head())
        return df

    def get_authors_df(self,df):
        authors = set(''.join(list(df.loc[:, 'Author(s) ID'])).split(';'))
        authors.remove('')
        authors_df = pd.DataFrame(columns=['Author Name', 'Author Id', 'papers','Num papers', 'Num citations', 'Coauthors'])
        index = 0

        for author in authors:
            author_data = df[df['Author(s) ID'].str.contains(author)]
            author_papers = author_data.index.values
            author_citations = author_data.loc[:, 'Cited by'].values
            paper_count = len(author_papers)

            id_location = list(author_data['Author(s) ID'].str.split(';').values)[0].index(author)
            author_name = list(author_data['Authors'].str.split(',').values)[0][id_location].strip()
            author_coauthors = pd.DataFrame(author_data.loc[:, 'Author(s) ID']).copy()
            record = {'Author Name': author_name, 'Author Id': author, 'papers': author_papers,'Num papers':paper_count,
                      'Num citations': author_citations, 'Coauthors': author_coauthors['Author(s) ID'].values}
            authors_df = authors_df.append(record, ignore_index=True)
        return authors_df

    def gen_permutation(self,authors_df, size=0):
        permutation = list(authors_df.loc[:, 'Author Id'])
        exists = False
        #while exists:
        random.shuffle(permutation)
            # for author in authors:
            #     val=random.getrandbits(1)
            #     if val:
            #         permutation.append(author)
            # exists = self.does_permutation_exist(permutation)
        return permutation

    def does_permutation_exist(self,permutation):
        permutation_size=len(permutation)
        permutations_of_size=self.all_permutations[permutation_size]
        if permutation in permutations_of_size:
            self.duplicate_permutations_counter+=1
            return True
        else:
            self.all_permutations[permutation_size].append(permutation)
            return False

    def value_single_permutation_full(self,permutation, authors_df, column_name):

        value_coalition_without_current_author=0
        total_val = 0
        papers=set()
        for author_id in permutation:

            author_data = authors_df[authors_df['Author Id'] == author_id]
            authors_contrib = 0
            # sum up citations for current author, only for papers not counted yet
            for idx, paper_id in enumerate(author_data['papers'].values[0]):
                if not paper_id in papers:
                    authors_contrib += author_data['Num citations'].values[0][idx]
                    papers.add(paper_id)

            total_val += authors_contrib

            value_coalition_with_current_author = total_val / len(papers)
            author_marginal_contribution = (value_coalition_with_current_author - value_coalition_without_current_author)
            authors_df.loc[authors_df['Author Id'] == author_id, column_name] += author_marginal_contribution
            value_coalition_without_current_author=value_coalition_with_current_author


    def confidence_shapley(self,df_authors, column_name):
        self.init_permutations()

        x=0
        k=0

        min_num_samples=(2*math.pow(citation_to_papers_ratio,2)*math.log(2/self.delta,math.e))/(math.pow(self.eps,2))
        print('total num permutations {}'.format(min_num_samples))
        min_num_samples=min_num_samples/self.num_pcs
        print('total num permutations {}'.format(min_num_samples))

        print('total num permutations {}'.format(min_num_samples))

        while k<min_num_samples:
            permutation=self.gen_permutation(df_authors)
            self.value_single_permutation_full(permutation,df_authors, column_name=column_name)
            if k%2000==0:
                print('num samples is {}'.format(k))
            k+=1
        authors_df[column_name]=authors_df[column_name]/min_num_samples
        return


if __name__ == '__main__':
    journals = Journals()
    name = journals.dirsIS[-1]
    file_path = r''
    file_name=file_path+name+'\\'+name+'.csv'
    print(file_name)
    js=Journal_shapley_value_full_star()
    df=js.extract_data(file_name)
    authors_df = js.get_authors_df(df)
    ffv = Fractional_And_Full_value()
    ffv.get_authors_fractional_values(authors_df)
    ffv.get_authors_full_values(authors_df)

    authors_df['citation_to_papers_ratio']=authors_df['Full']/authors_df['Num papers']
    print(authors_df.sort_values(by='citation_to_papers_ratio').tail(30))
    citation_to_papers_ratio = authors_df['citation_to_papers_ratio'].max()
    # citation_to_papers_ratio=49
    print(file_name)

    print('num authors {}'.format(len(authors_df)))
    print(citation_to_papers_ratio)
    column_name='citation_to_papers_ratio'
    column_name='Shapley_star'
    authors_df[column_name]=0
    print(datetime.now())
    js.confidence_shapley(authors_df, column_name=column_name)
    authors_df = authors_df.sort_values(by=column_name, ascending=False)
    print(authors_df)
    print(authors_df[column_name].sum())
    # authors_df.to_csv('shap_subsets_10_2.csv')
    authors_df.to_csv(file_path+name+'\\student-shap_full_star_all_authors.csv')
    print(datetime.now())


    exit(0)

    # df_for_shapley_papers=js.remove_authors_with_low_num_papers(df)

    # authors_df=js.calc_authors_shapley_num_citations(df)
    # authors_df=authors_df.sort_values(by="Shapley - Num citations",ascending=False)
    # js.get_authors_hindex_affiliation(authors_df)
    # authors_df.to_csv('C:\\Users\\Shir\\OneDrive - Bar Ilan University\\research\\Journals_data\\PUCScopus_2019_Shapley_full_star_citations.csv')
    # js.calc_total_combinations(21)
