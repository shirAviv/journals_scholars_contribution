from datetime import date,datetime,timedelta
import pandas as pd
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import math
from journals import Journals



class Neg_contrib:
    num_papers_total = 0
    num_cites_total = 0

    def full_neg(self,df_authors):
        df_authors['neg_contrib_full']=0
        df_authors['neg_contrib_full']=df_authors.apply(lambda row: pd.Series(self.calc_full_neg(row['Num papers'],row['Full'])),axis=1)
        print(df_authors)

    def frac_neg(self,df_authors):
        df_authors['neg_contrib_frac']=0
        df_authors['neg_contrib_frac']=df_authors.apply(lambda row: pd.Series(self.calc_frac_neg(row['Num papers'],row['Fractional'],row['Coauthors'])),axis=1)
        print(df_authors)


    def calc_full_neg(self,num_papers,num_cites):
        no_author_val=(self.num_cites_total - num_cites) / (self.num_papers_total - num_papers)
        IF=self.num_cites_total/self.num_papers_total
        IF_author=IF-no_author_val
        return IF_author

    def calc_frac_neg(self,num_papers,frac_cites, co_authors):
        IF=self.num_cites_total/self.num_papers_total
        num_co_authors = co_authors.count(';')
        num_frac_papers=num_papers / num_co_authors
        if num_papers>1:
            num_frac_papers=0
            co_authors=co_authors.split(' ')
            if (len(co_authors)!=num_papers):
                print('error for {}'.format(frac_cites))
            for authors in co_authors:
                num_frac_papers+=(1/authors.count(';'))

        IF_no_author=(self.num_cites_total-frac_cites)/(self.num_papers_total-num_frac_papers)
        return IF-IF_no_author




if __name__ == '__main__':
    journals = Journals()
    neg_contrib=Neg_contrib()
    for current_journal in journals.dirsIS:
        current_journal = journals.dirsIS[0]
        # file_path = r'~/Shir/Research/IS/'+current_journal
        file_path = ''+current_journal
        file_name_authors = '/' + 'shap_full_star_all_authors.csv'

        df_authors = pd.read_csv(file_path + file_name_authors, index_col=0)
        df_authors.reset_index(inplace=True)
        file_name_papers = '/' + current_journal+'.csv'  #
        df_papers = pd.read_csv(file_path + file_name_papers, index_col=0)
        df_papers.reset_index(inplace=True)
        neg_contrib.num_papers_total=len(df_papers)
        neg_contrib.num_cites_total=np.sum(df_papers['Cited by'])
        neg_contrib.full_neg(df_authors)
        print('num papers total {}'.format(neg_contrib.num_papers_total))
        print('num cites total {}'.format(neg_contrib.num_cites_total))
        neg_contrib.frac_neg(df_authors)
        df_authors.to_csv(file_path + file_name_authors)
        break







