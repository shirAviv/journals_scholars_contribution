from datetime import date,datetime,timedelta
import pandas as pd
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import math
from journals import Journals
import os



class Comparison:

    def extract_data(self, file1, file2):
        df1=pd.read_csv(file1)
        df2=pd.read_csv(file2)

        # print(df1.head())
        # print(df2.head())

        return df1, df2

    def sort(self,df1,df2, df1_by, df2_by):
        df1_sorted=df1.sort_values(by=df1_by, ascending=[False,False], ignore_index=True, key=None)
        df2_sorted=df2.sort_values(by=df2_by, ascending=[False,False], ignore_index=True, key=None)

        # print(df1_sorted)
        # print(df2_sorted)
        return df1_sorted,df2_sorted

    def gen_ranks(self,df1,df2):

        rank_array1=list(df1.index.values)
        rank_array2 = []
        for idx in range(rank_array1[-1]+1):
            author_id=int(df1.iloc[idx,:]['Author Id'])
            df2_idx=df2.loc[df2['Author Id'] == author_id].index.values[0]
            rank_array2.append(df2_idx)
        rank_array1 = [x + 1 for x in rank_array1]
        rank_array2 = [x + 1 for x in rank_array2]
        # print(rank_array1)
        # print(rank_array2)
        return rank_array1,rank_array2

    def calc_kendall_tau(self,rank1,rank2):
        tau, p_value = stats.kendalltau(rank1, rank2)
        print('tau {}, pVal {}'.format(tau,p_value))
        return tau,p_value

    def calc_tau_for_shapley_star(self, file):

        df, df_shap_star = comp.extract_data(file, file)
        df, df_shap_star = comp.sort(df, df_shap_star, df1_by=['Fractional', 'Author Id'],
                                   df2_by=['Shapley_star', 'Author Id'])
        frac_rank, shap_star_rank = comp.gen_ranks(df, df_shap_star)
        print('kendall tau for Fractional and shapley star')
        tau_frac,p_value_frac=comp.calc_kendall_tau(frac_rank, shap_star_rank)
        df, df_shap_star = comp.sort(df, df_shap_star, df1_by=['Full', 'Author Id'],
                                          df2_by=['Shapley_star', 'Author Id'])
        full_rank, shap_star_rank = comp.gen_ranks(df, df_shap_star)
        print('kendall tau for Full and shapley star')
        tau_full,p_value_full=comp.calc_kendall_tau(full_rank, shap_star_rank)
        df, df_shap_star = comp.sort(df, df_shap_star, df1_by=['citation_to_papers_ratio', 'Author Id'],
                                     df2_by=['Shapley_star', 'Author Id'])
        ratio_rank, shap_star_rank = comp.gen_ranks(df, df_shap_star)
        print('kendall tau for Citation to paper ratio and shapley star')
        tau_cite_pap_ratio,p_value_cite_pap_ratio=comp.calc_kendall_tau(ratio_rank, shap_star_rank)
        df, df_shap_star = comp.sort(df, df_shap_star, df1_by=['Num papers', 'Author Id'],
                                     df2_by=['Shapley_star', 'Author Id'])
        # num_papers_rank, shap_star_rank = comp.gen_ranks(df, df_shap_star)
        # print('kendall tau for number of papers and shapley star')
        # comp.calc_kendall_tau(num_papers_rank, shap_star_rank)
        df, df_shap_star = comp.sort(df, df_shap_star, df1_by=['neg_contrib_full', 'Author Id'],
                                     df2_by=['Shapley_star', 'Author Id'])
        rank, shap_star_rank = comp.gen_ranks(df, df_shap_star)
        print('kendall tau for Neg contrib Full and shapley star')
        tau_neg_full, p_value_neg_full = comp.calc_kendall_tau(rank, shap_star_rank)
        df, df_shap_star = comp.sort(df, df_shap_star, df1_by=['neg_contrib_frac', 'Author Id'],
                                     df2_by=['Shapley_star', 'Author Id'])
        rank, shap_star_rank = comp.gen_ranks(df, df_shap_star)
        print('kendall tau for Neg contrib Fractional and shapley star')
        tau_neg_frac, p_value_neg_frac = comp.calc_kendall_tau(rank, shap_star_rank)

        df_full_neg, df_frac_neg = comp.sort(df, df_shap_star, df1_by=['neg_contrib_full', 'Author Id'],
                                     df2_by=['neg_contrib_frac', 'Author Id'])
        rank_neg_full, rank_neg_frac = comp.gen_ranks(df_full_neg, df_frac_neg)
        print('kendall tau for Neg contrib Full and Neg contrib fractional')
        tau_neg_full_frac, p_value_neg_full_frac = comp.calc_kendall_tau(rank_neg_full, rank_neg_frac)

        # tau_full_frac
        df_full, df_frac = comp.sort(df, df_shap_star, df1_by=['Full', 'Author Id'],
                                             df2_by=['Fractional', 'Author Id'])
        rank_full, rank_frac = comp.gen_ranks(df_full, df_frac)
        print('kendall tau for   Full and   fractional')
        tau_full_frac, p_value_full_frac = comp.calc_kendall_tau(rank_full, rank_frac)

        # tau_full_neg_full
        df_full, df_full_neg = comp.sort(df, df_shap_star, df1_by=['Full', 'Author Id'],
                                             df2_by=['neg_contrib_full', 'Author Id'])
        rank_full, rank_neg_full = comp.gen_ranks(df_full, df_full_neg)
        print('kendall tau for   Full and Neg contrib full')
        tau_full_neg_full, p_value_full_neg_full = comp.calc_kendall_tau(rank_full, rank_neg_full)

        # tau_full_neg_frac
        df_full, df_frac_neg = comp.sort(df, df_shap_star, df1_by=['Full', 'Author Id'],
                                             df2_by=['neg_contrib_frac', 'Author Id'])
        rank_full, rank_neg_frac = comp.gen_ranks(df_full, df_frac_neg)
        print('kendall tau for  Full and Neg contrib fractional')
        tau_full_neg_frac, p_value_full_neg_frac = comp.calc_kendall_tau(rank_full, rank_neg_frac)

        # tau_frac_neg_full
        df_frac, df_full_neg = comp.sort(df, df_shap_star, df1_by=['Fractional', 'Author Id'],
                                             df2_by=['neg_contrib_full', 'Author Id'])
        rank_frac, rank_neg_full = comp.gen_ranks(df_frac, df_full_neg)
        print('kendall tau for  Fractional and Neg contrib full')
        tau_frac_neg_full, p_value_frac_neg_full = comp.calc_kendall_tau(rank_frac, rank_neg_full)

        # tau_frac_neg_frac
        df_frac, df_frac_neg = comp.sort(df, df_shap_star, df1_by=['Fractional', 'Author Id'],
                                             df2_by=['neg_contrib_frac', 'Author Id'])
        rank_frac, rank_neg_frac = comp.gen_ranks(df_frac, df_frac_neg)
        print('kendall tau for Fractional and Neg contrib fractional')
        tau_frac_neg_frac, p_value_frac_neg_frac = comp.calc_kendall_tau(rank_frac, rank_neg_frac)

        IF=df['Shapley_star'].sum()
        print('Shapley star sum=IF= '+str(IF))


        record={'shapley_IF':IF,'tau_frac_shap':tau_frac,'p_value_frac_shap':p_value_frac,
                            'tau_full_shap':tau_full,'p_value_full_shap':p_value_full,
                            'tau_neg_full_shap':tau_neg_full,'p_value_neg_full_shap':p_value_neg_full,
                            'tau_neg_frac_shap':tau_neg_frac,'p_value_neg_frac_shap':p_value_neg_frac,
                            'tau_full_frac': tau_full_frac, 'p_value_full_frac': p_value_full_frac,
                            'tau_full_neg_full': tau_full_neg_full, 'p_value_full_neg_full': p_value_full_neg_full,
                            'tau_full_neg_frac': tau_full_neg_frac, 'p_value_full_neg_frac': p_value_full_neg_frac,
                            'tau_frac_neg_full': tau_frac_neg_full, 'p_value_frac_neg_full': p_value_frac_neg_full,
                            'tau_frac_neg_frac': tau_frac_neg_frac, 'p_value_frac_neg_frac': p_value_frac_neg_frac,
                            'tau_neg_full_neg_frac':tau_neg_full_frac,'p_value_neg_full_neg_frac':p_value_neg_full_frac}
        return record

    def gini(self,x):
        total = 0
        for i, xi in enumerate(x[:-1], 1):
            total += np.sum(np.abs(xi - x[i:]))
        total2 = 0
        for xi in x[:-1]:
            for xj in x[:-1]:
                total2 += np.abs(xi - xj)

        return total2 / (2*((len(x) ** 2) * np.mean(x)))

    def gini2_pos_vals(self,x):
        ## first sort
        sorted_arr = x.copy()
        sorted_arr.sort()
        n = x.size
        mean=np.mean(x)
        print(mean)
        coef_ = 2. / n
        const_ = (n + 1.) / n
        weighted_sum = sum([(i + 1) * yi for i, yi in enumerate(sorted_arr)])
        return coef_ * weighted_sum / (sorted_arr.sum()) - const_

    def gini_neg(self,x):

        # sorted_arr = x.copy()
        # sorted_arr.sort()
        n = x.size
        total_net=x.sum()
        total_pos_net=x.loc[x>0].sum()
        total_neg_net = abs((x.loc[x<0].sum()))

        total=0
        for xi in x[:-1]:
            for xj in x[:-1]:
                total += np.abs(xi - xj )

        gini=total/(2*(n-1)*(total_pos_net+total_neg_net))
        gini = total / (2 * n  * (total_pos_net + total_neg_net))
        return gini


    def lorenz_curve(self,x1, x2,x3, journal_name, label1, label2, label3):
        X1_lorenz = x1.cumsum()
        X1_lorenz = np.insert(X1_lorenz, 0, 0)
        X1_lorenz[0], X1_lorenz[-1]
        X2_lorenz = x2.cumsum()
        X2_lorenz = np.insert(X2_lorenz, 0, 0)
        X2_lorenz[0], X2_lorenz[-1]
        X3_lorenz = x3.cumsum()
        X3_lorenz = np.insert(X3_lorenz, 0, 0)
        X3_lorenz[0], X3_lorenz[-1]

        fig, ax = plt.subplots(figsize=[6, 6])
        ## scatter plot of Lorenz curve
        ax.scatter(np.arange(X1_lorenz.size) / (X1_lorenz.size - 1), X1_lorenz,
                   marker='x', color='cyan', s=20, label=label1)
        ax.scatter(np.arange(X2_lorenz.size) / (X2_lorenz.size - 1), X2_lorenz,
                   marker='o', color='darkblue', s=20, label=label2)
        ax.scatter(np.arange(X3_lorenz.size) / (X3_lorenz.size - 1), X3_lorenz,
                   marker='^', color='red', s=20, label=label3)

        ## line plot of equality
        ax.plot([0, 1], [0, x1.sum()], color='blue')
        ax.plot([0, 1], [0, x3.sum()], color='k')

        # set the x-spine
        ax.spines['left'].set_position('zero')

        # turn off the right spine/ticks
        ax.spines['right'].set_color('none')
        ax.yaxis.tick_left()

        # set the y-spine
        ax.spines['bottom'].set_position('zero')

        # turn off the top spine/ticks
        ax.spines['top'].set_color('none')
        ax.xaxis.tick_bottom()

        plt.xlabel('Cumalative share of authors marginal contribution')
        plt.ylabel('Cumalative IF')
        plt.title('Lorenz curve for authors citation contribution - '+journal_name)

        plt.legend(loc=(0.1, 0.8))

        plt.show()


    def percent_neg_authors(self,df, column='Shapley_star'):
        shap_neg=sum(df[column]<0)*100/len(df)
        return shap_neg

    def ratio_neg_values(self,df,column='Shapley_star'):
        sum_neg_values=sum(df.loc[df[column] < 0, column])
        sum_pos_values=sum(df.loc[df[column] >= 0, column])
        ratio_neg_to_pos= abs(sum_neg_values)/(sum_pos_values)
        return ratio_neg_to_pos, sum_neg_values






if __name__ == '__main__':
    start_time = datetime.now()
    print(start_time)
    comp=Comparison()
    journals = Journals()

    journals = Journals()
    df_stats=pd.DataFrame()
    record={}

    for current_journal in journals.dirsIS:
        if current_journal=='test':
            continue
        for file in os.listdir(path=journals.file_path + current_journal):
            if not file.startswith('shap_full_star_all_authors.csv'):
                continue
            print('now creating for {}'.format(current_journal))
            orig_file = os.path.join(journals.file_path + current_journal, current_journal + '.csv')
            orig_df = pd.read_csv(orig_file)
            orig_df['Cited by']=orig_df['Cited by'].fillna(0)
            IF = np.mean(orig_df['Cited by'])
            incoming_cites=np.sum(orig_df['Cited by'])
            num_papers=orig_df.shape[0]
            num_papers_less_cites_than_IF=orig_df[orig_df['Cited by']<IF]['Cited by'].count()
            obj_path = os.path.join(journals.file_path + current_journal, file)
            df_shap = pd.read_csv(obj_path)
            df_shap.reset_index(inplace=True)
            record['journal name']=current_journal
            '''

            column = 'Shapley_star'
            ratio, sum_neg = comp.ratio_neg_values(df_shap, column)
            print('neg values ratio for {} is {}%'.format(column, ratio))
            record['ratio neg to pos values' + column] = ratio
            record['sum neg values ' + column] = sum_neg

            column = 'neg_contrib_full'
            ratio, sum_neg = comp.ratio_neg_values(df_shap, column)
            print('neg values ratio for {} is {}%'.format(column, ratio))
            record['ratio neg to pos values' + column] = ratio
            record['sum neg values ' + column] = sum_neg

            column = 'neg_contrib_frac'
            ratio, sum_neg = comp.ratio_neg_values(df_shap, column)
            print('neg values ratio for {} is {}%'.format(column, ratio))
            record['ratio neg to pos values ' + column] = ratio
            record['sum neg values ' + column] = sum_neg

            df_stats = df_stats.append(record, ignore_index=True)
            continue
            '''
            record['num_authors']=len(df_shap)
            record['IF']=IF
            record['num_papers']=num_papers
            record['incoming_cites']=incoming_cites
            column='Shapley_star'
            ratio, sum_neg = comp.ratio_neg_values(df_shap, column)
            print('neg values ratio for {} is {}%'.format(column,ratio))
            record['ratio neg to pos values' + colum] = ratio
            record['sum neg values '+column] = sum_neg
            per = comp.percent_neg_authors(df_shap,column)
            print('neg authors per for {} is {}%'.format(column,per))
            record['% neg authors '+column]=per

            # record['shap IF']=shap_IF
            record['num_papers_less_cites_than_IF']=num_papers_less_cites_than_IF

            orig_file_name = os.path.join(journals.file_path + current_journal, current_journal + '.csv')
            df, df_papers_ = comp.extract_data(orig_file_name, orig_file_name)
            num_papers = len(df_papers_)
            df_shap['Full_normalized'] = df_shap['Full'] / num_papers
            df_shap['Fractional_normalized'] = df_shap['Fractional'] / num_papers
            # df.to_csv(file_path+journal_name+'\\'+'shap_full_star_all_authors.csv')

            df_shap['Shapley_star'].values.sort()
            gini = comp.gini(df_shap['Shapley_star'].values)
            record['Shapley_star gini']=gini
            print('Shapley_star gini is {}'.format(gini))
            gini = comp.gini_neg(df_shap['Shapley_star'])
            record['Shapley_star gini neg'] = gini
            print('Shapley_star gini_neg is {}'.format(gini))
            shap_values_sorted = df_shap['Shapley_star'].values.copy()
            # comp.lorenz_curve(df['Shapley_star'].values)
            df_shap['Fractional_normalized'].values.sort()
            frac_values_sorted = df_shap['Fractional_normalized'].values.copy()
            gini = comp.gini(df_shap['Fractional_normalized'].values)
            record['Fractional_normalized gini'] = gini
            print('Fractional_normalized gini is {}'.format(gini))
            gini = comp.gini_neg(df_shap['Fractional_normalized'])
            print('Fractional_normalized gini_neg is {}'.format(gini))
            df_shap['Full_normalized'].values.sort()
            full_values_sorted = df_shap['Full_normalized'].values.copy()
            gini = comp.gini(df_shap['Full_normalized'].values)
            record['Full_normalized gini'] = gini
            print('Full_normalized gini is {}'.format(gini))
            gini = comp.gini_neg(df_shap['Full_normalized'])
            print('Full_normalized gini_neg is {}'.format(gini))

            # comp.lorenz_curve(shap_values_sorted,frac_values_sorted,full_values_sorted,current_journal, label1='Shapley star', label2='Fractional', label3='Full')

            df_shap['neg_contrib_full'].values.sort()
            neg_full_values_sorted = df_shap['neg_contrib_full'].values.copy()
            gini = comp.gini(df_shap['neg_contrib_full'].values)
            print('neg_contrib_full gini is {}'.format(gini))
            gini = comp.gini_neg(df_shap['neg_contrib_full'])
            record['neg_contrib_full gini'] = gini
            print('neg_contrib_full gini_neg is {}'.format(gini))
            neg_contrib_full_total_sum = df_shap['neg_contrib_full'].sum()
            print('neg_contrib_full sum= ' + str(neg_contrib_full_total_sum))
            record['neg_contrib_full_total_sum']=neg_contrib_full_total_sum

            column = 'neg_contrib_full'
            ratio, sum_neg = comp.ratio_neg_values(df_shap, column)
            print('neg values ratio for {} is {}%'.format(column, ratio))
            record['ratio neg to pos values' + column] = ratio
            record['sum neg values ' + column] = sum_neg
            per = comp.percent_neg_authors(df_shap, column)
            print('neg authors per for {} is {}%'.format(column, per))
            record['% neg authors ' + column] = per

            df_shap['neg_contrib_frac'].values.sort()
            neg_frac_values_sorted = df_shap['neg_contrib_frac'].values.copy()
            gini = comp.gini(df_shap['neg_contrib_frac'].values)
            print('neg_contrib_frac gini is {}'.format(gini))
            gini = comp.gini_neg(df_shap['neg_contrib_frac'])
            print('neg_contrib_frac gini_neg is {}'.format(gini))
            record['neg_contrib_frac gini'] = gini
            neg_contrib_frac_total_sum = df_shap['neg_contrib_frac'].sum()
            print('neg_contrib_frac sum= ' + str(neg_contrib_frac_total_sum))
            record['neg_contrib_frac_total_sum'] = neg_contrib_frac_total_sum

            column = 'neg_contrib_frac'
            ratio, sum_neg = comp.ratio_neg_values(df_shap, column)
            print('neg values ratio for {} is {}%'.format(column, ratio))
            record['ratio neg to pos values ' + column] = ratio
            record['sum neg values ' + column] = sum_neg
            per = comp.percent_neg_authors(df_shap, column)
            print('neg authors per for {} is {}%'.format(column, per))
            record['% neg authors ' + column] = per

            df_stats=df_stats.append(record, ignore_index=True)
            # comp.lorenz_curve(shap_values_sorted,neg_frac_values_sorted,neg_full_values_sorted,current_journal, label1='Shapley star', label2='Negative contribution Fractional', label3='Negative contribution Full')

    df_stats.set_index('journal name', inplace=True)
    df_stats.to_csv(journals.file_path+'\\'+'shap_full_star_stats.csv')



'''
    for current_journal in journals.dirsIS:
        if current_journal == 'test':
            continue
        for file in os.listdir(path=journals.file_path + current_journal):
            if not file.startswith('shap_full_star_all_authors.csv'):
                continue
            print('now creating for {}'.format(current_journal))
            obj_path = os.path.join(journals.file_path + current_journal, file)
            df_shap = pd.read_csv(obj_path)
            df_shap.reset_index(inplace=True)
            comp.lorenz_curve(shap_values_sorted,frac_values_sorted,full_values_sorted,current_journal)
'''
    # df['Shapley_star_offset'].values.sort()
    # gini = comp.gini(df['Shapley_star_offset'].values)
    # print('Shapley_star_offset gini is {}'.format(gini))
    # gini = comp.gini_neg(df['Shapley_star_offset'])
    # print('Shapley_star_offset gini_neg is {}'.format(gini))
    # shap_values_sorted = df['Shapley_star_offset'].values.copy()


