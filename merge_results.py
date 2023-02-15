import os
import pandas as pd
from journals import Journals


path = r''

num_pcs=1

class MergeData:
    def fix_column(self, files_path):
        num_files=0
        for file in os.listdir(path=files_path):
            if not file.endswith('student-shap_full_star_all_authors.csv'):
                continue
            num_files += 1
            obj_path = os.path.join(files_path, file)
            df_shap = pd.read_csv(obj_path)
            df_shap.rename(columns = {'citation_to_papers_ratio':'Shapley_star'}, inplace = True)
            df_shap['citation_to_papers_ratio'] = df_shap['Full'] / df_shap['Num papers']
            df_shap.to_csv(files_path + '/'+file)

    def merge(self, files_path):
        df = pd.read_csv(files_path+r'\shapley_all_authors_base.csv')
        df[column_name]=0
        num_files=0
        list_of_files = list(os.listdir(path))
        for file in os.listdir(path=files_path):
            if not file.endswith('student-shap_full_star_all_authors.csv'):
                continue
            num_files+=1
            obj_path = os.path.join(files_path, file)
            df_shap = pd.read_csv(obj_path)
            for idx, author_data in df.iterrows():
                author_id=author_data['Author Id']
                shap_value=df_shap.loc[df_shap['Author Id']==author_id, column_name].values
                df.loc[df['Author Id'] == author_id, column_name] += shap_value
        df[column_name]=df[column_name]/num_pcs

        print(df)
        print(num_files)
        df = df.sort_values(by=column_name, ascending=False)
        df.to_csv(files_path + '/shap_full_star_all_authors.csv')


if __name__ == '__main__':
    journals = Journals()
    name = journals.dirsIS[0]
    column_name='Shapley_star'
    merge=MergeData()
    merge.merge(path+name)
