import pandas as pd

def save_data(inputpath, outputpath='Dataset\\Bazate\\test', columns_to_drop=[]):
    df = pd.read_csv(inputpath)
    df = df.dropna(axis=1, how='all')  
    df['units'] = pd.to_numeric(df['units'], errors='coerce').fillna(0).astype(int)

    # Original pivot operation
    data_pivoted = df.pivot_table(index='donor_id', columns='name', values='data', aggfunc='last').reset_index()

    # Select only the unique 'donor_id' and 'units' columns from the original DataFrame
    units_column = df[['donor_id', 'units']].drop_duplicates(subset='donor_id')



    # Merge the 'units' column into the pivoted DataFrame
    data_pivoted = data_pivoted.merge(units_column, on='donor_id', how='left')
    vaccine_response = df[['donor_id', 'vaccine_response']].drop_duplicates()
    age = df[['donor_id', 'visit_age']].drop_duplicates()
    gender = df[['donor_id', 'gender']].drop_duplicates()
    race = df[['donor_id', 'race']].drop_duplicates()
    data = data_pivoted.merge(vaccine_response, on='donor_id', how='left')
    data = data.merge(age, on='donor_id', how='left')
    data = data.merge(gender, on='donor_id', how='left')
    data = data.merge(race, on='donor_id', how='left')




    
    
    data = data[(data["vaccine_response"].isna() == False)]
    nr_drp = 0
    nr_fill = 0
    pr =  2/4
    for column in data.columns:
        if data[column].isna().sum() >= len(data[column])*pr:
            data.drop(column, axis=1, inplace=True)
            nr_drp += 1
        elif data[column].isna().sum() < len(data[column])*pr and data[column].dtype in ['float64', 'int64']:
            median_value = data[column].median()  # Calculate the median
            data[column].fillna(median_value, inplace=True)  # Fill the missing values with the median
            nr_fill += 1

    nr_drp2 = 0
    for column in columns_to_drop:
        if column in data.columns:
            data.drop(column, axis=1, inplace=True)
            nr_drp2 += 1

    data.to_csv(outputpath + '.csv', index=False)  

    with open(outputpath + '.txt', 'w', encoding='utf-8') as f:
        f.write('Droped columns because too much NaN: ' + str(nr_drp) + '\n')
        f.write('Filled columns: ' + str(nr_fill) + '\n')
        f.write('Droped columns because feature importance is 0: ' + str(nr_drp2) + '\n')
        f.write('Drop if NaN > ' + str(pr*100) + '%\n')

    
    print(f"\n\n\nColumns dropped: {nr_drp}\nColumns filled: {nr_fill}") 


input_path = 'Dataset\\FluPRINT_database\\fluprint_export.csv'
output_path_test = 'Dataset\\Bazate\\test'
output_path1 = 'Dataset\\Bazate\\NaN50'
output_path2 = 'Dataset\\Bazate\\NaN75'
output_path3 = 'Dataset\\Bazate\\NaN25'
output_path4 = 'Dataset\\Bazate\\NaN25_DT_FI0'
output_path5 = 'Dataset\\Bazate\\NaN50_DT_FI0'
columns_to_drop1 = ["gamma-delta T cells", 'central memory CD8+ T cells', 'effector CD4+ T cells', 'effector CD8+ T cells', 'effector memory CD8+ T cells', 'plasmablasts',
        'naive CD8+ T cells', 'Tregs', 'memory B cells', 'transitional B cells', 'monocytes', 'naive B cells', 'naive CD4+ T cells', 'central memory CD4+ T cells',
        'IgD-CD27+ B cells', 'T cells', 'NKT cells', 'NK cells', 'IgD-CD27- B cells', 'IgD+CD27- B cells', 'HLADR-CD38+CD8+ T cells', 'HLADR-CD38+CD4+ T cells', 
        'CD85j+CD8+ T cells', 'CD8+ T cells',   
                'CD4+ T cells',    
          'CD28+CD8+ T cells',    
         'CD161-CD45RA+ Tregs',    
          'CD161+CD8+ T cells',   
         'CD161+CD45RA- Tregs',    
        'units']
columns_to_drop2 = [ 'L50_MIP1B',
                     'L50_SCF',
                     'L50_IL8',
                    'L50_TNFA',
                    'L50_TGFB',
                    'L50_TGFA',
                    'L50_IP10',
                  'L50_LEPTIN',
                     'L50_LIF',
                     'L50_NGF',
                    'L50_MCP1',
                  'L50_RANTES',
                  'L50_PDGFBB',
                    'L50_PAI1',
                    'L50_MCP3',
                    'L50_MCSF',
                   'L50_MIP1A',
                    'L50_TNFB',
            'PD1+CD4+ T cells',
                   'L50_VCAM1',
              'memory B cells',
                       'units',
        'transitional B cells',
                'plasmablasts',
          'naive CD8+ T cells',
          'naive CD4+ T cells',
               'naive B cells',
                   'monocytes',
         'gamma-delta T cells',
                    'L50_VEGF',
'effector memory CD4+ T cells',
       'effector CD8+ T cells',
       'effector CD4+ T cells',
 'central memory CD8+ T cells',
 'central memory CD4+ T cells',
                       'Tregs',
                     'T cells',
                     'L50_IL6',
                   'NKT cells',
                     'L50_IL7',
                     'B cells',
                     'L50_IL5',
           'IgD-CD27- B cells',
           'IgD+CD27- B cells',
           'IgD+CD27+ B cells',
     'HLADR-CD38+CD8+ T cells',
     'HLADR+CD38-CD4+ T cells',
     'HLADR+CD38+CD8+ T cells',
     'HLADR+CD38+CD4+ T cells',
             'HLADR+ NK cells',
           'CD94+CD8+ T cells',
           'CD94+CD4+ T cells',
          'CD85j+CD8+ T cells',
                'CD8+ T cells',
           'CD4+CD28+ T cells',
           'CD4+CD27+ T cells',
                'CD4+ T cells',
           'CD28+CD8+ T cells',
           'CD27+CD8+ T cells',
         'CD161-CD45RA+ Tregs',
          'CD161+CD8+ T cells',
         'CD161+CD45RA- Tregs',
         'CD161+CD45RA+ Tregs',
          'CD161+CD4+ T cells',
           'IgD-CD27+ B cells',
                   'L50_CD40L',
                     'L50_IL4',
                   'L50_ENA78',
                     'L50_IL2',
             'CD16+ monocytes',
                    'L50_IL1B',
                    'L50_IL1A',
                   'L50_IL17F',
                    'L50_IL17',
                    'L50_IL15',
                    'L50_IL13',
                 'L50_IL12P70',
                 'L50_IL12P40',
                    'L50_IL10',
                    'L50_IFNG',
                    'L50_IFNB',
                    'L50_IFNA',
                   'L50_ICAM1',
                    'L50_GROA',
                   'L50_GMCSF',
                    'L50_GCSF',
                    'L50_FGFB',
                    'L50_FASL',
                 'L50_EOTAXIN',
                   'L50_IL1RA']
#save_data(input_path,outputpath=output_path5,columns_to_drop=columns_to_drop2)