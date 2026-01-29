

from ucimlrepo import fetch_ucirepo 

def get_data():  
    #=========Copied from UCIML==========#
    # fetch dataset 
    fertility = fetch_ucirepo(id=244) 
    
    # data (as pandas dataframes) 
    X = fertility.data.features 
    y = fertility.data.targets 
    
    # metadata 
    #print(fertility.metadata) 
    
    # variable information 
    #print(fertility.variables) 
    #====================================

    feature_names = fertility.variables[fertility.variables["role"] == 'Feature']["name"].tolist()
    target_name = fertility.variables[fertility.variables["role"] == "Target"]["name"].values[0]

    df = pd.DataFrame(fertility.data.features, columns=feature_names)
    df[target_name] = fertility.data.targets

    return df, target_name

