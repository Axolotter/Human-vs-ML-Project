def human_sort(df):
    acc = df['accident']
    surg = df['surgical_intervention']
    smoke = df['smoking']
    if(acc.eq(0)):
        if(smoke.eq(1) & surg.eq(0)):
            return "O"
        elif(surg.eq(1)) & smoke.ne(1):
            return "O"
        else:
            return "N"
    else:
        return "N"