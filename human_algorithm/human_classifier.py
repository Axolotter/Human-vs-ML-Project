def human_sort(acc, surg, smoke):
    if(acc == 0):
        if(smoke == 1 and surg==0):
            return "O"
        elif(surg == 1 and smoke != 1):
            return "O"
        else:
            return "N"
    else:
        return "N"