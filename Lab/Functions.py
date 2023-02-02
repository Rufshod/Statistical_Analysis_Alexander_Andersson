
# Renaming function
def func_rename(dataframe, oldname:str,  newname:str):
    """Function for renaming something in a dataframe, enter the dataframe, oldname and new name. """
    return dataframe["make"].replace(oldname, newname, inplace=True) # Renaming and returning the dataframe

#Functions for converting
def func_conversion(dataframe, column:str):
    """Functions to convert pounds to kg and mpg to liter per 100km"""
    if column == "weight":          #Checks the type of conversion
        for p in range(len(dataframe)):
            dataframe[column][p] = round(dataframe[column][p]*0.45359237) #Correct math for converting
        return dataframe
    elif column == "lper100km": #Checks for type of conversion
        for p in range(len(dataframe)):
            dataframe[column][p] = round(235.215 / dataframe[column][p],2) #Correct math for converting 
        return dataframe
    else: return print("Something went wrong")
