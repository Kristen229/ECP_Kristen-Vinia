import os


import os

import os

def save_csv(dataframe, directory, filename):
    """
    Transforme un dataframe en fichier CSV dans le dossier et sous le nom spécifiés.
    """
    if not filename.lower().endswith('.csv'):
        filename += '.csv'
        
    path_complete = os.path.join(directory, filename)
    
    dataframe.to_csv(path_complete, index=True, encoding='utf-8-sig', sep=';')
    
    print(f" Fichier sauvegardé : {path_complete}")

def categorize_food(food_name):
    """
    Classe un nom d'aliment dans une catégorie (Cereals, Oils, Sugar, Other).
    """
    name_lower = food_name.lower()

    cereals_keywords = ['maize', 'rice', 'sorghum', 'millet', 'wheat', 'beans', 'peas', 'groundnuts', 'soybeans']
    oils_keywords = ['oil', 'groundnut', 'palm', 'nut']
    sugar_keywords = ['sugar'] # Aucun dans votre liste, mais inclus pour complétude
    meat_keywords = ['meat']
    dairy_keywords = ['milk', 'cheese']


    # Vérification des catégories
    for keyword in cereals_keywords:
        if keyword in name_lower:
            if keyword in ['beans', 'peas', 'groundnuts', 'soybeans', 'bambara']:
                return 'Other' # Les légumineuses/arachides sont souvent classées séparément
            return 'Cereals'

    for keyword in oils_keywords:
        if keyword in name_lower:
            return 'Oils'

    for keyword in sugar_keywords:
        if keyword in name_lower:
            return 'Sugar'

    for keyword in meat_keywords:
        if keyword in name_lower:
            return 'Meat'

    for keyword in dairy_keywords:
        if keyword in name_lower:
            return 'Dairy'

    return 'Other'

