"""
Fonctions utilitaires
"""

import json

def export_dict_to_json(dict, output_path:str):
     """export d'un dict en json"""
     with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(dict, json_file, indent=4, ensure_ascii=False)

def import_json_to_dict(input_path:str):
    """import d'un json en dict"""
    with open(input_path, 'r', encoding='utf-8') as json_file:
            dict_test = json.load(json_file)
    return dict_test