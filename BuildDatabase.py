import pandas as pd
from neo4j import GraphDatabase
import os
import codecs

def BuildNeo4j(data_path):

    uri = os.getenv("neo4j_uri")
    username = os.getenv("neo4j_username")
    password = os.getenv("neo4j_password")

    driver = GraphDatabase.driver(codecs.decode(uri, 'unicode_escape'), auth=(username, password))
    session = driver.session()
    # ## Clean All
    # session.run("MATCH (n) DETACH DELETE n")

    ### ---  read triples
    df = pd.read_csv(data_path, sep='\t', header=None, names=['head', 'relation', 'tail'])

    for index, row in df.iterrows():
        head_name = row['head']
        tail_name = row['tail']
        relation_name = row['relation']

        query = (
            "MERGE (h:Entity { name: $head_name }) "
            "MERGE (t:Entity { name: $tail_name }) "
            "MERGE (h)-[r:`" + relation_name + "`]->(t)"
        )
        session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)
    
    ### --- Add Label
    label_dict = {
        "can_check_disease": ["Disease"],
        "has_symptom": ["Symptom"],
        "need_medical_test": ["Medical_Test"],
        "need_medication": ["Medication"],
        "possible_cure_disease": ["Disease"],
        "possible_disease": ["Disease"]
    }

    for relations in label_dict.keys():
        query = (
            f"MATCH ()-[r:{relations}]->(end)"
            f"SET end:{label_dict[relations][0]}"
        )
        session.run(query)
    
    print("Neo4j build successfully.")