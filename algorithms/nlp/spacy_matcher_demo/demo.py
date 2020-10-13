import os

import spacy
# from spacy.lang.en import English
from spacy.pipeline import EntityRuler

# nlp = English()
nlp = spacy.blank("en")
ruler = EntityRuler(nlp, validate=True)


def load_patterns(entity_file_dir):
    patts = []
    for txt in os.listdir(entity_file_dir):
        entity_type = txt.split('.')[0]
        with open(os.path.join(entity_file_dir, txt), 'r', encoding='utf8') as f:
            for ent_record in f:
                ent_info = ent_record.strip().split(',')
                single_pattern = {"label": entity_type,
                                  "pattern": [{"LOWER": t.lower()} for t in ent_info[0].strip().split()]}
                # it means the current entity has parrent entity
                if len(ent_info) > 1:
                    single_pattern.update({"id": "::".join(ent_info[1:])})
                patts.append(single_pattern)

    return patts


patterns = load_patterns('./entity_files')

ruler.add_patterns(patterns)
nlp.add_pipe(ruler)


def recognize_ents(text):
    doc1 = nlp(text)
    ent_res = []
    for ent in doc1.ents:
        if "::" in ent.ent_id_:
            potential_parent = ent.ent_id_.split("::")
            
            parrent_ent_type = 'none'
            # determine parrent entity type based on current entity type 
            if ent.label_ == 'car_model':
                parrent_ent_type = "car_brand"
            if ent.label_ == 'city':
                parrent_ent_type = "region"
            # append current entity
            ent_res.append({"entity": ent.text, "type": ent.label_, "id": potential_parent[0]})
            
            # parrent entity has no id
            if len(potential_parent) == 2:
                ent_res.append({"entity": ent.text, "type": parrent_ent_type})
            
            # parrent entity has id 
            if len(potential_parent) == 3:
                ent_res.append({"entity": potential_parent[1], "type": parrent_ent_type, "id": potential_parent[2]})
        # current entity has id
        elif ent.ent_id_:
            ent_res.append({"entity": ent.text, "type": ent.label_, "id": ent.ent_id_})
        # current entity has no id
        else:
            ent_res.append({"entity": ent.text, "type": ent.label_})
    # print(ent_res)
    return ent_res


test_texts = ["rent car in Los Angeles",
              "rent a Mercedes Benz in Los Angeles",
              "rent Benz in Los Angeles",
              "rent bmw m3 in New York",
              "tickets to the great wall"]
for t in test_texts:
    recognize_ents(t)

# %timeit recognize_ents(test_texts[0])
# 51 µs ± 1.25 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

