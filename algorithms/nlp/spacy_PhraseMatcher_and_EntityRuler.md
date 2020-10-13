#### 1. spacy PhraseMatcher
```python
import spacy
from spacy.matcher import PhraseMatcher

terms = ["Great Wall Hover", "GAC Trumpchi", "Baic motor"]

nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser'])
matcher = PhraseMatcher(nlp1.vocab, attr="LOWER")

# patterns = [nlp(text) for text in terms]
# patterns = [nlp.make_doc(text) for text in terms]
patterns = list(nlp.tokenizer.pipe(terms))
matcher.add("TerminologyList", None, *patterns)

doc = nlp("I want to rent a Great wall Hover or GAC Trumpchi")
matches = matcher(doc)
macthed_phrases = [doc[start:end] for match_id, start, end in matches]

print(macthed_phrases)
# [Great wall Hover, GAC Trumpchi]
```

#### 2. spacy EntityRuler
```python
from spacy.lang.en import English
from spacy.pipeline import EntityRuler

nlp = English()
ruler = EntityRuler(nlp)
patterns = [
    {"label": "city", "pattern": [{"LOWER": "los"}, {"LOWER": "angeles"}], "id": "10992"},
    {"label": "city", "pattern": [{"LOWER": "l.a"}], "id": "10992"}
]
ruler.add_patterns(patterns)
nlp.add_pipe(ruler)

doc1 = nlp("Los angeles is a big city")
print([(ent.text, ent.label_, ent.ent_id_) for ent in doc1.ents])
# [('Los angeles', 'city', '10992')]

doc2 = nlp("L.A is a big city.")
print([(ent.text, ent.label_, ent.ent_id_) for ent in doc2.ents])
# [('L.A', 'city', '10992')]
```
对于有简单层级关系的实体匹配，如车系和品牌的对应关系，可以考虑把父级实体处理在EnrityRuler的id中，当识别到子实体时，通过解析其id还原父级实体，例如： 
```json
{"label": "city", "pattern": [{"LOWER": "l.a"}], "id": "10992_region_America_10158"}
```
其中10992表示L.A城市id，region表示上一级实体的类型为国家，America表示上一级实体为美国，10158表示美国的国家id

