# write your code here
import en_core_web_sm
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from collections import Counter
from nltk.corpus import words

path = input() or "test/clockwork_orange.txt"

with open(path) as f:
    text = f.read()

en_sm_model = en_core_web_sm.load()

matrix = np.array(['Token', 'Lemma', 'POS', "Entity_type", "IOB_tag"])

doc = en_sm_model(text)

for token in doc:
    if not any(char in token.text for char in "><_/*\\") and token.text.strip() and token.text != '\n':
        matrix = np.vstack([matrix, [token.text, token.lemma_, token.pos_, token.ent_type_, token.ent_iob_]])

df = pd.DataFrame(matrix[1::], columns=matrix[0])

# Count multi-word named entities
multi_word_entities = sum(1 for ent in doc.ents if len(ent) > 1)
print(f"Number of multi-word named entities: {multi_word_entities}")

# Count lemmas 'devotchka'
devotchka_count = df[df["Lemma"] == 'devotchka']["Lemma"].count()
print(f"Number of lemmas 'devotchka': {devotchka_count}")

# Count tokens with the stem 'milk'
milk_count = sum(1 for token in doc if 'milk' in token.text)
print(f"Number of tokens with the stem 'milk': {milk_count}")

# Most recurring named entity type
entity_tags = [token.ent_type_ for token in doc if token.ent_type_]
most_recurring_entity_tag = max(set(entity_tags), key=entity_tags.count)
print(f"Most frequent entity type: {most_recurring_entity_tag}")

# Most frequent named entity
most_frequent_entity = df[df['Entity_type'] != '']['Token'].value_counts().idxmax()
most_frequent_entity_tag = df[df['Token'] == most_frequent_entity]['Entity_type'].iloc[0]
print(f"Most frequent named entity token: ('{most_frequent_entity}', '{most_frequent_entity_tag}')")

# Ten most common non-English words
filtered_df = df[(df['Lemma'].str.len() > 4) &
                 (df['POS'].isin(['ADJ', 'ADV', 'NOUN', 'VERB'])) &
                 (~df['Lemma'].isin(words.words()))]


lemma_counts = Counter(filtered_df['Lemma'])
most_common_non_english_words = lemma_counts.most_common(10)

print("Most common non-English words:", most_common_non_english_words)

# Calculate correlation
df['POS_binary'] = df['POS'].apply(lambda x: 1 if x in ['NOUN', 'PROPN'] else 0)
df['Entity_binary'] = df['Entity_type'].apply(lambda x: 1 if x else 0)
correlation = pearsonr(df['POS_binary'].astype(float), df['Entity_binary'].astype(float))[0]
print(f"Correlation between NOUN and PROPN and named entities: {correlation:.2f}")
