from nltk.corpus import wordnet as wn

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# Load pre-trained Word2Vec model (you can download it from https://code.google.com/archive/p/word2vec/)
# In this example, I'm using the pre-trained Google News Word2Vec model.
model_path = '/home/user/lh/PMA_lihang/code_lihang/word2vec/model/GoogleNews-vectors-negative300.bin'  # Update the path to the Word2Vec model file
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

def calculate_similarity(word1, word2):
    try:
        similarity = word2vec_model.similarity(word1, word2)
        return similarity
    except KeyError as e:
        # print(f"One or both words not in vocabulary: {e}")
        return 0

word_domain = "computer science"

def get_computer_science_definition(word):
    #          
    synsets = wn.synsets(word)
    
    #             
    cs_definitions = ""
    max_similarity_score = 0

    for synset in synsets:
        definition = synset.definition()
        print(definition)
        
        #                   ，        
        if 'computer' in definition or 'programming' in definition:
            cs_definitions = definition
    print("------------------")

    if cs_definitions == "":
        for synset in synsets:
        #     
            similarity_score = 0
            definition = synset.definition()
            for word1 in word_domain.split(" "):
                for word2 in definition.split(" "):
                    # print(word1,word2)
                    similarity_score += calculate_similarity(word1, word2)
                    # print(f"Similarity between '{word1}' and '{word2}': {calculate_similarity(word1, word2)}")
            if similarity_score > max_similarity_score:
                print(f"Similarity between '{word_domain}' and '{definition}': {similarity_score}")
                max_similarity_score = similarity_score
                cs_definitions = definition

    return cs_definitions

#     
print(get_computer_science_definition('data'))
