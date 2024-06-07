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
        print(f"One or both words not in vocabulary: {e}")
        return None

# Example usage
word1 = 'king'
word2 = 'queen'

similarity_score = calculate_similarity(word1, word2)

if similarity_score is not None:
    print(f"Similarity between '{word1}' and '{word2}': {similarity_score}")
else:
    print("Unable to calculate similarity.")
