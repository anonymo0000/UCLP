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
# s1 = "chained initialization vectors"
s1 = "command"

l1 = "field arguments parameter"

l2 = "crafted data"

l3 = "execute script"

l4 = "HTTP protocol correlation"

l5 = "Call API"


word1 = 'name'
word2 = 'field'
word2 = 'argument'
# word2 = 'parameter'


for ln in [l1,l2,l3,l4,l5]:
    similarity_score = 0
    for word2 in ln.split(" "):
        for word1 in s1.split(" "):
            # print(word1,word2)
            similarity_score += calculate_similarity(word1, word2)
            print(f"Similarity between '{word1}' and '{word2}': {calculate_similarity(word1, word2)}")
    print(f"Similarity between '{s1}' and '{ln}': {similarity_score}")




# # Example usage
# word1 = 'king'
# word2 = 'queen'

# similarity_score = calculate_similarity(word1, word2)

# if similarity_score is not None:
#     print(f"Similarity between '{word1}' and '{word2}': {similarity_score}")
# else:
#     print("Unable to calculate similarity.")
