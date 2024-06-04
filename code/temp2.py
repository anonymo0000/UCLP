# from nltk.wsd import lesk

# from PyDictionary import PyDictionary

# import nltk
# # nltk.download('wordnet')

# word1 = "argument"
# synsets_word1 = lesk(word1, word1)

# print(synsets_word1)

# def get_computer_science_definition(word):
#     dictionary = PyDictionary()

#     try:
#         #          
#         definitions = dictionary.meaning(word)

#         #             
#         computer_science_definition = definitions#.get('Computer Science', None)

#         if computer_science_definition:
#             #             
#             print(f"          : {', '.join(computer_science_definition)}")
#         else:
#             print("             ")

#     except Exception as e:
#         print(f"         : {e}")

# #           
# user_input = "argument"
# get_computer_science_definition(user_input)


from nltk.corpus import wordnet as wn

def get_computer_science_definition(word):
    #          
    synsets = wn.synsets(word)
    # print(synsets)
    # return 
    #             
    cs_definitions = {}
    
    for synset in synsets:
        definition = synset.definition()
        print(definition)
        break
    #     flag = True
    #     #                   ，        
    #     if 'computer' in definition or 'programming' in definition:
    #         flag = False
    #         cs_definitions[synset.name()] = definition
        
    #     if flag:
    #         cs_definitions[synset.name()] = definition
            
    # return cs_definitions

#     

l1 = "field arguments parameter"
l1 = "(computer science) a set of one or more adjacent characters comprising a unit of information (computer science) a reference or value that is passed to a function, procedure, subroutine, command, or program"

l2 = "crafted data"

l3 = "execute script"

l4 = "HTTP protocol correlation"

l5 = "Call API"

for word in l1.split(" "):
    print(get_computer_science_definition(word))
# print(get_computer_science_definition('argument'))
# print(get_computer_science_definition('field'))
