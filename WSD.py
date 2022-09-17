import nltk
import torch
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

nltk.download('popular')

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True).eval()

glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')

my_dict = dict({})
for idx, key in enumerate(glove_vectors.wv.vocab):
    my_dict[key] = glove_vectors.wv.get_vector(key)

"""**Tokenizing text**"""

"""**Filtering for Verbs, Adjectives, Adverbs and Nouns**"""


def pos_tag(text):
    text = tokenizer.tokenize(text)
    return nltk.pos_tag(text)


len(pos_tag(text))

# POS tags using Penn Treebank Project keys
# big_4 = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WRB']
big_4 = ['NN', 'NNS', 'NNP', 'NNPS']
stop_words = set(stopwords.words('english'))


def get_word_vec(text, idx):
    # tokenizing the text
    tokens = tokenizer.tokenize(text)
    tokens_to_ids = tokenizer.convert_tokens_to_ids(tokens)
    torch.eval()
    # deactivates the gradient calculations, saves memory, and speeds up computation
    tokens_tensor = torch.tensor([tokens_to_ids])
    model.eval()
    similarities = list()
    # extracting embeddedings creating by BERT in the last layer (12th) for the given word
    with torch.no_grad():
        output = model(tokens_tensor)
        for i in range(1, 13):
            last_hidden_state = output[2][i]
            word_embed_1 = last_hidden_state
            # index represents the index the word occured at in the text
            # this lets us grab the correct word embedding
            ret_vec = word_embed_1[0][idx].reshape(1, -1)
            return ret_vec


def format_bert_tokens(exam1):
    true_index_count = 0
    word_map = {}
    last_was_sub = False
    tokens_bert = tokenizer.tokenize(exam1)
    # we need to find the index the word is in within these subwords. I am grouping together words that are being split into a sublist. I need to average these vectors when comparing to the other word. 
    # I look for the word in this list, if it isn't split into subwords I grab the real index of it in the list and use that position as the position of the word in the tokenized list
    # if its a subword, get all the items with the same numbers in the list, and the one occuring right before it and average the vectors, use that to compare it to the other word  
    for idx, i in enumerate(tokens_bert):
        if '##' in i:
            word_map[idx] = true_index_count
            word_map[idx - 1] = true_index_count
            last_was_sub = True
            continue
        if last_was_sub and not '##' in i:
            true_index_count += 1
        word_map[idx] = true_index_count
        true_index_count += 1
        last_was_sub = False
    return word_map


def vector_index_dict(word_map, exam1):
    vector_dict = {}

    for real_index, norm_index in word_map.items():

        sub_word_vec = []

        if real_index == 0 or real_index == 1 and norm_index == word_map[real_index + 1]:
            vector_dict[0] = torch.stack([get_word_vec(exam1, 0), get_word_vec(exam1, 1)]).mean(dim=0)

            continue

        if real_index > 0 and real_index < len(word_map) - 1:

            if norm_index == word_map[real_index - 1] or norm_index == word_map[real_index + 1]:

                # then its a sub word
                for real, norm in word_map.items():

                    if norm == norm_index:
                        sub_word_vec.append(get_word_vec(exam1, real))
                ##
                ## average vectors here, replace for sub_word_vec
                y = torch.stack(sub_word_vec).mean(dim=0)
                vector_dict[real_index] = y

            else:

                vector_dict[real_index] = get_word_vec(exam1, real_index)

        else:

            vector_dict[real_index] = get_word_vec(exam1, real_index)

    return vector_dict


def unique_tensors(vector_dict):
    tensors = [v for v in vector_dict.values()]
    unique_tensors = []
    for idx, i in enumerate(tensors):
        amount = 0
        for x in tensors:
            if cosine_similarity(i, x) > .99:
                amount += 1
                if amount > 1:
                    tensors.pop(idx)
    return tensors


vec_dict = vector_index_dict(format_bert_tokens(text), text)
uni_tensors = unique_tensors(vec_dict)
tensors = [v for v in vec_dict.values()]

sims = list()

# new variable created to remove dependencies for quick changes
normalized_clean_str = text
tagged_text = pos_tag(text)

# looping through the index and elements of word tokenized text
for index, i in enumerate(word_lst):
    # only running if the tag associated with the word is in the defined 'big_4' list
    if tagged_text[index][1] in big_4 and i not in stop_words:

        local_sims = [{index: i}]
        # getting the word vector using BERT (contains contextual embeddings) of each word in the loop
        # main_word_vec = get_word_vec(normalized_clean_str, index)
        # generating a word vector for each word in the list and comparing it to the original word
        for position, x in enumerate(word_lst):
            if tagged_text[position][1] in big_4 and x not in stop_words:
                try:
                    # checking cosine similarity of vectors generator by the W2V model
                    similarity = cosine_similarity(word_2_vec(i).reshape(1, -1), word_2_vec(x).reshape(1, -1))
                    # checked_word_vec = get_word_vec(normalized_clean_str, position)
                    # similarity = cosine_similarity(main_word_vec, checked_word_vec)
                    print('checking: ', i, x, 'Similarity: ', similarity[0][0])
                except KeyError:
                    continue
                # making sure not comparing to the same exact word
                if not position == index:
                    # if the cosine similarity is above .7, we deem the word as similar 
                    if similarity > .6:
                        # if the similairty is high, cross check with BERT to make sure the context does not change the meaning

                        # main_word_vec = get_word_vec(normalized_clean_str, index)
                        # checked_word_vec = get_word_vec(normalized_clean_str, position)
                        # bert_similarity = cosine_similarity(main_word_vec, checked_word_vec)
                        # print('________________________')
                        # print(i, x, index, position, 'bert sim:', bert_similarity)
                        bert_similarity = cosine_similarity(tensors[index], tensors[position])
                        # print('________________________')
                        # if the similarity remains high, add it to datastructure
                        if bert_similarity > .6:
                            local_sims.append({x: position})
                    # else:

                    # print(index, '|', position,':',i,'|',x)

        sims.append(local_sims)

"""**Creating function to generate contextually aware synonyms using the Wordnet dataset**"""


def compare_syn_vec(word, word_vec):
    syns = wn.synsets(word)
    # creating a dictionary to store the example of the synonym generated and
    # the position the word occurs in the synonym example
    examples = {}
    # generates a list containing only the lemmas for the wordnet synonym generations
    solo_syns = [i.lemmas()[0].name() for i in wn.synsets(word)]
    syns_simscore = {}
    location = None
    # looping through every synonym generated by wordnet and its index
    for idx, i in enumerate(syns):
        # generating examples, if the length of the list the example occurs in is 0
        # there is no example and the loop shall skip this iteration
        if len(i.examples()) == 0:
            continue
        # getting one example of the synonym used in a sentence from the first index
        try:
            for pos, x in enumerate(word_tokenize(i.examples()[1])):
                if fuzz.ratio(x, solo_syns[idx]) > 85:
                    location = pos
                    examples[' '.join(word_tokenize(i.examples()[1]))] = pos
                    continue
        # getting on example of the synonym used in a sentence from the 0th index
        # this is done because the 0th index example did not always have the syn used
        # in the sentence. So the first is originally used, but if only one example
        # is given, there is an IndexError and a try/except block must be used to then
        # use the first example given
        except IndexError:
            try:
                for pos, x in enumerate(word_tokenize(i.examples()[0])):
                    if fuzz.ratio(x, solo_syns[idx]) > 85:
                        location = pos
                        examples[' '.join(word_tokenize(i.examples()[0]))] = pos
                        continue
            except IndexError:
                continue
    # updating the syns_simscore dictionary with the key as a tuple of (the original word, the synonym, the position the synonym occurs in the example
    # the value is the result of calling the cosine similairty function on the word vector of the original word and the
    # word vector of the embedded synonym using context (both done with BERT's self attention blocks) 
    for idx, (k, v) in enumerate(examples.items()):
        syns_simscore[(word, solo_syns[idx], idx)] = cosine_similarity(word_vec, get_word_vec(k, v))[0][0]
    return syns_simscore
