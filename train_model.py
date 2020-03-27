import os
import json
import pickle
import numpy as np
from collections import Counter
from architecture import ExtSummModel


def load_data(word2idx, data_paths, data_type="train"):
    cache_dir = "cache"
    doc_path, abstract_path, labels_path = data_paths
    docs = []
    start_ends = []
    abstracts = []
    labels = []

    processed_data_dir = os.path.join(cache_dir, data_type)
    if not os.path.isdir(processed_data_dir):
        os.makedirs(processed_data_dir)
    processed_data_path = os.path.join(processed_data_dir, "data.txt")
    if os.path.exists(processed_data_path):
        with open(processed_data_path, "rb") as fp:  # Unpickling
            data = pickle.load(fp)
        return data

    # actual inputs
    doc_path = os.path.join(doc_path, data_type)
    if os.path.exists(processed_data_path):
        with open(processed_data_path, "rb") as fp:  # Unpickling
            data = pickle.load(fp)
        return data

    for file_ in os.listdir(doc_path):
        with open(os.path.join(doc_path, file_), 'r') as doc_in:
            doc_json = json.load(doc_in)
            one_doc = []
            for sentence in doc_json['inputs']:
                one_doc.append(sentence['tokens'])
            docs.append(one_doc)
            section_start = 1
            sections = []
            for section_len in doc_json['section_lengths']:
                sections.append([section_start, section_start + section_len - 1])
                section_start += section_len
            start_ends.append(np.array(sections))

    # abstracts
    abstract_path = os.path.join(abstract_path, data_type)
    for file_ in os.listdir(abstract_path):
        with open(os.path.join(abstract_path, file_), 'r') as abstract_in:
            for line in abstract_in.read().splitlines():
                abstracts.append(line)  # should only have 1 line

    # labels
    labels_path = os.path.join(labels_path, data_type)
    for file_ in os.listdir(labels_path):
        with open(os.path.join(labels_path, file_), 'r') as labels_in:
            labels_json = json.load(labels_in)
            labels.append(labels_json['labels'])
    convert_word_to_idx(word2idx, docs)
    with open(processed_data_path, "wb") as fp:  # Pickling
        pickle.dump((docs, start_ends, abstracts, labels), fp)
    return docs, start_ends, abstracts, labels


def create_embeddings(glove_dir):
    """
    :param glove_dir:
    :return: embedding_matrix, word2idx
    """
    idx = 0
    word2idx = dict()
    embedding_matrix = []

    with open(glove_dir, "r") as glove_in:
        for line in glove_in:
            line = line.split()
            word = line[0]
            word2idx[word] = idx
            idx += 1
            embedding = np.array(line[1:]).astype(np.float)
            embedding_matrix.append(embedding)

    return np.asarray(embedding_matrix), word2idx


def convert_idx_to_sent_embeddings(weight_matrix, docs):
    for k, doc in enumerate(docs):
        sent_embeddings = []
        for i, sentence in enumerate(doc):
            word_embeddings = []
            # convert all the words to its corresponding embeddings. If UNK, skip the word
            for j, word_index in enumerate(sentence):
                if word_index >= 0:
                    word_embeddings.append(weight_matrix[word_index])
            if len(word_embeddings) > 0: # FIXME this line is problematic
                word_embeddings = np.stack(word_embeddings)
                # Average of word embeddings
                sent_embeddings.append(np.mean(word_embeddings, axis=0))
        docs[k] = sent_embeddings


def get_ratio(labels):
    sampled = np.random.choice(len(labels), 50000, replace=False)
    pos = neg = 0.0
    for idx in sampled:
        counted = Counter(labels[idx])
        pos += counted[1]
        neg += counted[0]
    return neg / pos


def convert_word_to_idx(word2idx, docs):
    for doc in docs:
        for i, sentence in enumerate(doc):
            # convert all the words to its corresponding indices. If UNK, skip the word
            word_indexes = []
            for word in sentence:
                if word in word2idx:
                    word_indexes.append(word2idx[word])
            doc[i] = word_indexes


def train_model():
    glove_dir = "embeddings"
    embedding_size = 300
    weight_matrix, word2idx = create_embeddings(f"{glove_dir}/glove.6B.{embedding_size}d.txt")
    print("Created embeddings")

    # load data
    data_paths = ("arxiv/inputs/", "arxiv/human-abstracts/", "arxiv/labels/")
    # (doc, start_end, abstract, label)
    test_set = load_data(word2idx, data_paths, data_type="test")
    print("Test set loaded. Length:", len(test_set[0]))
    val_set = load_data(word2idx, data_paths, data_type="val")
    print("Val set loaded. Length:", len(val_set[0]))
    train_set = load_data(word2idx, data_paths, data_type="train")
    print("Train set loaded. Length:", len(train_set[0]))

    # compute positive-negative ratio
    neg_pos_ratio = get_ratio(train_set[-1])
    print("Negative to positive ratio is ", pos_ratio)

    # initialize model
    model = ExtSummModel(weight_matrix, word2idx, neg_pos_ratio=neg_pos_ratio)
    print("Model initialization completed")

    #convert_idx_to_sent_embeddings(weight_matrix, train_set[0])
    # train the model
    model.fit(train_set, lr=0.0001, epochs=50, batch_size=32)
    # save the model
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(curr_dir, "model")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    model.save(os.path.join(model_dir, "extsumm.bin"))


def main():
    train_model()


if __name__ == "__main__":
    main()
