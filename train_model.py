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
    processed_data_path = os.path.join(processed_data_dir, data_type + ".cache")
    if os.path.isfile(processed_data_path):
        with open(processed_data_path, "rb") as fp:  # Unpickling
            data = pickle.load(fp)
        print("> Loaded cached file for", data_type)
        return data

    # actual inputs
    doc_path = os.path.join(doc_path, data_type)
    for file_ in os.listdir(doc_path):
        with open(os.path.join(doc_path, file_), "r") as doc_in:
            doc_json = json.load(doc_in)
            one_doc = []
            for sentence in doc_json["inputs"]:
                one_doc.append(sentence["tokens"])
            docs.append(convert_doc_to_idx(word2idx, one_doc))
            section_start = 1
            sections = []
            doc_length = len(doc_json["inputs"])
            for section_len in doc_json["section_lengths"]:
                end = section_start + section_len - 1
                if end > doc_length:
                    print(f"section end {end} is larger than doc_length {doc_length} in file {file_}")
                    end = doc_length
                sections.append([section_start, end])
                section_start += section_len
            start_ends.append(np.array(sections))

    # abstracts
    abstract_path = os.path.join(abstract_path, data_type)
    for file_ in os.listdir(abstract_path):
        with open(os.path.join(abstract_path, file_), "r") as f:
            abstracts.append(f.readline())

    # labels
    labels_path = os.path.join(labels_path, data_type)
    for file_ in os.listdir(labels_path):
        with open(os.path.join(labels_path, file_), "r") as f:
            labels_json = json.load(f)
            labels.append(labels_json["labels"])

    with open(processed_data_path, "wb") as f:  # Pickling
        pickle.dump((docs, start_ends, abstracts, labels), f)
    return docs, start_ends, abstracts, labels


def convert_doc_to_idx(word2idx, doc):
    idx_doc = []
    for i, sentence in enumerate(doc):
        # convert all the words to its corresponding indices. If UNK, skip the word
        word_indexes = []
        for word in sentence:
            if word in word2idx:
                word_indexes.append(word2idx[word])
        # guard against completely empty sentences
        if len(word_indexes) == 0:
            word_indexes.append(word2idx["."])
        idx_doc.append(word_indexes)
    return idx_doc


def create_embeddings(glove_dir):
    """
    :param glove_dir:
    :return: embedding_matrix, word2idx
    """
    cache_dir = "cache"
    processed_data_dir = os.path.join(cache_dir, "embeddings")
    if not os.path.isdir(processed_data_dir):
        os.makedirs(processed_data_dir)
    processed_data_path = os.path.join(processed_data_dir, "embeddings.cache")
    if os.path.exists(processed_data_path):
        with open(processed_data_path, "rb") as fp:  # Unpickling
            data = pickle.load(fp)
        print("> Loaded cached file for embeddings")
        return data

    idx = 0
    word2idx = dict()
    embedding_matrix = []

    with open(glove_dir, "r") as glove_in:
        for line in glove_in:
            line = line.split()
            word = line[0]
            word2idx[word] = idx
            idx += 1
            embedding = np.array(line[1:]).astype(np.float32)
            embedding_matrix.append(embedding)

    with open(processed_data_path, "wb") as fp:
        pickle.dump((np.asarray(embedding_matrix), word2idx), fp)
    return np.asarray(embedding_matrix), word2idx


def get_ratio(labels):
    sampled = np.random.choice(len(labels), 5000, replace=False)
    pos = neg = 0.0
    for idx in sampled:
        counted = Counter(labels[idx])
        pos += counted[1]
        neg += counted[0]
    return neg / pos



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
    print("Data lengths in training set:", len(train_set[0]), len(train_set[1]), len(train_set[2]), len(train_set[3]))

    # compute positive-negative ratio
    neg_pos_ratio = get_ratio(test_set[-1])
    print("Negative to positive ratio is {:.2f}".format(neg_pos_ratio))

    # initialize model
    model = ExtSummModel(weight_matrix, neg_pos_ratio=neg_pos_ratio)
    print("Model initialization completed")

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(curr_dir, "model")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # train the model
    model.fit(train_set, lr=0.001, epochs=1, batch_size=32)
    model.save(os.path.join(model_dir, "extsumm-1.bin"))
    model.fit(train_set, lr=0.0008, epochs=1, batch_size=32)
    model.save(os.path.join(model_dir, "extsumm-2.bin"))
    model.fit(train_set, lr=0.0004, epochs=1, batch_size=32)
    model.save(os.path.join(model_dir, "extsumm-3.bin"))
    model.fit(train_set, lr=0.0002, epochs=1, batch_size=32)
    model.save(os.path.join(model_dir, "extsumm-4.bin"))
    model.fit(train_set, lr=0.0001, epochs=1, batch_size=32)
    model.save(os.path.join(model_dir, "extsumm-5.bin"))


def main():
    train_model()


if __name__ == "__main__":
    main()
