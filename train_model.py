import os
import json
import pickle
import numpy as np
from architecture import ExtSummModel


def load_data(word2idx, data_paths, data_type="train"):
    cache_dir = "cache"
    doc_path, abstract_path, labels_path = data_paths
    docs = []
    start_ends = []
    abstracts = []
    labels = []

    # actual inputs
    doc_path = doc_path + data_type
    processed_data_path = os.path.join(cache_dir, data_type, "data.txt")
    if os.path.exists(processed_data_path):
        with open(processed_data_path, "rb") as fp:  # Unpickling
            data = pickle.load(fp)
        return data

    for file in os.listdir(doc_path):
        with open(os.path.join(doc_path, file), 'r') as doc_in:
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
    abstract_path = abstract_path + data_type
    for file in os.listdir(abstract_path):
        with open(os.path.join(abstract_path, file), 'r') as abstract_in:
            for line in abstract_in.read().splitlines():
                abstracts.append(line)  # should only have 1 line

    # labels
    labels_path = labels_path + data_type
    for file in os.listdir(labels_path):
        with open(os.path.join(labels_path, file), 'r') as labels_in:
            labels_json = json.load(labels_in)
            labels.append(labels_json['labels'])
    convert_word_to_idx(word2idx, docs)
    with open(processed_data_path, "wb") as fp:  # Pickling
        pickle.dump((docs, start_ends, abstracts, labels), fp)
    return docs, start_ends, abstracts, labels


def create_embeddings(glove_dir, embedding_size):
    """
    :param glove_dir:
    :param vocab: dict, the entire vocabulary from word to index, from train, test and eval set
    :return: embedding_matrix, word2idx
    """
    idx = 0
    word2idx = dict()
    embedding_matrix = []

    with open(glove_dir, "rb") as glove_in:
        for line in glove_in:
            line = line.decode().split()
            word = line[0]
            word2idx[word] = idx
            idx += 1
            embedding = np.array(line[1:]).astype(np.float)
            embedding_matrix.append(embedding)

    # last entry reserved for OoV words
    embedding_matrix.append(np.random.normal(scale=0.6, size=(embedding_size,)))
    word2idx["UNK"] = idx
    return np.asarray(embedding_matrix), word2idx


def convert_word_to_idx(word2idx, data):
    for doc, start_end, abstract, label in zip(*data):
        for i, sentence in enumerate(doc):
            # convert all the words to its corresponding indices. If UNK, assign to the last entry
            doc[i] = [word2idx[word] if word in word2idx else len(word2idx) - 1
                      for word in sentence]


def train_model():
    # Perform a forward cycle with fictitious data
    glove_dir = "embeddings"
    embedding_size = 300
    weight_matrix, word2idx = create_embeddings(f"{glove_dir}/glove.6B.{embedding_size}d.txt", embedding_size)

    model = ExtSummModel(weight_matrix, word2idx)

    data_paths = ("arxiv/inputs/", "arxiv/human-abstracts/", "arxiv/labels/")

    # (doc, start_end, abstract, label)
    train_set = load_data(word2idx, data_paths, data_type="train")
    test_set = load_data(word2idx, data_paths, data_type="test")
    val_set = load_data(word2idx, data_paths, data_type="val")

    # train the model
    model.fit(train_set, lr=0.001, epochs=50, batch_size=128)
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
