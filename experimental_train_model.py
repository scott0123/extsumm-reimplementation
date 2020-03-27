import os
import json
import numpy as np
from architecture import ExtSummModel

def load_data(data_paths, data_type="train", limit=10**8):
    doc_path, abstract_path, labels_path = data_paths
    docs = []
    start_ends = []
    abstracts = []
    labels = []

    # actual inputs
    i = 0
    doc_path = os.path.join(doc_path + data_type)
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
        if i > limit:
            break
        i += 1

    # abstracts
    i = 0
    abstract_path = os.path.join(abstract_path + data_type)
    for file_ in os.listdir(abstract_path):
        with open(os.path.join(abstract_path, file_), 'r') as abstract_in:
            for line in abstract_in.read().splitlines():
                abstracts.append(line)  # should only have 1 line
        if i > limit:
            break
        i += 1

    # labels
    i = 0
    labels_path = os.path.join(labels_path + data_type)
    for file_ in os.listdir(labels_path):
        with open(os.path.join(labels_path, file_), 'r') as labels_in:
            labels_json = json.load(labels_in)
            labels.append(labels_json['labels'])
        if i > limit:
            break
        i += 1
    return (docs, start_ends, abstracts, labels)


def train_model():
    # Perform a forward cycle with fictitious data
    model = ExtSummModel(neg_pos_ratio=47.78)
    print("Model initialization completed")
    data_paths = ("arxiv/inputs/", "arxiv/human-abstracts/", "arxiv/labels/")

    # (doc, start_end, abstract, label)
    val_set = load_data(data_paths, data_type="val")
    print("Val set loaded. Length:", len(val_set[0]))

    # train the model
    model.fit(val_set, lr=0.001, epochs=50, batch_size=32)
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
