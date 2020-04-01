import os
import sys
import json
import numpy as np
from train_model import load_data, create_embeddings
from architecture import ExtSummModel, batch_generator
import rouge
import nltk
nltk.download("punkt")


# code used from https://pypi.org/project/py-rouge/
def prepare_results(metric, p, r, f):
    return "\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}"\
        .format(metric, "P", 100.0 * p, "R", 100.0 * r, "F1", 100.0 * f)


def print_rouge(model_path, data, docs):
    # get the hypothesis text
    len_limit = 200
    model = ExtSummModel.load(model_path)
    batch_Xs_generator = batch_generator(*data, batch_size=32, shuffle=False)
    predictions = []
    for batch, batch_Xs in enumerate(batch_Xs_generator):
        prediction = model.predict(batch_Xs)
        predictions.extend(prediction)

    # get sentences
    hypothesis = []
    for i, doc in enumerate(docs):
        temp_count = 0
        temp_abs = []
        pred = np.argsort(predictions[i][:len(doc)])[::-1]
        selected_idx = []
        for idx in pred:
            temp_count += len(doc[idx])
            selected_idx.append(idx)
            if temp_count > len_limit:
                break
        for idx in sorted(selected_idx):
            temp_abs.append(" ".join(doc[idx]))
        hypothesis.append(" ".join(temp_abs))

    evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l", "rouge-w"],
                            max_n=4,
                            limit_length=True,
                            length_limit=200,
                            length_limit_type="words",
                            apply_avg=True,
                            apply_best=False,
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=True)
    scores = evaluator.get_scores(hypothesis, list(data[2]))
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        print(prepare_results(metric, results["p"], results["r"], results["f"]))


def load_test_docs(data_paths, data_type="test"):
    doc_path, abstract_path, labels_path = data_paths
    docs = []
    doc_path = os.path.join(doc_path, data_type)
    doc_files = []
    for file_ in os.listdir(doc_path):
        doc_files.append(file_)
    for file_ in sorted(doc_files):
        with open(os.path.join(doc_path, file_), "r") as doc_in:
            doc_json = json.load(doc_in)
            one_doc = []
            for sentence in doc_json["inputs"]:
                one_doc.append(sentence["tokens"])
            docs.append(one_doc)
    return docs


def test_rouge():
    data_paths = ("arxiv/inputs/", "arxiv/human-abstracts/", "arxiv/labels/")
    glove_dir = "embeddings"
    embedding_size = 300
    model_path = sys.argv[1]
    weight_matrix, word2idx = create_embeddings(f"{glove_dir}/glove.6B.{embedding_size}d.txt")
    test_set = load_data(word2idx, data_paths, data_type="test")
    test_docs = load_test_docs(data_paths, data_type="test")
    print_rouge(model_path, test_set, test_docs)


def main():
    test_rouge()


if __name__ == "__main__":
    main()
