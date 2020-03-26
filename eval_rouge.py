import rouge
from train_model import load_data

# code used from https://pypi.org/project/py-rouge/
def prepare_results(metric, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'\
        .format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def print_rouge(model_path, data):
    # get the hypothesis text
    model = ExtSummModel.load(model_path)
    predictions = model.predict(data)
    docs, start_ends, abstracts, labels = zip(*data)
    hypothesis = []
    for doc in docs:
        temp_hyp = []
        for i, sentence, in enumerate(doc):
            if predictions[i] == 1:
                temp_hyp.append(' '.join(sentence))
        hypothesis.append(' '.join(temp_hyp))

    # set up rouge evaluator
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                            max_n=4,
                            limit_length=True,
                            length_limit=200,
                            length_limit_type='words',
                            apply_avg=True,
                            apply_best=False,
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=True)
    scores = evaluator.get_scores(hypothesis, list(abstracts))
    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        print(prepare_results(metric, results['p'], results['r'], results['f']))

def test_rouge():
    data_paths = ("arxiv/inputs/", "arxiv/human-abstracts/", "arxiv/labels/")
    test_set = load_data(data_paths, data_type="test")
    print_rouge("model/extsumm.bin", test_set)


def main():
    test_rouge()


if __name__ == "__main__":
    main()
