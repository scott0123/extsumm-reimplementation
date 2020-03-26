import os
from train_model import load_data
from architecture import ExtSummModel

def test_model():
    # Perform a forward cycle with fictitious data
    model = ExtSummModel.load("model/extsumm.bin")
    data_paths = ("arxiv/inputs/", "arxiv/human-abstracts/", "arxiv/labels/")

    # (doc, start_end, abstract, label)
    test_set = load_data(data_paths, data_type="test")

    # find the test accuracy of the model
    accuracy = model.predict_and_eval(test_set)
    print(f"Testing accuracy: {accuracy}")


def main():
    test_model()


if __name__ == "__main__":
    main()
