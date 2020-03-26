# Import statements
import os
import json
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import rouge


# The Extractive Summarization Model, re-implemented
class ExtSummModel(nn.Module):
    def __init__(self, embedding_size=300, gru_units=128, gru_layers=1, dense_units=128,
                 dropout=0.3, glove_dir="embeddings", freeze_embedding=True):
        super().__init__()
        weight_matrix, word2idx = self.create_embeddings(f"{glove_dir}/glove.6B.{embedding_size}d.txt", embedding_size)
        # Used to save model hyperparamers
        self.config = {
            "embedding_size": embedding_size,
            "gru_units": gru_units,
            "gru_layers": gru_layers,
            "glove_dir": glove_dir,
            "freeze_embedding": freeze_embedding,
            "word2idx": word2idx,
        }
        # Embedding layer
        self.embedding_layer = nn.Embedding.from_pretrained(
            torch.from_numpy(weight_matrix),
            freeze=freeze_embedding,
        )
        # Bidirectional GRU layer
        self.bi_gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=gru_units,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Attention-related parameters (*4 because we use concatenated representations, each being 2)
        self.v_attention = nn.Parameter(torch.randn(gru_units * 4, 1))
        self.W_attention = nn.Parameter(torch.randn(gru_units * 4, gru_units * 4))
        # Dense layer 1
        self.dense1 = nn.Linear(
            in_features=gru_units * 4,
            out_features=dense_units,
        )
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Dense layer 2
        self.dense2 = nn.Linear(
            in_features=dense_units,
            out_features=1,
        )
        # Use GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

    def create_embeddings(self, glove_dir, embedding_size):
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

    def forward(self, documents, topic_start_ends):
        # documents: batch_size x num_sent x num_word x sent_dim (list of list of list)
        # topic_start_ends: batch_size x [num_topics x 2], sentence indexes should start from 1 (list of 2D np arrays)
        sent_encoded = self.sent_encoder(documents)   # (batch_size x num_sent x num_word x word_dim)
        # (batch_size x num_sent x num_word x word_dim) -> (batch_size x num_word x word_dim)
        sent_rep, doc_rep, topic_rep = self.doc_encoder(sent_encoded, topic_start_ends)
        logits = self.decoder(sent_rep, doc_rep, topic_rep)
        return logits

    def sent_encoder(self, documents):
        """
        documents: list (batch) of list (doc) of list (sent) word embeddings indices
        """
        # number of sentences in a doc
        actual_lengths = []
        embeddings = []
        for doc in documents:
            # all average sentence vectors in the document
            avg_sent_vecs = []
            for sent in doc:
                word_indices_tensor = torch.LongTensor(sent)
                word_embedding_tensor = self.embedding_layer(word_indices_tensor).float()
                avg_sent_vec = torch.mean(word_embedding_tensor, dim=0)
                avg_sent_vecs.append(avg_sent_vec)
            embeddings.append(torch.stack(avg_sent_vecs))
            actual_lengths.append(len(doc))

        padded_embeddings = pad_sequence(
            embeddings,
            batch_first=True,
        )
        packed_sent_embeddings = pack_padded_sequence(
            padded_embeddings,
            actual_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        return packed_sent_embeddings

    def doc_encoder(self, packed_sent_embeddings, topic_start_ends):
        gru_out_packed, hidden = self.bi_gru(packed_sent_embeddings)

        pad_gru_output, _ = pad_packed_sequence(gru_out_packed, batch_first=True)
        sent_rep = pad_gru_output  # batch_size x seq_len x num_directions * hidden_size
        batch_size, seq_len, twice_hidden_size = pad_gru_output.shape

        doc_rep = hidden.view(batch_size, twice_hidden_size).expand(seq_len, -1, -1).transpose(1, 0)
        topic_rep = np.zeros(pad_gru_output.shape)
        hidden_size = self.config["gru_units"]
        # Pad zeros at the beginning and the end of hidden states
        pad_gru_output = F.pad(pad_gru_output, pad=(0, 0, 1, 1), mode="constant", value=0).detach().numpy()
        for batch_idx in range(batch_size):
            starts = topic_start_ends[batch_idx][:, 0]
            ends = topic_start_ends[batch_idx][:, 1]
            num_topics = len(starts)
            topic_mat = np.zeros((num_topics, twice_hidden_size))  # num_topics x num_directions * hidden_size
            for topic_idx in range(num_topics):
                # forward
                topic_mat[topic_idx, :hidden_size] = pad_gru_output[batch_idx, ends[topic_idx], :hidden_size] - \
                                                     pad_gru_output[batch_idx, starts[topic_idx] - 1, :hidden_size]
                # backward
                topic_mat[topic_idx, hidden_size:] = pad_gru_output[batch_idx, starts[topic_idx], hidden_size:] - \
                                                     pad_gru_output[batch_idx, ends[topic_idx] + 1, hidden_size:]
            for topic_idx in range(num_topics):
                for sent_idx in range(starts[topic_idx] - 1, ends[topic_idx]):
                    topic_rep[batch_idx, sent_idx] = topic_mat[topic_idx]
        topic_rep = torch.from_numpy(topic_rep).float().to(
            self.device)  # batch_size x seq_len x num_directions * hidden_size
        return sent_rep, doc_rep, topic_rep

    def decoder(self, sent_rep, doc_rep, topic_rep):
        # calculating (d:sr) and (l:sr)
        cat_doc_sent = torch.cat((doc_rep, sent_rep), 2)
        cat_topic_sent = torch.cat((topic_rep, sent_rep), 2)
        # calculating Wa(d:sr) and Wa(l:sr)
        W_ds_mult = torch.matmul(cat_doc_sent, self.W_attention)
        W_ts_mult = torch.matmul(cat_topic_sent, self.W_attention)
        # calculating score = v * tanh(...)
        doc_scores = torch.matmul(torch.tanh(W_ds_mult), self.v_attention)
        topic_scores = torch.matmul(torch.tanh(W_ts_mult), self.v_attention)
        # calculating weight = score^d / (score^d + score^l)
        # calculating weight = score^l / (score^d + score^l)
        sum_scores = doc_scores + topic_scores # TODO: paper different from implementation
        doc_weights = doc_scores / sum_scores # TODO: paper different from implementation
        topic_weights = topic_scores / sum_scores # TODO: paper different from implementation
        # calculating context = weight^d * d + weight^l * l
        context = doc_weights * doc_rep + topic_weights * topic_rep
        # calculating input = (sr:context)
        input_ = torch.cat((sent_rep, context), 2)
        h = self.dense1(input_)
        h = F.relu(h)
        h = self.dropout(h)
        # final part altered to use logits for computational stability
        logits = self.dense2(h).squeeze(2)
        return logits

    def convert_word_to_idx(self, data):
        word2idx = self.config["word2idx"]
        for doc, start_end, abstract, label in data:
            for i, sentence in enumerate(doc):
                # convert all the words to its corresponding indices. If UNK, assign to the last entry
                doc[i] = [word2idx[word] if word in word2idx else len(word2idx) - 1
                          for word in sentence]

    def fit(self, Xs, lr, epochs, batch_size=32):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.convert_word_to_idx(Xs)        # convert word to its index
        for epoch in range(epochs):
            Xs_batch_iter = batch_iter(Xs, batch_size=batch_size)
            # ys_batch_iter = batch_iter(ys, batch_size=batch_size)

            # Iterate over mini-batches for the current epoch
            for batch, batch_Xs in enumerate(Xs_batch_iter):
                docs, start_ends, abstracts, labels = zip(*batch_Xs)
                # Clear the gradients of parameters
                optimizer.zero_grad()
                # Perform forward pass to get neural network output
                logits = self.forward(docs, start_ends).to(self.device)
                # True labels
                batch_ys = []
                for label in labels:
                    batch_ys.append(torch.FloatTensor(label))
                batch_ys_tensor = pad_sequence(batch_ys, padding_value=-1).permute(1, 0).to(self.device)
                label_mask = batch_ys_tensor.gt(-1).float()
                # Calculate the loss
                loss = F.binary_cross_entropy_with_logits(logits, batch_ys_tensor, weight=label_mask)
                probas = torch.sigmoid(logits)
                predicted = (probas > 0.5).float()
                accuracy = (predicted == batch_ys_tensor).sum().item() / batch_ys_tensor.numel()
                # Call `backward()` on `loss` for back-propagation to compute
                # gradients w.r.t. model parameters
                loss.backward()
                # Perform one step of parameter update using the newly-computed gradients
                optimizer.step()
                print(f"Epoch {epoch+1}, batch {batch+1}, loss={loss.item():.4f}, acc={accuracy:.4f}")

    def predict(self, Xs):
        self.eval()
        self.convert_word_to_idx(Xs)
        docs, start_ends, abstracts, labels = zip(*Xs)
        logits = self.forward(docs, start_ends)
        confidence = F.sigmoid(logits)
        return (confidence.numpy() > 0.5).float()

    def save(self, model_path):
        model_state = {
            "state_dict": self.state_dict(),
            "config":  self.config
        }
        torch.save(model_state, model_path)

    @classmethod
    def load(cls, model_path):
        model_state = torch.load(str(model_path), map_location=lambda storage, loc: storage)
        args = model_state["config"]
        model = cls(**args)
        model.load_state_dict(model_state["state_dict"])
        # Use GPU if available
        if torch.cuda.is_available():
            model.device = torch.device("cuda")
        else:
            model.device = torch.device("cpu")
        model.to(model.device)
        return model


# taken from 01-intro-ner-pytorch/ner.py
def batch_iter(data, batch_size=32):
    batch_num = int(np.ceil(len(data) / batch_size))
    index_array = list(range(len(data)))
    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        batch_examples = [data[idx] for idx in indices]
        yield batch_examples


def load_data(data_paths, data_type="train"):
    doc_path, abstract_path, labels_path = data_paths
    docs = []
    start_ends = []
    abstracts = []
    labels = []

    # actual inputs
    doc_path = os.path.join(doc_path + data_type)
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
            if len(docs) == 100:
                break

    # abstracts
    abstract_path = os.path.join(abstract_path + data_type)
    for file in os.listdir(abstract_path):
        with open(os.path.join(abstract_path, file), 'r') as abstract_in:
            for line in abstract_in.read().splitlines():
                abstracts.append(line)  # should only have 1 line
            if len(abstracts) == 100:
                break

    # labels
    labels_path = os.path.join(labels_path + data_type)
    for file in os.listdir(labels_path):
        with open(os.path.join(labels_path, file), 'r') as labels_in:
            labels_json = json.load(labels_in)
            labels.append(labels_json['labels'])
            if len(labels) == 100:
                break
    return [(doc, start_end, abstract, label) for (doc, start_end, abstract, label)
            in zip(docs, start_ends, abstracts, labels)]


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


def test_forward():
    model = ExtSummModel()
    example_batch = [[[1, 2], [0, 7, 3], [5, 6, 7]], [[5, 3], [6, 7], [2], [3, 4, 5, 6]]]  # TODO
    example_starts_ends = [np.array([[1, 2], [3, 3]]), np.array([[1, 1], [2, 3], [4, 4]])]
    model.forward(example_batch, example_starts_ends)
    print("Forward function complete")


def test_with_data():
    # Perform a forward cycle with fictitious data
    model = ExtSummModel()
    data_paths = ("arxiv/inputs/", "arxiv/human-abstracts/", "arxiv/labels/")

    # [(doc, start_end, abstract, label)]
    train_set = load_data(data_paths, data_type="train")
    test_set = load_data(data_paths, data_type="test")
    val_set = load_data(data_paths, data_type="val")

    model.fit(train_set, 0.001, 50)

    # TODO: Shuffle the data


def test_rouge():
    data_paths = ("arxiv/inputs/", "arxiv/human-abstracts/", "arxiv/labels/")
    train_set = load_data(data_paths, data_type="train")
    print_rouge("", train_set)


def main():
    # test_with_data()
    test_rouge()


if __name__ == "__main__":
    main()
