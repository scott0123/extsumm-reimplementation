# Import statements
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence


# The Extractive Summarization Model, re-implemented
class ExtSummModel(nn.Module):
    def __init__(self, input_size, hidden_size, fc_num, gru_layers=1, glove_dir=None, word_dim=100):
        super().__init__()
        # Used to save model hyperparamers
        self.config = {
            # TODO
            "hidden_size": hidden_size,
            "word_dim": word_dim,
            "trainable_embedding": True,
        }

        # Embedding layer
        self.UNK = "UNK"
        weight_matrix, word2idx = self.create_embeddings(glove_dir)
        # num_embeddings, embedding_dim = weights_matrix.size()
        self.embedding_layer = nn.Embedding(len(weight_matrix), self.config["trainable_embedding"])
        self.embedding_layer.from_pretrained(torch.from_numpy(weight_matrix), freeze=self.config["trainable_embedding"])

        # Bidirectional GRU layer
        self.bi_gru = nn.GRU(
            input_size,
            hidden_size,
            gru_layers,
            batch_first=True,
            bidirectional=True
        )
        # Attention-related parameters
        self.v_attention = ...
        self.W_attention = ...
        # Dense layer 1
        self.dense1 = nn.Linear(
            ...,
        )
        # Dropout layer
        self.dropout = nn.Dropout(...)
        # Dense layer 2
        self.dense2 = nn.Linear(
            ...,
        )
        # Use GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

    def create_embeddings(self, glove_dir):
        """
        :param glove_dir:
        :param vocab: dict, the entire vocabulary from word to index, from train, test and eval set
        :return: embedding_matrix, word2idx
        """
        word2idx = dict()
        idx = 0
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
        embedding_matrix.append(np.random.normal(scale=0.6, size=(self.config["word_dim"],)))
        word2idx[self.UNK] = idx
        return embedding_matrix, word2idx

    def forward(self, documents, topic_start_ends):
        # packed_sent_embedds: batch_size x num_sent x num_word x sent_dim (list)
        # topic_start_ends: batch_size x num_topics (varies) * 2, sentence indexes should start from 1 (list)

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
                word_embedding_tensor = self.embeddings(word_indices_tensor)
                avg_sent_vec = torch.mean(word_embedding_tensor, dim=1)
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

        # TODO: confirm hidden"s shape is batch_size x num_layers * num_directions x hidden_size
        doc_rep = hidden.view(batch_size, twice_hidden_size).expand(-1, seq_len, -1)
        topic_rep = np.zeros(pad_gru_output.shape)
        hidden_size = self.config["hidden_size"]
        # Pad zeros at the beginning and the end of hidden states
        pad_gru_output = F.pad(pad_gru_output, pad=(0, 0, 1, 1), mode="constant", value=0)
        for batch_ind in range(batch_size):
            start_ends = topic_start_ends[batch_ind]  # num_topics * 2
            num_topics, _ = start_ends.shape
            topic_mat = np.zeros((num_topics, twice_hidden_size)) # num_topics x num_directions * hidden_size
            for topic_ind in range(num_topics):
                # forward
                topic_mat[topic_ind, :hidden_size] = pad_gru_output[batch_ind, start_ends[topic_ind+1, 1], :hidden_size] - \
                                             pad_gru_output[batch_ind, start_ends[topic_ind, 1], :hidden_size]
                # backward
                topic_mat[topic_ind, hidden_size:] = pad_gru_output[batch_ind, start_ends[topic_ind, 0], hidden_size:] - \
                                             pad_gru_output[batch_ind, start_ends[topic_ind+1, 0], hidden_size:]
            for topic_ind in range(num_topics):
                for sent_ind in range(start_ends[topic_ind, 0] - 1, start_ends[topic_ind, 1]):
                    topic_rep[batch_ind, sent_ind] = topic_mat[topic_ind]
        topic_rep = torch.from_numpy(topic_rep)  # batch_size x seq_len x num_directions * hidden_size
        return sent_rep, doc_rep, topic_rep

    def decoder(self, sent_rep, doc_rep, topic_rep):
        cat_doc_sent = torch.cat((doc_rep, sent_rep), 1)
        cat_topic_sent = torch.cat((topic_rep, sent_rep), 1)
        doc_scores = torch.bmm(self.v_attention, torch.tanh(torch.bmm(self.W_attention, cat_doc_sent)))
        topic_scores = torch.bmm(self.v_attention, torch.tanh(torch.bmm(self.W_attention, cat_topic_sent)))
        sum_scores = doc_scores + topic_scores
        doc_weights = doc_scores / sum_scores
        topic_weights = topic_scores / sum_scores
        context = torch.mm(doc_weights, doc_rep) + torch.mm(topic_weights, topic_rep)
        input_ = torch.cat((sent_rep, context), 1)
        h = self.dense1(input_)
        h = F.relu(h)
        h = self.dropout(h)
        # final part altered to use logits for computational stability
        logits = self.dense2(h)
        return logits


    def fit(self, Xs, ys, lr, epochs, batch_size=32):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.NLLLoss()
        for epoch in range(epochs):
            Xs_batch_iter = batch_iter(Xs, batch_size=batch_size)
            ys_batch_iter = batch_iter(ys, batch_size=batch_size)
            # Iterate over mini-batches for the current epoch
            for batch, (batch_Xs, batch_ys) in enumerate(zip(Xs_batch_iter, ys_batch_iter)):
                # Clear the gradients of parameters
                optimizer.zero_grad()
                # Perform forward pass to get neural network output
                logits = self.forward(batch_Xs).to(self.device)
                logsigmoid = F.logsigmoid(logits)
                # True labels
                batch_ys_tensor = torch.tensor(batch_ys).to(self.device)
                # Calculate the loss
                loss = loss_fn(log_softmax, batch_ys_tensor)
                accuracy = (predicted == batch_ys_tensor).sum().item() / len(batch_ys_tensor)
                # Call `backward()` on `loss` for back-propagation to compute
                # gradients w.r.t. model parameters
                loss.backward()
                # Perform one step of parameter update using the newly-computed gradients
                optimizer.step()
                print(f"Epoch {epoch+1}, batch {batch+1}, loss={loss.item():.4f}, acc={accuracy:.4f}")
    

    def predict(self, Xs):
        self.eval()
        logits = self.forward(Xs)
        confidence = F.sigmoid(logits)
        # TODO
        return confidence.numpy()


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


def main():
    # Perform a forward cycle with fictitious data
    model = ExtSummModel()
    example_doc = [] # TODO
    model.forward(example_doc)


if __name__ == "__main__":
    main()
