# Import statements
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence


# The Extractive Summarization Model, re-implemented
class ExtSummModel(nn.Module):
    def __init__(self, input_size, hidden_size, fc_num, gru_layers=1, glove_dir=None, word_dim=100, vocab=None):
        super().__init__()
        # Used to save model hyperparamers
        self.config = {
            # TODO
            'hidden_size': hidden_size,
            'word_dim': word_dim,
            'trainable_embedding': True,
        }

        # Embedding layer
        weights_matrix = self.create_embeddings(glove_dir, vocab)
        num_embeddings, embedding_dim = weights_matrix.size()
        self.embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_layer.from_pretrained(torch.from_numpy(weights_matrix),
                                             freeze=self.config['trainable_embedding'])

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

    def forward(self, packed_sent_embedds, topic_start_ends):
        # packed_sent_embedds: batch_size x doc_num x num_sentences x sent_dim
        # topic_start_ends: batch_size x num_topics (varies) * 2, sentence indexes should start from 1

        sent_encoded = self.sent_encoder(packed_sent_embedds)   # (batch_size x doc_num x num_sentences x word_dim)

        # TODO: flat the sent_encoded from
        # (batch_size x doc_num x num_sentences x word_dim) -> (batch_size x num_sentences x word_dim)
        sent_rep, doc_rep, topic_rep = self.encoder(sent_encoded, topic_start_ends)
        logits = self.decoder(sent_rep, doc_rep, topic_rep)
        return logits

    def create_embeddings(self, glove_dir, vocab):
        """
        :param glove_dir:
        :param vocab: dict, the entire vocabulary from word to index, from train, test and eval set
        :return:
        """
        embedding_matrix = dict()

        with open(glove_dir, 'rb') as glove_in:
            for line in glove_in:
                line = line.decode().split()
                word = line[0]
                embedding = np.array(line[1:]).astype(np.float)
                embedding_matrix[word] = embedding

        vocab_size = len(vocab)
        weights_matrix = np.zeros((vocab_size, self.config['word_dim']))
        for word, i in vocab.items():
            try:
                weights_matrix[i] = embedding_matrix[word]
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(self.config['word_dim'],))
        return weights_matrix

    def sent_encoder(self, packed_sent_embedds):
        sent_encoded = self.embedding_layer(packed_sent_embedds)    # (batch, doc_dum, sentence_num, word_num, word_dim)
        return torch.mean(sent_encoded, -2)      # (batch, doc_dum, sentence_num, word_dim)

    def doc_encoder(self, packed_sent_embedds, topic_start_ends):
        gru_out_packed, hidden = self.bi_gru(packed_sent_embedds)

        pad_gru_output, _ = pad_packed_sequence(gru_out_packed, batch_first=True)
        sent_rep = pad_gru_output  # batch_size x seq_len x num_directions * hidden_size
        batch_size, seq_len, twice_hidden_size = pad_gru_output.shape

        # TODO: confirm hidden's shape is batch_size x num_layers * num_directions x hidden_size
        doc_rep = hidden.view(batch_size, twice_hidden_size).expand(-1, seq_len, -1)
        topic_rep = np.zeros(pad_gru_output.shape)
        hidden_size = self.config['hidden_size']
        # Pad zeros at the beginning and the end of hidden states
        pad_gru_output = F.pad(pad_gru_output, pad=(0, 0, 1, 1), mode='constant', value=0)
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
                print(f'Epoch {epoch+1}, batch {batch+1}, loss={loss.item():.4f}, acc={accuracy:.4f}')
    

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
