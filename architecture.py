# Import statements
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence


# The Extractive Summarization Model, re-implemented
class ExtSummModel(nn.Module):
    def __init__(self, weight_matrix=None, num_embeddings=400000, embedding_size=300, freeze_embedding=True,
                 gru_units=128, gru_layers=1, dense_units=128, dropout=0.3, neg_pos_ratio=47):
        super().__init__()
        if weight_matrix is not None:
            num_embeddings = weight_matrix.shape[0]
        # Used to save model hyperparamers
        self.config = {
            "num_embeddings": num_embeddings,
            "embedding_size": embedding_size,
            "freeze_embedding": freeze_embedding,
            "gru_units": gru_units,
            "gru_layers": gru_layers,
            "dense_units": dense_units,
            "dropout": dropout,
            "neg_pos_ratio": neg_pos_ratio,
        }
        # Embedding layer
        if weight_matrix is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(
                torch.from_numpy(weight_matrix),
                freeze=freeze_embedding,
            )
        else:
            self.embedding_layer = nn.Embedding(num_embeddings, embedding_size)
        # Bidirectional GRU layer
        self.bi_gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=gru_units,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Dense layer (baseline)
        self.dense_base = nn.Linear(
            in_features=gru_units * 2,
            out_features=dense_units,
        )
        # Dense layer (concat)
        self.dense_concat = nn.Linear(
            in_features=gru_units * 6,
            out_features=dense_units,
        )
        # Attention-related parameters (*4 because we use concatenated representations, each being 2)
        self.v_attention = nn.Parameter(torch.randn(gru_units * 4, 1))
        self.W_attention = nn.Parameter(torch.randn(gru_units * 4, gru_units * 4))
        # Dense layer (attention)
        self.dense_attn = nn.Linear(
            in_features=gru_units * 4,
            out_features=dense_units,
        )
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Final dense layer
        self.dense_out = nn.Linear(
            in_features=dense_units,
            out_features=1,
        )
        # Use GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, documents, topic_start_ends):
        # documents: batch_size x num_sent x num_word x sent_dim (list of list of list)
        # topic_start_ends: batch_size x [num_topics x 2], sentence indexes should start from 1 (list of 2D np arrays)
        sent_encoded = self.sent_encoder(documents)   # (batch_size x num_sent x num_word x word_dim)
        # (batch_size x num_sent x num_word x word_dim) -> (batch_size x num_word x word_dim)
        sent_rep, doc_rep, topic_rep = self.doc_encoder(sent_encoded, topic_start_ends)
        logits = self.decoder_attention(sent_rep, doc_rep, topic_rep)
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
                word_indices_tensor = torch.LongTensor(sent).to(self.device)
                word_embedding_tensor = self.embedding_layer(word_indices_tensor)
                avg_sent_vec = torch.mean(word_embedding_tensor, dim=0)
                avg_sent_vecs.append(avg_sent_vec)
            embeddings.append(torch.stack(avg_sent_vecs).to(self.device))
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
        topic_rep = torch.zeros(pad_gru_output.shape).to(self.device)
        hidden_size = self.config["gru_units"]
        # Pad zeros at the beginning and the end of hidden states
        pad_gru_output = F.pad(pad_gru_output, pad=(0, 0, 1, 1), mode="constant", value=0)
        for batch_idx in range(batch_size):
            starts = topic_start_ends[batch_idx][:, 0]
            ends = topic_start_ends[batch_idx][:, 1]
            num_topics = len(starts)
            topic_mat = torch.zeros((num_topics, twice_hidden_size)).to(self.device)  # num_topics x num_directions * hidden_size
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
        # batch_size x seq_len x num_directions * hidden_size
        return sent_rep, doc_rep, topic_rep

    def doc_encoder_base(self, packed_sent_embeddings, topic_start_ends):
        gru_out_packed, hidden = self.bi_gru(packed_sent_embeddings)

        pad_gru_output, _ = pad_packed_sequence(gru_out_packed, batch_first=True)
        sent_rep = pad_gru_output  # batch_size x seq_len x num_directions * hidden_size
        return sent_rep, None, None

    def decoder_attention(self, sent_rep, doc_rep, topic_rep):
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
        #sum_scores = doc_scores + topic_scores # TODO: paper different from implementation
        #doc_weights = doc_scores / sum_scores # TODO: paper different from implementation
        #topic_weights = topic_scores / sum_scores # TODO: paper different from implementation
        doc_weights = F.softmax(doc_scores, dim=1)
        topic_weights = F.softmax(topic_scores, dim=1)
        # calculating context = weight^d * d + weight^l * l
        context = doc_weights * doc_rep + topic_weights * topic_rep
        # calculating input = (sr:context)
        input_ = torch.cat((sent_rep, context), 2)
        h = self.dense_attn(input_)
        h = F.relu(h)
        h = self.dropout(h)
        # final part altered to use logits for computational stability
        logits = self.dense_out(h).squeeze(2)
        return logits

    def decoder_concat(self, sent_rep, doc_rep, topic_rep):
        cat = torch.cat((doc_rep, topic_rep, sent_rep), 2)
        h = self.dense_concat(cat)
        h = F.relu(h)
        h = self.dropout(h)
        # final part altered to use logits for computational stability
        logits = self.dense_out(h).squeeze(2)
        return logits

    def decoder_base(self, sent_rep, *_):
        h = self.dense_base(sent_rep)
        h = F.relu(h)
        h = self.dropout(h)
        # final part altered to use logits for computational stability
        logits = self.dense_out(h).squeeze(2)
        return logits

    def fit(self, Xs, lr, epochs, batch_size=32):
        self.train()
        neg_pos_ratio = torch.FloatTensor([self.config["neg_pos_ratio"]]).to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        for epoch in range(epochs):
            batch_Xs_generator = batch_generator(*Xs, batch_size=batch_size, shuffle=True)
            # Iterate over mini-batches for the current epoch
            for batch, batch_Xs in enumerate(batch_Xs_generator):
                docs, start_ends, abstracts, labels = batch_Xs
                # Clear the gradients of parameters
                optimizer.zero_grad()
                # Perform forward pass to get neural network output
                logits = self.forward(docs, start_ends).to(self.device)
                # True labels
                batch_ys = []
                for label in labels:
                    batch_ys.append(torch.FloatTensor(label).to(self.device))
                batch_ys_tensor = pad_sequence(batch_ys, padding_value=-1).permute(1, 0).to(self.device)
                label_mask = batch_ys_tensor.gt(-1).float()
                # Calculate the loss
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    batch_ys_tensor,
                    weight=label_mask,
                    pos_weight=neg_pos_ratio,
                )
                accuracy = self.calculate_accuracy(batch_ys_tensor, labels, logits)
                # Call `backward()` on `loss` for back-propagation to compute
                # gradients w.r.t. model parameters
                loss.backward()
                # Perform one step of parameter update using the newly-computed gradients
                optimizer.step()
                print(f"Epoch {epoch+1}, batch {batch+1}, loss={loss.item():.4f}, acc={accuracy:.4f}")

    @staticmethod
    def calculate_accuracy(batch_ys_tensor, labels, logits):
        probas = torch.sigmoid(logits)
        predicted = (probas > 0.5).float()
        num_to_predict = sum([len(seq) for seq in labels])
        accuracy = torch.sum(predicted == batch_ys_tensor).item() / num_to_predict
        return accuracy

    def predict(self, Xs):
        self.eval()
        docs, start_ends, abstracts, labels = Xs
        logits = self.forward(docs, start_ends)
        confidence = torch.sigmoid(logits)
        return confidence.detach().cpu().numpy()

    def predict_and_eval(self, Xs):
        self.eval()
        docs, start_ends, abstracts, labels = Xs
        logits = self.forward(docs, start_ends).to(self.device)
        ys = []
        for label in labels:
            ys.append(torch.FloatTensor(label).to(self.device))
        ys_tensor = pad_sequence(ys, padding_value=-1).permute(1, 0).to(self.device)
        label_mask = ys_tensor.gt(-1).float()
        accuracy = self.calculate_accuracy(ys_tensor, labels, logits)
        return accuracy

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


def batch_generator(*data, batch_size=32, shuffle=True):
    batch_num = int(np.ceil(len(data[0]) / batch_size))
    index_array = list(range(len(data[0])))
    if shuffle:
        np.random.shuffle(index_array)
    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        batch_examples = tuple([item[idx] for idx in indices] for item in data)
        yield batch_examples


def test_forward():
    model = ExtSummModel()
    example_batch = [[[1, 2], [0, 7, 3], [5, 6, 7]], [[5, 3], [6, 7], [2], [3, 4, 5, 6]]]
    example_starts_ends = [np.array([[1, 2], [3, 3]]), np.array([[1, 1], [2, 3], [4, 4]])]
    model.forward(example_batch, example_starts_ends)
    print("Forward function complete")


def main():
    test_forward()

if __name__ == "__main__":
    main()
