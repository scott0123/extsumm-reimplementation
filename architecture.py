# Import statements
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

# The Extractive Summarization Model, re-implemented
class ExtSummModel(nn.Module):
    def __init__(self, input_size, hidden_size, fc_num, gru_layers=1):
        super().__init__()
        # Used to save model hyperparamers
        self.config = {
            # TODO
            'hidden_size': hidden_size
        }

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

    def forward(self, packed_sent_embedds, doc_lengths): # Xs: batch_size x num_sentences (varies) * sent_dim
        # According to the authors' implementation, sentences' embeddings are directly passed to the model
        # No fine tune on the embeddings
        sent_rep, doc_rep, topic_rep = self.doc_encoder(packed_sent_embedds)
        logits = self.decoder(sent_rep, doc_rep, topic_rep)
        return logits

    def doc_encoder(self, packed_sent_embedds):
        gru_out_packed, hidden = self.bi_gru(packed_sent_embedds)

        pad_gru_output, _ = pad_packed_sequence(gru_out_packed, batch_first=True)
        sent_rep = pad_gru_output
        batch_size, seq_len, _ = pad_gru_output.size()

        # TODO: confirm hidden shape is batch_size x num_layers * num_directions x hidden_size
        doc_rep = hidden.view(hidden.size()[0], self.config['hidden_size'] * 2).expand(-1, seq_len, -1)
        # TODO
        topic_rep = None
        return sent_rep, doc_rep, topic_rep


    def doc_encoder(self, sent_emb):
        # TODO
        pass

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
