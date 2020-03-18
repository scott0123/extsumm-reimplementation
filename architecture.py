# Import statements
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sentence, pad_packed_sequence, pack_padded_sequence

# The Extractive Summarization Model, re-implemented
class ExtSummModel(nn.Module):
    def __init__(self, ...): # TODO
        super().__init__()
        # Used to save model hyperparamers
        self.config = {
            # TODO
        }
        # Embedding layer
        self.embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(...), freeze=False)
        # Bidirectional GRU layer
        self.bi_gru = nn.GRU(
            ...,
            bidirectional=True,
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


    def forward(self, Xs):
        sent_emb = self.sent_encoder(Xs)
        sent_rep, doc_rep, topic_rep = self.doc_encoder(sent_emb)
        logits = self.decoder(sent_rep, doc_rep, topic_rep)
        return logits


    def sent_encoder(self, Xs):
        # TODO
        pass


    def doc_encoder(self, sent_emb):
        # TODO
        pass


    def decoder(sent_rep, doc_rep, topic_rep):
        cat_doc_sent = torch.cat((doc_rep, sent_rep), 1)
        cat_topic_sent = torch.cat((topic_rep, sent_rep), 1)
        doc_scores = torch.bmm(self.v_attention, torch.tanh(torch.bmm(self.W_attention, cat_doc_sent)))
        topic_scores = torch.bmm(self.v_attention, torch.tanh(torch.bmm(self.W_attention, cat_doc_topic)))
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
