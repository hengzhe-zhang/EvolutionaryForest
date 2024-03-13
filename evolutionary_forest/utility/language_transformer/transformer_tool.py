import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from hebo.models.nn.eac.positional_encoding import PositionalEncoding
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader

from evolutionary_forest.component.translation_transformer import (
    generate_square_subsequent_mask,
)
from evolutionary_forest.utility.language_transformer.convert_tool import (
    generate_and_evaluate_trees,
)


class SequenceDNN(nn.Module):
    def __init__(self, input_dim, output_vocab_size, hidden_dims, num_tokens):
        super(SequenceDNN, self).__init__()
        self.input_dim = input_dim
        self.output_vocab_size = output_vocab_size
        self.hidden_dims = hidden_dims
        self.num_tokens = num_tokens

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

        # Output layer: predicting a sequence, so we need num_tokens output layers
        self.output_layers = nn.ModuleList(
            [nn.Linear(hidden_dims[-1], output_vocab_size) for _ in range(num_tokens)]
        )

    def forward(self, x):
        # Flatten the input in case it's more than 1D
        x = x.view(-1, self.input_dim)

        # Input layer
        x = F.relu(self.input_layer(x))

        # Hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Output layers
        outputs = []
        for layer in self.output_layers:
            # Each output layer predicts one token in the sequence
            outputs.append(layer(x))

        # Stack outputs to have them in the shape [batch_size, num_tokens, output_vocab_size]
        outputs = torch.stack(outputs, dim=1)

        return outputs


# self.dnn = SequenceDNN(
#     input_dim=num_points,  # Each gp_output value is a single dimension
#     output_vocab_size=output_vocab_size,
#     hidden_dims=[4, 4],  # Example hidden layer dimensions
#     num_tokens=num_token,  # Sequence length
# )
class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_vocab_size,
        d_model,
        nhead,
        num_decoder_layers,
        dim_feedforward,
        num_token,
    ):
        super(DecoderOnlyTransformer, self).__init__()
        self.d_model = d_model
        # self.input_embedding = nn.Linear(input_dim, d_model)
        self.input_embedding = nn.Sequential(
            *[nn.Linear(input_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model)]
        )
        self.target_embedding = nn.Embedding(output_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout=0),
            num_decoder_layers,
        )
        self.fc_out = nn.Linear(d_model, output_vocab_size)

    def forward(self, src, tgt, tgt_mask):
        # dnn = self.dnn(src)

        # Embed and encode the source sequence
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        # src = self.pos_encoder(src)

        # Embed and encode the target sequence
        tgt = self.target_embedding(tgt) * math.sqrt(self.d_model)
        tgt = tgt.transpose(0, 1)
        tgt = self.pos_encoder(tgt)

        # Pass the encoded target and source through the decoder
        output = self.transformer_decoder(tgt, src, tgt_mask=tgt_mask)

        # Final linear layer to predict output vocab
        return self.fc_out(output)


if __name__ == "__main__":
    num_trees = 100
    num_points = 3
    training_data = generate_and_evaluate_trees(num_trees, num_points)
    # Define hyperparameters and model
    d_model = 8
    nhead = 2
    num_decoder_layers = 2
    dim_feedforward = 8
    input_dim = num_points  # Each gp_output value is a single dimension

    SOS_TOKEN = "<SOS>"  # Define the SOS token
    END_TOKEN = "<UNK>"  # Define the SOS token
    training_data["vocab"][SOS_TOKEN] = len(
        training_data["vocab"]
    )  # Add SOS token to the vocabulary
    SOS_TOKEN_ID = training_data["vocab"][SOS_TOKEN]  # Get the ID of the SOS token
    END_TOKEN_ID = training_data["vocab"][END_TOKEN]  # Get the ID of the SOS token

    output_vocab_size = len(training_data["vocab"])

    # Convert training data to PyTorch tensors and DataLoader
    gp_outputs_tensor = torch.tensor(training_data["gp_outputs"], dtype=torch.float)
    word_token_sequences_tensor = [
        torch.tensor(seq, dtype=torch.long)
        for seq in training_data["word_token_sequences"]
    ]
    word_token_sequences_padded = pad_sequence(
        word_token_sequences_tensor, batch_first=True
    )

    model = DecoderOnlyTransformer(
        input_dim,
        output_vocab_size,
        d_model,
        nhead,
        num_decoder_layers,
        dim_feedforward,
        word_token_sequences_padded.size(1),
    )

    # model = SequenceDNN(
    #     input_dim=num_points,  # Each gp_output value is a single dimension
    #     output_vocab_size=len(training_data["vocab"]),
    #     hidden_dims=[8, 8],  # Example hidden layer dimensions
    #     num_tokens=word_token_sequences_padded.size(1),  # Sequence length
    # )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = Lion(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

    dataset = TensorDataset(gp_outputs_tensor, word_token_sequences_padded)
    dataloader = DataLoader(dataset, batch_size=num_trees)

    # Training loop
    model.train()
    num_epochs = 30000

    # Choose how many samples you want to display
    num_samples_to_display = 3

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (gp_out, word_tokens) in enumerate(dataloader):
            optimizer.zero_grad()
            if isinstance(model, SequenceDNN):
                # No need to unsqueeze gp_out since the DNN model handles the input dimension
                output = model(gp_out)

                # Compute loss; need to adjust output dimensions for CrossEntropyLoss
                loss = criterion(
                    output.view(-1, output_vocab_size), word_tokens.view(-1)
                )
            else:
                # Generate tgt_mask dynamically based on the current batch's sequence length
                current_batch_size, seq_length = word_tokens.size()
                tgt_mask = generate_square_subsequent_mask(seq_length).to(
                    word_tokens.device
                )  # Use the full sequence length now

                # Prepare the initial decoder input with SOS tokens
                initial_tgt_input = torch.full(
                    (word_tokens.size(0), 1),
                    SOS_TOKEN_ID,
                    dtype=torch.long,
                    device=word_tokens.device,
                )
                tgt_input = torch.cat([initial_tgt_input, word_tokens[:, :-1]], dim=1)

                targets = word_tokens

                gp_out = gp_out.unsqueeze(0)

                output = model(gp_out, tgt_input, tgt_mask=tgt_mask)
                mask = targets.reshape(-1) != END_TOKEN_ID

                loss = criterion(
                    output.view(-1, output_vocab_size)[mask], targets.reshape(-1)[mask]
                )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Print sample predictions for the first batch in each epoch
            if i == 0:
                print(f"\nEpoch {epoch+1}, Batch {i+1}")
                print("Sample Predictions vs Ground Truth")
                output_predictions = F.softmax(output, dim=-1).argmax(dim=-1)
                for j in range(num_samples_to_display):
                    if isinstance(model, SequenceDNN):
                        predicted_sequence = output_predictions[j].tolist()
                    else:
                        predicted_sequence = output_predictions[:, j].tolist()
                    target_sequence = word_tokens[
                        j
                    ].tolist()  # Skip the first token which is used as input

                    # Decode the sequences using your vocabulary
                    predicted_tokens = [
                        list(training_data["vocab"].keys())[
                            list(training_data["vocab"].values()).index(k)
                        ]
                        for k in predicted_sequence
                        if k in training_data["vocab"].values()
                    ]
                    target_tokens = [
                        list(training_data["vocab"].keys())[
                            list(training_data["vocab"].values()).index(k)
                        ]
                        for k in target_sequence
                        if k in training_data["vocab"].values()
                    ]

                    print(f"  Target: {' '.join([str(t) for t in target_tokens])}")
                    print(
                        f"Predicted: {' '.join([str(t) for t in predicted_tokens])}\n"
                    )

        # Update the learning rate
        scheduler.step()

        # (Optional) Log the current learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Current Learning Rate: {current_lr}, "
            f"Loss: {total_loss / len(dataloader)}"
        )
