import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from emo_utils import *

torch.manual_seed(1)
np.random.seed(1)


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing each word mapped to its index
    max_len -- maximum number of words in a sentence

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    m = len(X)  # number of training examples
    X_indices = np.zeros((m, max_len), dtype=np.int64)

    for i in range(m):
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            if j < max_len and w in word_to_index:
                X_indices[i, j] = word_to_index[w]
                j += 1
    return X_indices


class PretrainedEmbedding(nn.Module):
    """
    Pretrained embedding layer using GloVe vectors
    """

    def __init__(self, word_to_vec_map, word_to_index):
        super(PretrainedEmbedding, self).__init__()
        vocab_len = len(word_to_index) + 1  # adding 1 for padding/unknown
        emb_dim = word_to_vec_map["cucumber"].shape[0]  # dimensionality of GloVe vectors

        # Initialize embedding matrix
        emb_matrix = np.zeros((vocab_len, emb_dim))
        for word, index in word_to_index.items():
            if word in word_to_vec_map:
                emb_matrix[index, :] = word_to_vec_map[word]

        # Create embedding layer
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(emb_matrix),
            padding_idx=0,
            freeze=True  # Set to False if you want to fine-tune
        )

    def forward(self, x):
        return self.embedding(x)


class SentimentAnalysis(nn.Module):
    """
    Emojify-v2 model in PyTorch
    """

    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim, word_to_vec_map, word_to_index):
        super(SentimentAnalysis, self).__init__()

        # Embedding layer
        self.embedding = PretrainedEmbedding(word_to_vec_map, word_to_index)

        # LSTM layers
        self.lstm1 = nn.LSTM(emb_dim, hidden_dim, batch_first=True, dropout=0.5)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=0.5)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Embedding
        embeddings = self.embedding(x)

        # First LSTM layer - return all sequences for the next LSTM
        lstm1_out, (h1, c1) = self.lstm1(embeddings)
        lstm1_out = self.dropout(lstm1_out)

        # Second LSTM layer - use the output from first LSTM
        lstm2_out, (h2, c2) = self.lstm2(lstm1_out)
        lstm2_out = self.dropout(lstm2_out)

        # Use the last hidden state (last timestep)
        last_output = lstm2_out[:, -1, :]  # Take the last timestep output

        # Fully connected layer
        output = self.fc(last_output)
        output = self.softmax(output)

        return output


def train_model(model, train_loader, val_loader, num_epochs=300, learning_rate=0.001):
    """
    Train the PyTorch model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = 100. * correct / total
        val_accuracies.append(accuracy)

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

    return train_losses, val_accuracies


if __name__ == "__main__":
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Read train and test files
    X_train, Y_train = read_csv('train_emoji.csv')
    X_test, Y_test = read_csv('test_emoji.csv')
    maxLen = len(max(X_train, key=len).split())

    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    print(f"Max sentence length: {maxLen}")

    # Convert labels to PyTorch tensors
    Y_train_tensor = torch.LongTensor(Y_train)
    Y_test_tensor = torch.LongTensor(Y_test)

    # Read GloVe vectors
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

    print(f"Vocabulary size: {len(word_to_index)}")

    # Convert sentences to indices
    X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
    X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)

    print(f"X_train_indices shape: {X_train_indices.shape}")
    print(f"X_test_indices shape: {X_test_indices.shape}")

    # Convert to PyTorch tensors
    X_train_tensor = torch.LongTensor(X_train_indices)
    X_test_tensor = torch.LongTensor(X_test_indices)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model parameters
    vocab_size = len(word_to_index) + 1
    emb_dim = 50
    hidden_dim = 128
    output_dim = 5

    # Initialize model
    model = SentimentAnalysis(vocab_size, emb_dim, hidden_dim, output_dim, word_to_vec_map, word_to_index)
    model.to(device)

    # Print model architecture
    print("\nModel Architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    print("\nStarting training...")
    train_losses, val_accuracies = train_model(model, train_loader, test_loader, num_epochs=300)

    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    all_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            all_predictions.extend(pred.cpu().numpy())

    test_accuracy = 100. * correct / total
    print(f"\nFinal Test accuracy: {test_accuracy:.2f}%")

    # Compare prediction and expected emoji
    print("\nMisclassified examples:")
    model.eval()
    with torch.no_grad():
        # Get all predictions at once
        test_output = model(X_test_tensor)
        test_predictions = test_output.argmax(dim=1).cpu().numpy()

        misclassified_count = 0
        for i in range(len(X_test)):
            if test_predictions[i] != Y_test[i]:
                print(
                    f'Expected emoji: {label_to_emoji(Y_test[i])} prediction: {X_test[i]} {label_to_emoji(test_predictions[i]).strip()}')
                misclassified_count += 1
                if misclassified_count >= 10:  # Limit output to 10 examples
                    print("... and more")
                    break

    # Test your sentence
    print("\nTesting custom sentence:")
    x_test = np.array(['very happy'])
    X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
    X_test_tensor = torch.LongTensor(X_test_indices)

    model.eval()
    with torch.no_grad():
        output = model(X_test_tensor)
        prediction = output.argmax(dim=1).cpu().numpy()[0]

    print(f'{x_test[0]} {label_to_emoji(prediction)}')

    # Test more examples
    test_sentences = [
        'I am so sad',
        'I love you',
        'I hate you',
        'lets play baseball',
        'I want to eat pizza'
    ]

    print("\nTesting more sentences:")
    for sentence in test_sentences:
        x_test = np.array([sentence])
        X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
        X_test_tensor = torch.LongTensor(X_test_indices)

        model.eval()
        with torch.no_grad():
            output = model(X_test_tensor)
            prediction = output.argmax(dim=1).cpu().numpy()[0]

        print(f'{sentence} -> {label_to_emoji(prediction)}')