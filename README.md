# Stock-Price-Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Predict future stock prices using an RNN model based on historical closing prices from trainset.csv and testset.csv, with data normalized using MinMaxScaler.

## Design Steps
 ## Step 1:
 Import necessary libraries.

 ## Step 2:
Load and preprocess the data.

## Step 3:
Create input-output sequences.

## Step 4:
Convert data to PyTorch tensors.

## Step 5:
Define the RNN model.

## Step 6:
Train the model using the training data.

## Step 7:
Evaluate the model and plot predictions.



## Program
#### Name: PRAJAN P
#### Register Number: 212223240121
```Python
class RNNModel(nn.Module):
  def __init__(self, input_size = 1, hidden_size = 64, output_size = 1, num_layers = 2):
    super(RNNModel, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self , x):
    out, _ = self.rnn(x)
    out = self.fc(out[:, -1, :])
    return out

# Train the Model
criterion =nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
model.train()
train_losses = []
for epoch in range(epochs):
  epoch_loss = 0
  for x_batch , y_batch in train_loader:
    x_batch , y_batch = x_batch.to(device) , y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  train_losses.append(epoch_loss/len(train_loader))
  print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}")

```

## Output
### DATA SET
<img width="681" height="771" alt="image" src="https://github.com/user-attachments/assets/a9e9363b-63e1-45df-a171-9eff5bfca1f4" />

### Training loss over Epochs:
<img width="754" height="627" alt="image" src="https://github.com/user-attachments/assets/5814431e-9c34-49fd-b2c6-7f3b850a996e" />

### True Stock Price, Predicted Stock Price vs time

<img width="1085" height="728" alt="image" src="https://github.com/user-attachments/assets/1c82f048-2696-4bc3-b205-fed538f700c8" />


### Predictions 
<img width="335" height="81" alt="image" src="https://github.com/user-attachments/assets/09fbb4cd-a445-46d7-92c6-91b32f20033c" />

## Result


