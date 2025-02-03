import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from process import DataProcessor  
from model import CarPriceNN


file_path = "../data.csv"
processor = DataProcessor(file_path)
processor.load_data()
processor.split_data()
processor.encode_features()
X_train, X_test, y_train, y_test = processor.get_processed_data()

# convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# initialise model
model = CarPriceNN(input_dim=X_train.shape[1])
criterion = nn.MSELoss()  # MSE for regression
optimizer = optim.Adam(model.parameters(), lr=0.005) 

# training loop
epochs = 500
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    
    #backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    #validation loss
    model.eval()
    with torch.no_grad():
        val_preds = model(X_test_tensor)
        val_loss = criterion(val_preds, y_test_tensor)
        val_losses.append(val_loss.item())




# plot losses
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# test
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

# metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R² Score: {r2:.4f}")

# scatterplot for actual vs predicted
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()

torch.save(model.state_dict(), "car_price_model.pth")
print("model saved")