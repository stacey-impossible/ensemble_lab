from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import sklearn
from sklearn.ensemble import BaggingClassifier

# load dataset
x, y = load_digits(n_class=10, return_X_y=True)

print(x.shape)
print(np.unique(y))

index = 0

print(y[index])
plt.imshow(x[index].reshape(8, 8), cmap="Greys")

print(x[index])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    shuffle=True, random_state=42)
print(x_train.shape)

print(x_test.shape)

scaler = StandardScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()

    # слои свертки
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1, padding=2)
    self.conv1_s = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=2, padding=2)
    self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
    self.conv2_s = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=2, padding=1)
    self.conv3 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=3, stride=1, padding=1)
    self.conv3_s = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=2, padding=1)

    self.flatten = nn.Flatten()
    # полносвязный слой
    self.fc1 = nn.Linear(10, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv1_s(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv2_s(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv3_s(x))

    x = self.flatten(x)
    x = self.fc1(x)
    x = F.softmax(x)

    return x

batch_size = 64
learning_rate = 1e-3
epochs = 200

x_train_tensor = torch.tensor(x_train_scaled.reshape(-1, 1, 8, 8).astype(np.float32))
x_test_tensor = torch.tensor(x_test_scaled.reshape(-1, 1, 8, 8).astype(np.float32))

y_train_tensor = torch.tensor(y_train.astype(np.compat.long))
y_test_tensor = torch.tensor(y_test.astype(np.compat.long))

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

simple_cnn = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(simple_cnn.parameters(), lr=learning_rate)

for epoch in range(epochs):
  simple_cnn.train()
  train_samples_count = 0
  true_train_samples_count = 0
  running_loss = 0

  for batch in train_loader:
    x_data = batch[0]
    y_data = batch[1]

    y_pred = simple_cnn(x_data)
    loss = criterion(y_pred, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # обратное распространение ошибки

    running_loss += loss.item()

    y_pred = y_pred.argmax(dim=1, keepdim=False)
    true_classified = (y_pred == y_data).sum().item() # количество верно предсказанных картинок в текущем батче
    true_train_samples_count += true_classified
    train_samples_count += len(x_data) # размер текущего батча

  train_accuracy = true_train_samples_count / train_samples_count
  print(f"[{epoch}] train loss: {running_loss}, accuracy: {round(train_accuracy, 4)}")

  # тестовая выборка
  simple_cnn.eval()
  test_samples_count = 0
  true_test_samples_count = 0
  running_loss = 0

  for batch in test_loader:
    x_data = batch[0]
    y_data = batch[1]

    y_pred = simple_cnn(x_data)
    loss = criterion(y_pred, y_data)

    loss.backward()

    running_loss += loss.item()

    y_pred = y_pred.argmax(dim=1, keepdim=False)
    true_classified = (y_pred == y_data).sum().item()
    true_test_samples_count += true_classified
    test_samples_count += len(x_data)

  test_accuracy = true_test_samples_count / test_samples_count
  print(f"[{epoch}] test loss: {running_loss}, accuracy: {round(test_accuracy, 4)}")

class PytorchModel(sklearn.base.BaseEstimator):
  def __init__(self, net_type, net_params, optim_type, optim_params, loss_fn,
               input_shape, batch_size=32, accuracy_tol=0.02, tol_epochs=10):
    self.net_type = net_type
    self.net_params = net_params
    self.optim_type = optim_type
    self.optim_params = optim_params
    self.loss_fn = loss_fn

    self.input_shape = input_shape
    self.batch_size = batch_size
    self.accuracy_tol = accuracy_tol
    self.tol_epochs = tol_epochs
    
  def fit(self, X, y):
    self.net = self.net_type(**self.net_params)
    self.optim = self.optim_type(self.net.parameters(), **self.optim_params)

    uniq_classes = np.sort(np.unique(y))
    self.classes_ = uniq_classes

    X = X.reshape(-1, *self.input_shape)
    x_tensor = torch.tensor(X.astype(np.float32))
    y_tensor = torch.tensor(y.astype(np.long))
    train_dataset = TensorDataset(x_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                              shuffle=True, drop_last=False)
    last_accuracies = []
    epoch = 0
    keep_training = True
    while keep_training:
      self.net.train()
      train_samples_count = 0
      true_train_samples_count = 0

      for batch in train_loader:
        x_data, y_data = batch[0], batch[1]

        y_pred = self.net(x_data)
        loss = self.loss_fn(y_pred, y_data)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        y_pred = y_pred.argmax(dim=1, keepdim=False)
        true_classified = (y_pred == y_data).sum().item()
        true_train_samples_count += true_classified
        train_samples_count += len(x_data)

      train_accuracy = true_train_samples_count / train_samples_count
      last_accuracies.append(train_accuracy)

      if len(last_accuracies) > self.tol_epochs:
        last_accuracies.pop(0)

      if len(last_accuracies) == self.tol_epochs:
        accuracy_difference = max(last_accuracies) - min(last_accuracies)
        if accuracy_difference <= self.accuracy_tol:
          keep_training = False
  
  def predict_proba(self, X, y=None):
    X = X.reshape(-1, *self.input_shape)
    x_tensor = torch.tensor(X.astype(np.float32))
    if y:
      y_tensor = torch.tensor(y.astype(np.long))
    else:
      y_tensor = torch.zeros(len(X), dtype=torch.long)
    test_dataset = TensorDataset(x_tensor, y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                              shuffle=False, drop_last=False)

    self.net.eval()
    predictions = []
    for batch in test_loader:
      x_data, y_data = batch[0], batch[1]

      y_pred = self.net(x_data)

      predictions.append(y_pred.detach().cpu().numpy())

    predictions = np.concatenate(predictions)
    return predictions

  def predict(self, X, y=None):
    predictions = self.predict_proba(X, y)
    predictions = predictions.argmax(axis=1)
    return predictions

base_model = PytorchModel(net_type=SimpleCNN, net_params=dict(), optim_type=Adam,
                          optim_params={"lr": 1e-3}, loss_fn=nn.CrossEntropyLoss(),
                          input_shape=(1, 8, 8), batch_size=32, accuracy_tol=0.02,
                          tol_epochs=10)

base_model.fit(x_train_scaled, y_train)

preds = base_model.predict(x_test_scaled)
true_classified = (preds == y_test).sum()
test_accuracy = true_classified / len(y_test)

print(f"Test accuracy: {test_accuracy}")

meta_classifier = BaggingClassifier(base_estimator=base_model, n_estimators=10)

meta_classifier.fit(x_train_scaled.reshape(-1, 64), y_train)

print(meta_classifier.score(x_test_scaled.reshape(-1, 64), y_test))
