# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# import sklearn
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import random
import torch
import torch.nn.init as init
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split



# Data Directories
train_dir = 'C:/Users/aksha/Desktop/NSLR/training'
val_dir = 'C:/Users/aksha/Desktop/NSLR/testing'
test_dir = 'C:/Users/aksha/Desktop/NSLR/validation'


# Generate Training Data
def generateTrainingData(cropX, cropY, data):

    X = []
    Y = []

    for image_name in os.listdir(data):
        # Read the image
        img = cv2.imread(os.path.join(data, image_name))

        h, w, _ = img.shape
        img_copy = img.copy()

        # Randomly select a crop
        if w - cropX > 0:
            x1 = np.random.randint(0, w - cropX)
        else:
            x1 = 0

        if h - cropY > 0:
            y1 = np.random.randint(0, h - cropY)
        else:
            y1 = 0

        # Crop the image
        x2 = x1 + cropX
        y2 = y1 + cropY

        cropped_img = img_copy[y1:y2, x1:x2, :]

        # Resize the image
        resized_img = cv2.resize(cropped_img, (300, 300))

        # Copy the image, and replace the center with white
        x_img = resized_img.copy()
        y_img = x_img[20:40, 20:40, :].copy()
        x_img[20:40, 20:40, :] = 255

        # Convert to grayscale
        #x_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        #y_img = cv2.cvtColor(y_img, cv2.COLOR_BGR2GRAY)

        # Append the image to the list
        X.append(x_img)
        Y.append(y_img)

    return X, Y

def resizeImages(X, Y, size):
    X = [cv2.resize(x, (size, size)) for x in X]
    Y = [cv2.resize(y, (size, size)) for y in Y]

    return X, Y



#X, Y = resizeImages(X, Y, size)

X, Y = generateTrainingData(150, 150, train_dir)

# Generate Testing Data
def generateTestingData():

    X = []

    for image_name in os.listdir(test_dir):
      img = cv2.imread(os.path.join(test_dir, image_name))
      resized_img = cv2.resize(img, (60, 60))

      x_img = resized_img.copy()
      X.append(x_img)
  
    X = [torch.from_numpy(x) for x in X]
    X = torch.stack(X, dim=0)
    X = X.cuda()
  
    return X


r = random.randint(0, 577)

plt.imshow(X[r])
plt.show()

print(X[r].shape)

plt.imshow(Y[r])
plt.show()

print(Y[r].shape)

Yr = cv2.resize(Y[r], (150, 150))

plt.imshow(Yr)
plt.show()

print(Yr.shape)

# Create a decision tree classifier
# Create and train the model

model = DecisionTreeRegressor()
model.fit(X, Y)


# Make predictions for missing patches
missing_patches = [[13, 14, 15], [16, 17, 18]]
predictions = model.predict(missing_patches)

print(predictions)
# Should print [5,6]


# -------------------------Random Forest Model Training-------------------------#
# Get the data
X = np.load('data.npy')
y = np.load('labels.npy')

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create the Random Forest Regressor
rf_regressor = RandomForestRegressor(
    n_estimators=100, max_depth=20, min_samples_leaf=5, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions
y_pred = rf_regressor.predict(X_test)

# Calculate the accuracy
accuracy = rf_regressor.score(X_test, y_test)

print("Accuracy: ", accuracy)

# -------------------------Neural Network -------------------------#
class ImageInpainting():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self.w1 = torch.zeros((60*60*3, 40*40*3), device='cuda')
        self.b1 = torch.zeros((40*40*3), device='cuda')
        self.w2 = torch.zeros((40*40*3, 30*30*3), device='cuda')
        self.b2 = torch.zeros((30*30*3), device='cuda')
        self.w3 = torch.zeros((30*30*3, 25*25*3), device='cuda')
        self.b3 = torch.zeros((25*25*3), device='cuda')
        self.w4 = torch.zeros((25*25*3, 20*20*3), device='cuda')
        self.b4 = torch.zeros((20*20*3), device='cuda')

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float, device='cuda')
        x = x.flatten()
        x = x.reshape(1, -1)

        # Forward pass
        z1 = torch.matmul(x, self.w1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = torch.matmul(a1, self.w2) + self.b2
        a2 = self.sigmoid(z2)
        z3 = torch.matmul(a2, self.w3) + self.b3
        a3 = self.sigmoid(z3)
        z4 = torch.matmul(a3, self.w4) + self.b4
        a4 = self.sigmoid(z4)

        out = a4.cpu().detach().numpy()
        out = out.reshape((20,20,3))
  
        return out

    def train(self, epochs=1000, learning_rate=0.1):
      for epoch in range(epochs):

          for x, y in zip(self.X, self.Y):
              
              # print("ground truth: ")
              # plt.imshow(y.cpu().detach().numpy())
              # plt.show()
              # print("prediction: ")
              # image = self.predict(x)
              # plt.imshow(image)
              # plt.show()
              x = x.reshape(1,-1)
              x = torch.tensor(x, dtype=torch.float, device='cuda')
              y = torch.tensor(y, dtype=torch.float, device='cuda')
              y = y.flatten().reshape((1,20*20*3))
              y = y.type(torch.float) / 255

              # Forward pass
              z1 = torch.matmul(x, self.w1) + self.b1
              a1 = self.sigmoid(z1)
              z2 = torch.matmul(a1, self.w2) + self.b2
              a2 = self.sigmoid(z2)
              z3 = torch.matmul(a2, self.w3) + self.b3
              a3 = self.sigmoid(z3)
              z4 = torch.matmul(a3, self.w4) + self.b4
              a4 = self.sigmoid(z4)

              # Backward pass
              error = (y-a4)
              loss = torch.mean(torch.abs(error))

              print("Loss: ", loss)
        
              derror = error
              dz4 = derror * self.sigmoid_derivative(a4)
              dw4 = torch.matmul(a3.T, dz4)
              db4 = torch.sum(dz4, axis=0)
              da3 = torch.matmul(dz4, self.w4.T)
              dz3 = da3 * self.sigmoid_derivative(a3)
              dw3 = torch.matmul(a2.T, dz3)
              db3 = torch.sum(dz3, axis=0)
              da2 = torch.matmul(dz3, self.w3.T)
              dz2 = da2 * self.sigmoid_derivative(a2)
              dw2 = torch.matmul(a1.T, dz2)
              db2 = torch.sum(dz2, axis=0)
              da1 = torch.matmul(dz2, self.w2.T)
              dz1 = da1 * self.sigmoid_derivative(a1)
              dw1 = torch.matmul(x.T, dz1)
              db1 = torch.sum(dz1, axis=0)

               # Update weights and biases
              self.w4 += learning_rate * dw4
              self.b4 += learning_rate * db4
              self.w3 += learning_rate * dw3
              self.b3 += learning_rate * db3
              self.w2 += learning_rate * dw2
              self.b2 += learning_rate * db2
              self.w1 += learning_rate * dw1
              self.b1 += learning_rate * db1

      print(f"Epoch {epoch+1}/{epochs} Loss: {loss}")  


