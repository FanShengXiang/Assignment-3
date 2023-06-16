import numpy as np
import time
from net import LeNet5
from tools import normalization
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
import cv2

def read_path(directory):
    paths = []
    labels = []
    with open(directory, 'r') as f:
        for line in f:
            paths.append(line.split(' ')[0])
            labels.append(line.split(' ')[1])
    labels = np.array(labels, dtype=np.uint8)
    return paths, labels

class InputLayer:
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def forward(self, X):
        self.X = np.zeros((len(X), self.input_shape[0], self.input_shape[1]))
        for i, path in enumerate(X):
            with Image.open(path) as img:
                img = img.resize(self.input_shape[:2])
                img_arr = np.asarray(img)
                if len(img_arr.shape) == 3:
                    img_arr = img_arr[:,:,0] + img_arr[:,:,1] + img_arr[:,:,2]
                    #img_arr = img_arr[:, :, :self.input_shape[-1]]
                    self.X[i] = img_arr
        return self.X

def output_tensor(input_layer, output_file, img_paths):

    # check if output file exists
    if os.path.exists(output_file):
        # load output file and print its shape
        output = np.load(output_file)
        print('Output loaded from', output_file)
        print('Output shape:', output.shape)
        #output = output.reshape(output.shape[0]*3, 28*28)
        output = (output.astype(np.uint8)).reshape(output.shape[0], 28,28,1)
        return output
    else:
        # pass input image paths to the input layer
        output = input_layer.forward(img_paths)

        # save output to file
        np.save(output_file, output)
        print('Output saved to', output_file)
        print('Output shape:', output.shape)
        output = (output.astype(np.uint8)).reshape(output.shape[0], 28,28,1)
        return output

trainPaths, train_labels = read_path('train.txt')
testPaths, test_labels = read_path('test.txt')
valPaths, val_labels = read_path('val.txt')

train_images = output_tensor(InputLayer(input_shape=(28, 28, 3)), "trainOutput.npy", trainPaths)
test_images = output_tensor(InputLayer(input_shape=(28, 28, 3)), "testOutput.npy", testPaths)

batch_size = 100  # training batch size
test_batch = 50  # test batch size
epoch = 20
learning_rate = 1e-3

ax = []  # 保存 training 過程中x軸的數據（訓練次數）用於畫圖
ay_loss = []  # 保存 training 過程中y軸的數據（loss）用於畫圖
ay_acc = []
testx = [] # 保存 test 過程中x軸的數據（訓練次數）用於畫圖
testy_acc = []  # 保存 training 過程中y軸的數據（loss）用於畫圖
plt.ion()   
iterations_num = 0 # 紀錄訓練的迭代次數
plt.rcParams['font.sans-serif']=['SimHei']   #防止中文標籤亂碼
plt.rcParams['axes.unicode_minus'] = False

net = LeNet5.LeNet5()

for E in range(epoch):
    batch_loss = 0
    batch_acc = 0

    epoch_loss = 0
    epoch_acc = 0

    for i in range(train_images.shape[0] // batch_size):
        img = train_images[i*batch_size:(i+1)*batch_size].reshape(batch_size, 1, 28, 28)
        img = normalization.normalization(img)
        label = train_labels[i*batch_size:(i+1)*batch_size]
        loss, prediction = net.forward(img, label, is_train=True)   # 訓練階段

        epoch_loss += loss
        batch_loss += loss
        for j in range(prediction.shape[0]):
            if np.argmax(prediction[j]) == label[j]:
                epoch_acc += 1
                batch_acc += 1

        net.backward(learning_rate)

        if (i+1)%50 == 0:
            print(time.strftime("%Y-%m-%d %H:%M:%S") +
                  "   epoch:%5d , batch:%5d , avg_batch_acc:%.4f , avg_batch_loss:%.4f , lr:%f "
                  % (E+1, i+1, batch_acc/(batch_size*50), batch_loss/(batch_size*50), learning_rate))
            # 繪製loss和acc變化曲線
            plt.figure(1)
            iterations_num += 1
            plt.clf()
            ax.append(iterations_num)
            ay_loss.append(batch_loss/(batch_size*50))
            ay_acc.append(batch_acc/(batch_size*50))
            plt.subplot(1, 2, 1)
            plt.title('Train_Loss')
            plt.xlabel('Iteration', fontsize=10)
            plt.ylabel('loss', fontsize=10)
            plt.plot(ax, ay_loss, 'g-')

            plt.subplot(1, 2, 2)
            plt.title('Train_Acc')
            plt.xlabel('Iteration', fontsize=10)
            plt.ylabel('Acc', fontsize=10)
            plt.plot(ax, ay_acc, 'g-')
            plt.pause(0.4)

            batch_loss = 0
            batch_acc = 0



    print(time.strftime("%Y-%m-%d %H:%M:%S") +
          "    **********epoch:%5d , avg_epoch_acc:%.4f , avg_epoch_loss:%.4f *************"
          % (E+1, epoch_acc/train_images.shape[0], epoch_loss/train_images.shape[0]))
    # 在test set上進行測試
    test_acc = 0
    for k in range(test_images.shape[0] // test_batch):
        img = test_images[k*test_batch:(k+1)*test_batch].reshape(test_batch, 1 ,28, 28)
        img = normalization.normalization(img)
        label = test_labels[k*test_batch:(k+1)*test_batch]
        _, prediction = net.forward(img, label, is_train=False)   # 測試階段

        for j in range(prediction.shape[0]):
            if np.argmax(prediction[j]) == label[j]:
                test_acc += 1

    print("------------test_set_acc:%.4f---------------" % (test_acc / test_images.shape[0]))
    plt.figure(2)
    plt.clf()
    testx.append(E)
    testy_acc.append(test_acc / test_images.shape[0])
    plt.subplot()
    plt.title('Test_Acc')
    plt.xlabel('Iteration', fontsize=10)
    plt.ylabel('Acc', fontsize=10)
    plt.plot(testx, testy_acc, 'g-')
    plt.pause(0.4)  # 設置暫停時間，太快圖表無法正常顯示


plt.ioff()       
plt.show()      
#%%
"""
LeNet5(Keras Version)

"""
import tensorflow as tf
from tensorflow.keras import layers

# LeNet-5 model
def lenet5():
    model = tf.keras.Sequential()
    
    model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(50, activation='softmax'))
    
    return model




#%%
import numpy as np
from PIL import Image
import os

# Define the directory where your image data is stored


# Define the image size for resizing (if necessary)
image_size = (32, 32)

# Initialize empty lists for images and labels
def create_dataset(Paths):
    images = []
    labels = []
    
    # Iterate through each image file in the directory
    for filename in tqdm(Paths):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPEG'):  # Adjust file extensions as needed
            # Load and preprocess the image
            img = Image.open(filename)
            img = img.convert('RGB')  # Convert to RGB format if needed
    
            if image_size is not None:
                img = img.resize(image_size)  # Resize image if needed
    
            img_array = np.array(img)
            images.append(img_array)
    
            # Extract the label from the filename or directory structure
            # label = filename.split('.')[0]  # Modify this to extract label based on your data's naming convention
            labels.append(filename[7:16])
    
    # Convert the lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    return images ,labels

# Print the shape of the image and label arrays



#%%
trainPaths, train_labels = read_path('train.txt')
testPaths, test_labels = read_path('test.txt')
valPaths, val_labels = read_path('val.txt')
train_images, train_labels = create_dataset(trainPaths)
test_images, test_labels = create_dataset(testPaths)
val_images, val_labels = create_dataset(valPaths)
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder, StandardScaler
lb = LabelEncoder()
train_labels = np_utils.to_categorical(lb.fit_transform(train_labels))
test_labels = np_utils.to_categorical(lb.transform(test_labels))
val_labels = np_utils.to_categorical(lb.transform(val_labels))
# Train the model
#%%
# Create an instance of the LeNet-5 model
model = lenet5()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.CategoricalCrossentropy(), 
              metrics=['accuracy'])
import tensorflow as tf
tf.config.run_functions_eagerly(True)

model.fit(train_images, train_labels, epochs=50, batch_size=128,validation_data=(val_images,val_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc}")

#%%

import tensorflow as tf

# 將LeNet-5架構定義為靜態圖
def lenet5_static(x):
    # 卷積層
    x = tf.layers.conv2d(x, filters=6, kernel_size=5, activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
    x = tf.layers.conv2d(x, filters=16, kernel_size=5, activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)

    # 全連接層
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, units=120, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=84, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=10, activation=None)

    return x


inputs = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
outputs = lenet5_static(inputs)


# 將LeNet-5架構定義為動態圖
class LeNet5Dynamic(tf.keras.Model):
    def __init__(self):
        super(LeNet5Dynamic, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation=tf.nn.relu)
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation=tf.nn.relu)
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=120, activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(units=84, activation=tf.nn.relu)
        self.fc3 = tf.keras.layers.Dense(units=10, activation=None)

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
model = LeNet5Dynamic()

#%%
