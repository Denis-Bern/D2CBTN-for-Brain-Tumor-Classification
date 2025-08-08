import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.dpi'] = 300
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os

from tensorflow.keras.applications import EfficientNetB0, InceptionResNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

import ipywidgets as widgets
import io
from PIL import Image
from IPython.display import display,clear_output
from warnings import filterwarnings

# Get the list of all available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
# Specify all available GPUs in the MirroredStrategy configuration
strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{i}" for i in range(len(gpus))])


colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']


DATA_DIR = "Dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")
IMG_SIZE = (224,224)
RANDOM_SEED = 42

# Loop through each files
file_count = 0;
training_dict = {"glioma_tumor" : 0, "meningioma_tumor": 0, "no_tumor": 0, "pituitary_tumor":0}
testing_dict = {"glioma_tumor" : 0, "meningioma_tumor": 0, "no_tumor": 0, "pituitary_tumor":0}


for split in os.listdir(DATA_DIR):
    if split == "Training":
        for category in os.listdir(os.path.join(DATA_DIR, split)):
            for file_name in os.listdir(os.path.join(DATA_DIR, split, category)):
                file_count+=1
                training_dict[category] += 1
    if split == "Testing":
        for category in os.listdir(os.path.join(DATA_DIR, split)):
            for file_name in os.listdir(os.path.join(DATA_DIR, split, category)):
                file_count+=1
                testing_dict[category] += 1


total_dict = {}
for key in training_dict.keys():
    total_dict[key] = training_dict[key] + testing_dict[key]

print(file_count)
print(training_dict)
print(testing_dict)
print(total_dict)


import matplotlib.pyplot as plt
import numpy as np

labels_list = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
train_counts = list(training_dict.values())  # Convert dict_values to a list
test_counts = list(testing_dict.values())    # Convert dict_values to a list

x = np.arange(len(labels_list))  # the label locations
width = 0.35  # the width of the bars

# Set the size of the plot
fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, train_counts, width, label='Training')
rects2 = ax.bar(x + width/2, test_counts, width, label='Testing')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Counts', fontsize=18)
ax.set_title('Counts by split and class', fontsize=18)
ax.set_xticks(x)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.set_xticklabels(labels_list)  # Set the labels directly without passing them to set_xticks
ax.legend()

# Annotate bars with counts
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']

X_train = []
y_train = []
image_size = 224 #150
for i in labels:
    folderPath = os.path.join('Dataset','Training',i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size, image_size))
        X_train.append(img)
        y_train.append(i)

for i in labels:
    folderPath = os.path.join('Dataset','Testing',i)
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        y_train.append(i)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Plotting
num_rows = 3  # Swapping rows and columns
num_cols = 4  # Swapping rows and columns
num_images = min(len(labels) * num_cols, len(X_train))  # Ensure not to exceed available images

fig, ax = plt.subplots(num_rows, num_cols, figsize=(8, 6))  # Swapping figsize parameters

k = 0
for i in labels:
    j = 0
    count = 0
    while count < num_rows and k < num_images:  # Swapping num_rows and num_cols
        if y_train[j] == i:
            ax[k % num_rows, k // num_rows].imshow(X_train[j], cmap='gray')  # Swapping indices
            ax[k % num_rows, k // num_rows].set_title(f"{y_train[j]}")  # Swapping indices
            ax[k % num_rows, k // num_rows].axis('off')  # Swapping indices
            count += 1
            k += 1
        j += 1

plt.tight_layout()
plt.show()

X_train, y_train = shuffle(X_train, y_train, random_state=101)

for label in labels:
    count = np.sum(y_train == label)
    print(f"Class '{label}' has {count} images.")

print("X_train data: ", X_train.shape)
print("y_train data: ", y_train.shape)


y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train = y_train_new
y_train = tf.keras.utils.to_categorical(y_train)


#Building Model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers

num_classes = 4
input_shape = (224, 224, 3)
patch_size = (4, 4)  # 2-by-2 sized patches
dropout_rate = 0.03  # Dropout rate
num_heads = 8  # Attention heads
embed_dim = 64  # Embedding dimension
num_mlp = 256  # MLP layer size
qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window
image_dimension = 224  # Initial image size # 32

num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]

learning_rate = 1e-3
batch_size = 32 #128
num_epochs = 150
validation_split = 0.1
weight_decay = 0.0001
label_smoothing = 0.1

def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        x, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
    return windows


def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        windows,
        shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output

# Helper functions
def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        x, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
    return windows


def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        windows,
        shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output

# Window based multi-head self-attention
class WindowAttention(layers.Layer):
    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

    def build(self, input_shape):
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,)
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv


# The complete Swin Transformer model
class SwinTransformer(layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = DropPath(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

    def call(self, x):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size, self.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x

class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))

class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super().__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)

# Build the model
#from tensorflow.keras.layers import RandomCrop
def create_swin_transformer(input_shape, image_dimension, patch_size, num_patch_x, num_patch_y, embed_dim,
                            num_heads, window_size, shift_size, num_mlp, qkv_bias, dropout_rate, num_classes,
                            label_smoothing, learning_rate, weight_decay):
    
    input_layer = layers.Input(input_shape)
    x = layers.RandomCrop(image_dimension, image_dimension)(input_layer)
    x = layers.RandomFlip("horizontal")(x)
    x = PatchExtract(patch_size)(x)
    x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(x)
    #x = PatchEmbedding(num_patch_x=num_patch_x, num_patch_y=num_patch_y, embed_dim=embed_dim)(x)
    x = SwinTransformer(dim=embed_dim, num_patch=(num_patch_x, num_patch_y),num_heads=num_heads, window_size=window_size, shift_size=0,num_mlp=num_mlp,
                        qkv_bias=qkv_bias,dropout_rate=dropout_rate,)(x)

    x = SwinTransformer(dim=embed_dim, num_patch=(num_patch_x, num_patch_y), num_heads=num_heads, window_size=window_size, shift_size=shift_size,
                        num_mlp=num_mlp, qkv_bias=qkv_bias, dropout_rate=dropout_rate,)(x)

    x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)

    x = layers.GlobalAveragePooling1D()(x)

    output = layers.Dense(num_classes, activation="softmax")(x)
    
    # Create model
    model = models.Model(input_layer, output)
    
    # Compile model
    model.compile(loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
              optimizer=tf.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay),
              metrics=[keras.metrics.CategoricalAccuracy(name="accuracy"),
                       keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),],)
    
    return model

# Example usage:
model = create_swin_transformer(input_shape, image_dimension, patch_size, num_patch_x, num_patch_y, embed_dim,
                                num_heads, window_size, shift_size, num_mlp, qkv_bias, dropout_rate, num_classes,
                                label_smoothing, learning_rate, weight_decay)
model.summary()


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
import itertools
import time
import os
# Predict classes for test data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

def func_cv(X_train, y_train, cv_itr, n_cv):
    n_classes = 4
    ix = 0
    train_history = []
    cm = np.zeros((n_classes, n_classes), dtype=int)
    cm_agg = np.zeros((n_classes, n_classes), dtype=int)
   
    res_acc = np.zeros(n_cv)
    res_pre = np.zeros(n_cv)
    res_rec = np.zeros(n_cv)
    res_spec = np.zeros(n_cv)
    res_f1 = np.zeros(n_cv)
    
    # Lists to store classification reports from each fold
    all_classification_reports = []
    labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
    
    # Lists to store training and validation accuracies and losses from each fold
    all_train_acc = []
    all_train_loss = []
    all_val_acc = []
    all_val_loss = []
    
    ####################################################################################################
    # Overall Metrics for each classes
    ACC_glioma = np.zeros(n_cv)
    PPV_glioma = np.zeros(n_cv)
    TPR_glioma = np.zeros(n_cv)
    TNR_glioma = np.zeros(n_cv)
    F1_glioma  = np.zeros(n_cv)

    ACC_meningioma_tumor = np.zeros(n_cv)
    PPV_meningioma_tumor = np.zeros(n_cv)
    TPR_meningioma_tumor = np.zeros(n_cv)
    TNR_meningioma_tumor = np.zeros(n_cv)
    F1_meningioma_tumor = np.zeros(n_cv)

    ACC_no_tumor = np.zeros(n_cv)
    PPV_no_tumor = np.zeros(n_cv)
    TPR_no_tumor = np.zeros(n_cv)
    TNR_no_tumor = np.zeros(n_cv)
    F1_no_tumor = np.zeros(n_cv)

    ACC_pituitary_tumor = np.zeros(n_cv)
    PPV_pituitary_tumor = np.zeros(n_cv)
    TPR_pituitary_tumor = np.zeros(n_cv)
    TNR_pituitary_tumor = np.zeros(n_cv)
    F1_pituitary_tumor = np.zeros(n_cv)
    ####################################################################################################
    
    ## Overall
    ACC_ = np.zeros(n_cv)
    PPV_ = np.zeros(n_cv)
    TPR_ = np.zeros(n_cv)
    TNR_ = np.zeros(n_cv)
    F1_ = np.zeros(n_cv)
    

    # Create a directory to store PDF files if it doesn't exist
    output_dir = "10_FOLD_Swin_Transformer_Model"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for train_ix, test_ix in cv_itr.split(X_train, y_train):
        x_tr, y_tr = X_train[train_ix], y_train[train_ix]
        x_ts, y_ts = X_train[test_ix], y_train[test_ix]
        
        ####################################################################################################
        # Define data augmentation parameters
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Fit the data generator on the training data
        datagen.fit(x_tr)
        
        # Set the maximum number of augmented samples
        max_samples = 13056

        # Augment the training data and limit the number of samples
        augmented_samples = 0
        augmented_X_tr = []
        augmented_y_tr = []
        for X_batch, y_batch in datagen.flow(x_tr, y_tr, batch_size=1):
            augmented_X_tr.append(X_batch[0])
            augmented_y_tr.append(y_batch[0])
            augmented_samples += 1
            if augmented_samples >= max_samples:
                break

        # Convert the augmented data to numpy arrays
        augmented_X_tr = np.array(augmented_X_tr)
        augmented_y_tr = np.array(augmented_y_tr)
        
        
        # Plot examples of augmented images
        plt.figure(figsize=(8, 6))
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.imshow(augmented_X_tr[i].astype('uint8'))
            plt.axis('off')

        plt.tight_layout()
        plt.show()
        
        ####################################################################################################
        # MODEL 
        with strategy.scope():
            model = create_swin_transformer(input_shape, image_dimension, patch_size, num_patch_x, num_patch_y, embed_dim,
                                num_heads, window_size, shift_size, num_mlp, qkv_bias, dropout_rate, num_classes,
                                label_smoothing, learning_rate, weight_decay)
            
            history = model.fit(augmented_X_tr,augmented_y_tr, validation_data=(x_ts, y_ts), 
                                epochs = num_epochs, verbose=1, batch_size=32) #callbacks=[tensorboard,checkpoint,reduce_lr]
        #train_history = train_history.append(history)
        
        
        ####################################################################################################    
        print('\n\n')
        print(time.strftime('%X %x %Z'))
        print('{0}-CV RESULTS'.format(ix))
        ####################################################################################################
        # Predict classes for test data
        pred = model.predict(x_ts)
        pred_classes = np.argmax(pred, axis=1)
        y_ts_new = np.argmax(y_ts, axis=1)
        cm = confusion_matrix(y_ts_new, pred_classes)
        print(cm)

        #Plot the confusion matrix.
        sns.heatmap(cm,
                    annot=True,
                    fmt='g',
                    xticklabels=['glioma_tumor', 'no_tumor', 'meningioma_tumor','pituitary_tumor'],
                    yticklabels=['glioma_tumor', 'no_tumor', 'meningioma_tumor','pituitary_tumor'],
                    cmap='Blues')  # Set the colormap to Blues for blue color)
        plt.ylabel('Prediction',fontsize=18)
        plt.xlabel('Actual',fontsize=18)
        plt.title('Confusion Matrix',fontsize=18)
        # Save the plot as a PDF file
        plt.savefig(os.path.join(output_dir, f'Comfusion_matrix_10_FOLD_Swin_Transformer_Fold_{ix}.pdf'))
        plt.show()
        print('\n\n')
        
        # Aggregate confusion matrices
        #cm_agg += cm
        cm_agg = np.add(cm_agg, cm)
        ####################################################################################################
        # Classification Report
        classification_rep = classification_report(y_ts_new, pred_classes)
        print("\nClassification Report:")
        print(classification_rep)
        print('\n\n')
        
        # Append classification report to the list
        all_classification_reports.append(classification_rep)
        ####################################################################################################
        # Plot Learning Curves
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        epochs = range(len(acc))
        fig = plt.figure(figsize=(8, 5))
        plt.plot(epochs, acc, 'r', label="Training Accuracy")
        plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)
        plt.xticks(range(0, len(epochs), 15),fontsize=18)  # Set x-axis ticks at intervals of 5
        plt.yticks(fontsize=18)  # Set y-axis ticks fontsize
        plt.legend(fontsize=14)
        plt.grid(True)  # Add grid
        plt.gca().set_xlim(0, None)  # Set x-axis to start at 0
        plt.gca().set_ylim(0, None)  # Set y-axis to start at 0
        plt.legend(loc='lower right', fontsize=14)
        
        # Save the plot as a PDF file
        plt.savefig(os.path.join(output_dir, f'Training_curve_10_FOLD_Swin_Transformer_Fold_{ix}.pdf'))
        plt.show()
        print('\n\n')
        
        ####################################################################################################
        # Plot Loss Curves
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        fig = plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss, 'g', label="Training loss")
        plt.plot(epochs, val_loss, 'orange', label="Validation loss")
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.xticks(range(0, len(epochs), 15), fontsize=18)  # Set x-axis ticks at intervals of 5
        plt.yticks(fontsize=18)  # Set y-axis ticks fontsize
        plt.legend(fontsize=14)
        plt.grid(True)  # Add grid
        plt.gca().set_xlim(0, None)  # Set x-axis to start at 0
        plt.gca().set_ylim(0, None)  # Set y-axis to start at 0
        plt.legend(loc='upper right', fontsize=14)
        
        # Save the plot as a PDF file
        plt.savefig(os.path.join(output_dir, f'Loss_curve_10_FOLD_Swin_Transformer_Fold_{ix}.pdf'))
        plt.show()
        print('\n\n')
        
        # Append training and validation accuracies and losses to the lists
        all_train_acc.append(history.history['accuracy'])
        all_val_acc.append(history.history['val_accuracy'])
        all_train_loss.append(history.history['loss'])
        all_val_loss.append(history.history['val_loss'])
        ####################################################################################################
        
        # Binarize the output
        y_test_bin = label_binarize(y_ts_new, classes=[0, 1, 2, 3])

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        n_classes = 4

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curves
        plt.figure(figsize=(8, 6))

        colors = ['blue', 'orange', 'green', 'red']

        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{labels[i]} (area = {roc_auc[i]:.2f})')

        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'Micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks( fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('False Positive Rate', fontsize = 18)
        plt.ylabel('True Positive Rate', fontsize = 18)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize = 18)
        plt.legend(loc="lower right", fontsize = 14)
        # Save the plot as a PDF file
        plt.savefig(os.path.join(output_dir, f'Receiver Operating Characteristic_10_FOLD_Swin_Transformer_Fold_{ix}.pdf'))
        plt.show()
        
        # Calculate AUC
        print("\nAUC Scores:")
        for i in range(n_classes):
            print(f"AUC for class {labels[i]}: {roc_auc[i]:.2f}")
        # Micro-average AUC
        print(f"\nMicro-average AUC: {roc_auc['micro']:.2f}")
        print('\n\n')
        
        
        ####################################################################################################
        # Initialize variables to accumulate metrics for all classes
        total_TP = 0
        total_FP = 0
        total_FN = 0
        total_TN = 0

        # Initialize lists to store metrics for individual classes       

        # Loop through all classes
        for i in range(4):
            TP = cm[i, i]
            FP = np.sum(cm[:, i]) - TP
            FN = np.sum(cm[i, :]) - TP
            TN = np.sum(cm) - (TP + FP + FN)

            total_TP += TP
            total_FP += FP
            total_FN += FN
            total_TN += TN

            TPR = TP / (TP + FN) * 100
            TNR = TN / (TN + FP) * 100
            PPV = TP / (TP + FP) * 100
            ACC = (TP + TN) / (TP + FP + FN + TN) * 100
            F1 = 2 * TP / (2 * TP + FP + FN) * 100
           

            print(f"\nMetrics for class {labels[i]}:")
            print(f"Accuracy (ACC): {ACC:.2f}%")
            print(f"Precision (PPV): {PPV:.2f}%")
            print(f"Sensitivity (TPR): {TPR:.2f}%")
            print(f"Specificity (TNR): {TNR:.2f}%")
            print(f"F1-score (F1): {F1:.2f}%")
            print('\n\n')

            # 
            if(labels[i] =='glioma_tumor'):
                ACC_glioma[ix] = ACC
                PPV_glioma[ix] = PPV
                TPR_glioma[ix] = TPR
                TNR_glioma[ix] = TNR
                F1_glioma[ix] = F1
                
            if(labels[i] =='no_tumor'):
                ACC_meningioma_tumor[ix] = ACC
                PPV_meningioma_tumor[ix] = PPV
                TPR_meningioma_tumor[ix] = TPR
                TNR_meningioma_tumor[ix] = TNR
                F1_meningioma_tumor[ix] = F1
                
            if(labels[i] =='meningioma_tumor'):
                ACC_no_tumor[ix] = ACC
                PPV_no_tumor[ix] = PPV
                TPR_no_tumor[ix] = TPR
                TNR_no_tumor[ix] = TNR
                F1_no_tumor[ix] = F1
                
            if(labels[i] =='pituitary_tumor'):
                ACC_pituitary_tumor[ix] = ACC
                PPV_pituitary_tumor[ix] = PPV
                TPR_pituitary_tumor[ix] = TPR
                TNR_pituitary_tumor[ix] = TNR
                F1_pituitary_tumor[ix] = F1

        # Calculate overall metrics per fold
        overall_accuracy = (total_TP + total_TN) / (total_TP + total_FP + total_FN + total_TN) * 100
        overall_precision = total_TP / (total_TP + total_FP) * 100
        overall_sensitivity = total_TP / (total_TP + total_FN) * 100
        overall_specificity = total_TN / (total_TN + total_FP) * 100
        overall_f1_score = 2 * total_TP / (2 * total_TP + total_FP + total_FN) * 100
        
        
        ACC_[ix] = overall_accuracy
        PPV_[ix]= overall_precision
        TPR_[ix] = overall_sensitivity
        TNR_[ix]= overall_specificity
        F1_[ix]= overall_f1_score
        
        print('-' * 80)
        ix = ix + 1   
   
    ####################################################################################################
    # After the loop, compute the mean of each metric across all folds
    mean_overall_accuracy = np.mean(ACC_)
    mean_overall_precision = np.mean(PPV_)
    mean_overall_sensitivity = np.mean(TPR_)
    mean_overall_specificity = np.mean(TNR_)
    mean_overall_f1_score = np.mean(F1_)

    # Print the mean aggregated metrics
    print("\nMean Aggregated Metrics for 10 folds:")
    print(f"Aggregated Accuracy (ACC): {mean_overall_accuracy:.2f}%")
    print(f"Aggregated Precision (PPV): {mean_overall_precision:.2f}%")
    print(f"Aggregated Sensitivity (TPR): {mean_overall_sensitivity:.2f}%")
    print(f"Aggregated Specificity (TNR): {mean_overall_specificity:.2f}%")
    print(f"Aggregated F1-score (F1): {mean_overall_f1_score:.2f}%")
    print('\n\n')
    
    ####################################################################################################
    print('OVERALL FOR EACH CLASS')
    ####################################################################################################
    ## GLIOMA TUMOR
    ACC_glioma_ = np.mean(ACC_glioma)
    PPV_glioma_ = np.mean(PPV_glioma)
    TPR_glioma_ = np.mean(TPR_glioma)
    TNR_glioma_ = np.mean(TNR_glioma)
    F1_glioma_ = np.mean(F1_glioma)

    # Print the mean aggregated metrics
    print("\nMean Aggregated Metrics for GLIOMA TUMOR 10 folds:")
    print(f"Glioma Accuracy (ACC): {ACC_glioma_:.2f}%")
    print(f"Glioma Precision (PPV): {PPV_glioma_:.2f}%")
    print(f"Glioma Sensitivity (TPR): {TPR_glioma_:.2f}%")
    print(f"Glioma Specificity (TNR): {TNR_glioma_:.2f}%")
    print(f"Glioma F1-score (F1): {F1_glioma_:.2f}%")
    print('\n\n')
    ####################################################################################################
    
    ## No TUMOR
    ACC_no_tumor_ = np.mean(ACC_no_tumor)
    PPV_no_tumor_ = np.mean(PPV_no_tumor)
    TPR_no_tumor_ = np.mean(TPR_no_tumor)
    TNR_no_tumor_ = np.mean(TNR_no_tumor)
    F1_no_tumor_ = np.mean(F1_no_tumor)

    # Print the mean aggregated metrics
    print("\nMean Aggregated Metrics for No TUMOR 10 folds:")
    print(f"No-Tumor Accuracy (ACC): {ACC_no_tumor_:.2f}%")
    print(f"No-Tumor Precision (PPV): {PPV_no_tumor_:.2f}%")
    print(f"No-Tumor Sensitivity (TPR): {TPR_no_tumor_:.2f}%")
    print(f"No-Tumor Specificity (TNR): {TNR_no_tumor_:.2f}%")
    print(f"No-Tumor F1-score (F1): {F1_no_tumor_:.2f}%")
    print('\n\n')
    #####################################################################################################
    
    ## MENINGIOMA TUMOR
    ACC_meningioma_tumor_ = np.mean(ACC_meningioma_tumor)
    PPV_meningioma_tumor_ = np.mean(PPV_meningioma_tumor)
    TPR_meningioma_tumor_ = np.mean(TPR_meningioma_tumor)
    TNR_meningioma_tumor_ = np.mean(TNR_meningioma_tumor)
    F1_meningioma_tumor_ = np.mean(F1_meningioma_tumor)

    # Print the mean aggregated metrics
    print("\nMean Aggregated Metrics for MENINGIOMA TUMOR 10 folds:")
    print(f"Meningioma Accuracy (ACC): {ACC_meningioma_tumor_:.2f}%")
    print(f"Meningioma Precision (PPV): {PPV_meningioma_tumor_:.2f}%")
    print(f"Meningioma Sensitivity (TPR): {TPR_meningioma_tumor_:.2f}%")
    print(f"Meningioma Specificity (TNR): {TNR_meningioma_tumor_:.2f}%")
    print(f"Meningioma F1-score (F1): {F1_meningioma_tumor_:.2f}%")
    print('\n\n')
    #####################################################################################################
    
    ## PITUITARY TUMOR
    ACC_pituitary_tumor_ = np.mean(ACC_pituitary_tumor)
    PPV_pituitary_tumor_ = np.mean(PPV_pituitary_tumor)
    TPR_pituitary_tumor_ = np.mean(TPR_pituitary_tumor)
    TNR_pituitary_tumor_ = np.mean(TNR_pituitary_tumor)
    F1_pituitary_tumor_ = np.mean(F1_pituitary_tumor)

    # Print the mean aggregated metrics
    print("\nMean Aggregated Metrics for PITUITARY TUMOR 10 folds:")
    print(f"Pituitary Accuracy (ACC): {ACC_pituitary_tumor_:.2f}%")
    print(f"Pituitary Precision (PPV): {PPV_pituitary_tumor_:.2f}%")
    print(f"Pituitary Sensitivity (TPR): {TPR_pituitary_tumor_:.2f}%")
    print(f"Pituitary Specificity (TNR): {TNR_pituitary_tumor_:.2f}%")
    print(f"Pituitary F1-score (F1): {F1_pituitary_tumor_:.2f}%")
    #####################################################################################################
    
    # Plot the Aggregated confusion matrix
    print('\n\n')
    print("Aggregated confusion matrix")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_agg,
                annot=True,
                fmt='g',
                xticklabels=['glioma_tumor', 'no_tumor', 'meningioma_tumor','pituitary_tumor'],
                yticklabels=['glioma_tumor', 'no_tumor', 'meningioma_tumor','pituitary_tumor'],
                cmap='Blues')  # Set the colormap to Blues for blue color)
    plt.ylabel('Prediction',fontsize=18)
    plt.xlabel('Actual',fontsize=18)
    plt.title('Confusion Matrix',fontsize=18)
    plt.show()
    # Save the plot as a PDF file
    plt.savefig(os.path.join(output_dir, 'Aggregated_Comfusion_matrix_10_FOLD_Swin_Transformer.pdf'))
    #####################################################################################################
    
    # Aggregate training and validation accuracies and losses across all folds
    overall_train_acc = np.mean(np.array(all_train_acc), axis=0)
    overall_val_acc = np.mean(np.array(all_val_acc), axis=0)
    overall_train_loss = np.mean(np.array(all_train_loss), axis=0)
    overall_val_loss = np.mean(np.array(all_val_loss), axis=0)
    
    # Plot the overall training curve
    epochs = range(len(overall_train_acc))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, overall_train_acc, 'r', label="Training Accuracy")
    plt.plot(epochs, overall_val_acc, 'b', label="Validation Accuracy")
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.xticks(range(0, len(epochs), 15),fontsize=18)  # Set x-axis ticks at intervals of 5
    plt.yticks(fontsize=18)  # Set y-axis ticks fontsize
    plt.legend(fontsize=14)
    plt.grid(True)  # Add grid
    plt.gca().set_xlim(0, None)  # Set x-axis to start at 0
    plt.gca().set_ylim(0, None)  # Set y-axis to start at 0
    plt.legend(loc='lower right', fontsize=14)
    #plt.title('Training Curve',fontsize=18)
    
    # Save the plot as a PDF file
    plt.savefig(os.path.join(output_dir, f'Aggregated_Training_curve_10_FOLD_Swin_Transformer.pdf'))
    plt.show()
    
    # Plot the overall loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, overall_train_loss, 'g', label="Training Loss")
    plt.plot(epochs, overall_val_loss, 'orange', label="Validation Loss")
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.xticks(range(0, len(epochs), 15), fontsize=18)  # Set x-axis ticks at intervals of 5
    plt.yticks(fontsize=18)  # Set y-axis ticks fontsize
    plt.legend(fontsize=14)
    plt.grid(True)  # Add grid
    plt.gca().set_xlim(0, None)  # Set x-axis to start at 0
    plt.gca().set_ylim(0, None)  # Set y-axis to start at 0
    plt.legend(loc='upper right', fontsize=14)
    #plt.title('Loss Curve',fontsize=18)
    
    # Save the plot as a PDF file
    plt.savefig(os.path.join(output_dir, f'Aggregated_Loss_curve_10_FOLD_Swin_Transformer.pdf'))
    plt.show()
    #####################################################################################################
    
    # SAVE MODEL
    model.save_weights("10_FOLD_Swin_Transformer.h5") 

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import StratifiedShuffleSplit


n_cross_valid =10
kf = KFold(n_splits=n_cross_valid, shuffle=True, random_state=33)
#for unbalanced data sets,
# skf = StratifiedKFold(n_splits=n_cross_valid, shuffle=True, random_state=33) 
# sss = StratifiedShuffleSplit(n_splits=n_cross_valid, test_size=0.20, random_state=33)

import sys
import time

# org_stdout = sys.stdout
# f = open('AF_classification_result' + '.txt', 'w')
# sys.stdout = f

print('\n')
print('*'*80)
print('*'*80)
print('Swin_Transformer_MODEL')
print(time.strftime('%X %x %Z'))
print('*'*80)
print('*'*80)
func_cv(X_train, y_train, kf, n_cross_valid)  #skf , sss, kf 







