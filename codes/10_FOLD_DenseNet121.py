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

# Color
colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']


DATA_DIR = "Dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")
IMG_SIZE = (224,224)
RANDOM_SEED = 


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
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121

def build_densenet_model():
    # Load pre-trained DenseNet121 model without the top (fully connected) layers
    densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the layers of the pre-trained model
    #for layer in densenet.layers:
    #    layer.trainable = False
    
    # Add custom top layers for classification
    x = densenet.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    predictions = tf.keras.layers.Dense(4, activation='softmax')(x)
    
    # Combine the base model with custom top layers
    model = tf.keras.models.Model(inputs=densenet.input, outputs=predictions)
    
    model.compile(loss='categorical_crossentropy',optimizer = 'Adam', metrics= ['accuracy'])
    
    return model

# Create the model
model = build_densenet_model()
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
    labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    
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
    output_dir = "10_FOLD_DenseNet121_Model"
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
            model = build_densenet_model()
            history = model.fit(augmented_X_tr,augmented_y_tr, validation_data=(x_ts, y_ts), 
                                epochs =50, verbose=1, batch_size=32) #callbacks=[tensorboard,checkpoint,reduce_lr]
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
                    xticklabels=['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor'],
                    yticklabels=['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor'],
                    cmap='Blues')  # Set the colormap to Blues for blue color)
        plt.ylabel('Prediction',fontsize=18)
        plt.xlabel('Actual',fontsize=18)
        plt.title('Confusion Matrix',fontsize=18)
        # Save the plot as a PDF file
        plt.savefig(os.path.join(output_dir, f'Comfusion_matrix_10_FOLD_DenseNet121_Fold_{ix}.pdf'))
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
        plt.xticks(range(0, len(epochs), 4),fontsize=18)  # Set x-axis ticks at intervals of 5
        plt.yticks(fontsize=18)  # Set y-axis ticks fontsize
        plt.legend(fontsize=14)
        plt.grid(True)  # Add grid
        plt.gca().set_xlim(0, None)  # Set x-axis to start at 0
        plt.gca().set_ylim(0, None)  # Set y-axis to start at 0
        plt.legend(loc='lower right', fontsize=14)
        
        # Save the plot as a PDF file
        plt.savefig(os.path.join(output_dir, f'Training_curve_10_FOLD_DenseNet121_Fold_{ix}.pdf'))
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
        plt.xticks(range(0, len(epochs), 4), fontsize=18)  # Set x-axis ticks at intervals of 5
        plt.yticks(fontsize=18)  # Set y-axis ticks fontsize
        plt.legend(fontsize=14)
        plt.grid(True)  # Add grid
        plt.gca().set_xlim(0, None)  # Set x-axis to start at 0
        plt.gca().set_ylim(0, None)  # Set y-axis to start at 0
        plt.legend(loc='upper right', fontsize=14)
        
        # Save the plot as a PDF file
        plt.savefig(os.path.join(output_dir, f'Loss_curve_10_FOLD_DenseNet121_Fold_{ix}.pdf'))
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
        plt.savefig(os.path.join(output_dir, f'Receiver Operating Characteristic_10_FOLD_DenseNet121_Fold_{ix}.pdf'))
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
                
            if(labels[i] =='meningioma_tumor'):
                ACC_meningioma_tumor[ix] = ACC
                PPV_meningioma_tumor[ix] = PPV
                TPR_meningioma_tumor[ix] = TPR
                TNR_meningioma_tumor[ix] = TNR
                F1_meningioma_tumor[ix] = F1
                
            if(labels[i] =='no_tumor'):
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
    
    ## PITUITARY TUMOR
    ACC_pituitary_tumor_ = np.mean(ACC_pituitary_tumor)
    PPV_pituitary_tumor_ = np.mean(PPV_pituitary_tumor)
    TPR_pituitary_tumor_ = np.mean(TPR_pituitary_tumor)
    TNR_pituitary_tumor_ = np.mean(TNR_pituitary_tumor)
    F1_pituitary_tumor_ = np.mean(F1_pituitary_tumor)

    # Print the mean aggregated metrics
    print("\nMean Aggregated Metrics for PITUITARY TUMOR 10 folds:")
    print(f"glioma Accuracy (ACC): {ACC_pituitary_tumor_:.2f}%")
    print(f"glioma Precision (PPV): {PPV_pituitary_tumor_:.2f}%")
    print(f"glioma Sensitivity (TPR): {TPR_pituitary_tumor_:.2f}%")
    print(f"glioma Specificity (TNR): {TNR_pituitary_tumor_:.2f}%")
    print(f"glioma F1-score (F1): {F1_pituitary_tumor_:.2f}%")
    #####################################################################################################
    
    # Plot the Aggregated confusion matrix
    print('\n\n')
    print("Aggregated confusion matrix")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_agg,
                annot=True,
                fmt='g',
                xticklabels=['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor'],
                yticklabels=['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor'],
                cmap='Blues')  # Set the colormap to Blues for blue color)
    plt.ylabel('Prediction',fontsize=18)
    plt.xlabel('Actual',fontsize=18)
    plt.title('Confusion Matrix',fontsize=18)
    plt.show()
    # Save the plot as a PDF file
    plt.savefig(os.path.join(output_dir, 'Aggregated_Comfusion_matrix_10_FOLD_DenseNet121.pdf'))
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
    plt.xticks(range(0, len(epochs), 4),fontsize=18)  # Set x-axis ticks at intervals of 5
    plt.yticks(fontsize=18)  # Set y-axis ticks fontsize
    plt.legend(fontsize=14)
    plt.grid(True)  # Add grid
    plt.gca().set_xlim(0, None)  # Set x-axis to start at 0
    plt.gca().set_ylim(0, None)  # Set y-axis to start at 0
    plt.legend(loc='lower right', fontsize=14)
    #plt.title('Training Curve',fontsize=18)
    
    # Save the plot as a PDF file
    plt.savefig(os.path.join(output_dir, f'Aggregated_Training_curve_10_FOLD_DenseNet121.pdf'))
    plt.show()
    
    # Plot the overall loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, overall_train_loss, 'g', label="Training Loss")
    plt.plot(epochs, overall_val_loss, 'orange', label="Validation Loss")
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.xticks(range(0, len(epochs), 4), fontsize=18)  # Set x-axis ticks at intervals of 5
    plt.yticks(fontsize=18)  # Set y-axis ticks fontsize
    plt.legend(fontsize=14)
    plt.grid(True)  # Add grid
    plt.gca().set_xlim(0, None)  # Set x-axis to start at 0
    plt.gca().set_ylim(0, None)  # Set y-axis to start at 0
    plt.legend(loc='upper right', fontsize=14)
    #plt.title('Loss Curve',fontsize=18)
    
    # Save the plot as a PDF file
    plt.savefig(os.path.join(output_dir, f'Aggregated_Loss_curve_10_FOLD_DenseNet121.pdf'))
    plt.show()
    #####################################################################################################
    
    # SAVE MODEL
    model.save_weights("10_FOLD_DenseNet121.h5") 



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
print('DenseNet121_MODEL')
print(time.strftime('%X %x %Z'))
print('*'*80)
print('*'*80)
func_cv(X_train, y_train, kf, n_cross_valid)  #skf , sss, kf  