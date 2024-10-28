
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras import regularizers,utils
from keras.optimizers import Adam
from keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import History
import os
import numpy as np
import os

import matplotlib.pyplot as plt


from PIL import Image
def plotHistoryOfTraining(stat: History,plotTittle:str)->None:

    """
    Δημιουργία ενός plot που περιγράφει την εξέλιξη της εκπαιδευσης του μοντέλου σε σχεση με 
    το συνόλο επικύρωσης και εκπαίδευσης
    Args:
        stat (History):Η εξέλιξη της εκπαίδευσης του μοντέλου.
        plotTittle (str): ο τιτλος του plot.
    """

    plt.plot(stat.history['accuracy'], label='Training Accuracy')
    plt.plot(stat.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(plotTittle)
    plt.legend()
    plt.show()
    
def load_and_preprocess_image(image_path:str, target_size:tuple[int,int])->Image:
    """
    βρισκεί μια εικόνα και την μετατρέπει σε συγκεκριμένες διαστάσεις.
        Args:
            image_path (str) η θέση της εικόνας
            target_size (tuple[int,int]) οι διαστάσεις στόχος που πρεπει να μετατραπεί

        Returns:
            image:  η νέα φωτογραφία. 
    """
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image)/255.0
    return image

def dataAugmentation(Xset: np.ndarray, Yset: np.ndarray,datagen:ImageDataGenerator)->tuple [np.ndarray,np.array]:
    """
    Δεχεται ένα σύνολο εισόδων, το τροποποίει με βάση έναν κανόνα και το ξαναπροσθέτει στο αρχικό συνολο
    (παράλληλα με τον πινακα με τους στόχους)
        Args:
            Xset(np.ndarray): Συνολο Εισόδων
            Yset(np.ndarray): Συνολο στόχων
            datagen (ImageDataGenerator): Κανόνας τροποποίησης
        return:
            tuple [np.ndarray,np.array]:[τροποποιήμενη είσοδος,τροποποιήμενοι στόχοι]
    """
    newX=Xset
    datagen.fit(newX)
    X_augmented = np.concatenate([Xset,np.array(newX)])
    Y_augmented = np.concatenate([Yset,Yset])
    return X_augmented,Y_augmented
def collectData(folder:str)->tuple[np.ndarray, np.ndarray]:
    """
    βρισκει όλες τις εικόνες σε ένα φάκελο και συνθέτει έναν πινακα εισόδων και εξόδων με βάση το όνομα τους.
    args:
        folder(string) :Ονομά φακέλου
    returns:
        tuple[np.ndarray, np.ndarray]:[input (X),ExpectedOutPut (Y)]
    """
    image_files = os.listdir(folder)
    Xset = []
    Yset=[]
    for image_file in image_files:
        if image_file.startswith("cat."):
            Yset.append(0)
        else:
            Yset.append(1)
        image_path = os.path.join(folder, image_file)
        image = load_and_preprocess_image(image_path, target_size=(64, 64))                                    
        Xset.append(image)
    Xset = np.array(Xset)
    Yset=np.array(Yset)
    np.savez('TrainData.npz', array1=Xset, array2=Yset)
    return Xset,Yset

def createModel(Xtrain:np.ndarray,Ytrain:np.ndarray,Xtest:np.ndarray,Ytest:np.ndarray,epochi:int,batch_size:int)->Sequential:
    """
    Εδω πραγματοποιείται η σύνθεση και η εκπαίδευση του μοντέλου 
    Args:
        Xtrain (np.ndarray) : Το συνολο εισόδου εκπαιδευσής
        Ytrain (np.ndarray) :Το συνολο στόχων εκπαιδευσης
        Xtest   (np.ndarray) : Το συνολο εισόδου επικύρωσης
        Ytest   (np.ndarray): Το συνολο εισόδου επικύρωσης
        epochi (int) :ο αριθμος των εποχών που θα εκπαιδευτεί το μοντέλο
        batch_size (int) :Το μέγεθος του κάθε batch
    """
    model = Sequential([
    Conv2D(128, (3, 3),strides=2,padding="same", activation='relu', input_shape=(64, 64, 3),kernel_initializer="he_normal"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.35),
    Conv2D(64, (3, 3), padding="same",activation='relu',kernel_initializer="he_normal"),
    #BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.35),
    Conv2D(128, (3, 3),padding="same", activation='relu',kernel_initializer="he_normal"),
    Dropout(0.3),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu',kernel_initializer="he_normal",kernel_regularizer=regularizers.l2(0.001)),
    #BatchNormalization(),
    Dropout(0.25),
    Dense(2, activation='softmax',kernel_initializer="glorot_uniform")
])
    model.compile(Adam(learning_rate=0.0001),loss="binary_crossentropy", metrics=['accuracy'])
    trainingHistory=model.fit(Xtrain,Ytrain,batch_size=batch_size,epochs=epochi,verbose=1,validation_data=(Xtest,Ytest))
    return model,trainingHistory
def loadData()->None:
    """
    Απλή μεθοδος που ελέγχει αν υπάρχει φακέλος με φορτωμένα δεδομένα και τα ανασύρει
    ή αλλιως τα δημιουργει 
    """
    if os.path.exists('TrainData.npz'):
        loaded_data = np.load('TrainData.npz')
        X = loaded_data['array1']
        Y = loaded_data['array2']
    else:
        X,Y=collectData(folder=os.path.join("dataSet","train"))
    return X,Y
def doingDataAugmentantionStuff(X:np.ndarray,Y:np.ndarray)->tuple[np.ndarray, :np.ndarray]:
    datagen = ImageDataGenerator(
    rotation_range=20,       
    width_shift_range=0.2,   
    height_shift_range=0.2,  
    shear_range=0.2,         
    zoom_range=0.2,          
    horizontal_flip=True,    
    vertical_flip=True,      
    brightness_range=[0.8, 1.2] , 
    fill_mode="nearest"
)
    Xaug,Yaug=dataAugmentation(X,Y,datagen)
    return Xaug,Yaug






def main()->None:
    
    X,Y=loadData()
    Y=utils.to_categorical(Y,num_classes=2)
    Xtrain,Xunknown,Ytrain,Yunknown= train_test_split(X,Y,test_size=0.2,random_state=42)
    Xvalid,Xtest,Yvalid,Ytest= train_test_split(Xunknown,Yunknown,test_size=0.5,random_state=42)
    Xtrain,Ytrain=doingDataAugmentantionStuff(Xtrain,Ytrain)
    cnn_model,h=createModel(Xtrain,Ytrain,Xvalid,Yvalid,epochi=250,batch_size=60)
    cnn_model.evaluate(Xtest,Ytest,batch_size=1)
    cnn_model.save('ModelCNN.keras') 
    plotHistoryOfTraining(h,"Training process")

if __name__=="__main__":
    main()
    
    