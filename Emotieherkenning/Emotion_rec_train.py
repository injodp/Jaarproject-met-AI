import tensorflow as tf # Machine learning framework
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Voor augmentatie
from tensorflow.keras.models import Sequential # Modeltype
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # Neurale netwerk lagen

# Definieer paden naar dataset
train_map = "/home/injo/Emotion_Recognition/dataset/test"

# Voorbereiden van de dataset met beeldaumentatie
train_datagen = ImageDataGenerator(
    rescale = 1./225, # Notmalisatie ppixelwaarden naar 0-1 bereik
    rotation_range = 20, #willlekeurig rotaties toepassen
    horizontal_flip = True, # Willekeurig horizontaal spiegelen
    validation_split = 0.2 # 20% van de dataset gebruiken voor validatie
)

# Laad de trainingsdataset
train_generator = train_datagen.flow_from_directory(
    train_map, #map met dataset
    target_size = (48, 48), # afbeelding schalen naar 48 x 48 pixels
    color_mode = "grayscale", # gebruikt grijswaarden
    batch_size = 64, # aantal afbeeldingen per batch
    class_mode = "categorical", # meerdere klassen (emoties)
    subset = "training" # gebruikt 80% van de data voor training
)

#laad de validatieset
val_generator = train_datagen.flow_from_directory(
    train_map,
    target_size = (48, 48),
    color_mode = "grayscale",
    batch_size = 64,
    class_mode = "categorical",
    subset = "validation" # gebruik 20% van de data voor validatie
)

#opbouw neuraal netwerk
model = Sequential([
    Conv2D(32, (3,3), activation = "relu", input_shape = (48, 48, 1)), #eerste convolutielaag
    MaxPooling2D(2, 2), # verklein afbeelding met max pooling
    Conv2D(46, (3, 3), activation = "relu"), #tweede convolutielaag
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation = "relu"), #derde convolutielaag
    MaxPooling2D(2, 2),
    Flatten(), # zet de matrix om in een vector
    Dense(128, activation = "relu"), # volledig verbonden laag met 128 neuronen
    Dropout(0.5), #dropout om overfitting te verminderen
    Dense(len(train_generator.class_indices), activation = "softmax") #outputlag met het juiste aantal entities
])

# compileer het model
model.compile(loss="categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"]) # gebruikt adam optimizer

# train het model
model.fit(
    train_generator, # gebruik trainingsdataset
    validation_data = val_generator, #gebruikt validateidataset
    epochs = 100 # train het model gedurende 25 epochs
)

# sla het model op voor toekomstig gebruik
model.save("/home/injo/Emotion_Recognition/emotion_recognition_custom_model.h5")

print("Model getraind en opgeslgen als emotion_recognition_custom_model.h5")
