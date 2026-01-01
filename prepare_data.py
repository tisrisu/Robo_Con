# augmentation and data preprocessing (white background images)

import os 
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

raw_dir = "../dataset/raw"
output_dir = "../dataset/processed"

train_dir = os.path.join(output_dir , "train")
test_dir = os.path.join(output_dir , "test" )

if not os.path.exists(raw_dir):
    raise FileNotFoundError(f"Dataset not found: {raw_dir}")

os.makedirs(train_dir , exist_ok=True)
os.makedirs(test_dir , exist_ok = True)


datagen = ImageDataGenerator(
    
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.10,
    shear_range=0.1,
    fill_mode='nearest'
)

images_per_class = 200
train_split = 0.8
train_limit = int(images_per_class * train_split)

for symbol in os.listdir(raw_dir):
    symbol_path = os.path.join (raw_dir , symbol)

    if not os.path.isdir (symbol_path):
         continue 


    os.makedirs (os.path.join (train_dir , symbol) , exist_ok=True )
    os.makedirs (os.path.join (test_dir , symbol) , exist_ok=True )

    img_file = os.listdir(symbol_path)[0]
    img_path = os.path.join(symbol_path, img_file)

   


    img = cv2.imread(img_path)

    if img is None:
        print("Skipping unreadable file: " + img_path)
        continue

    img = cv2.resize (img , (224,224))
    img = img.reshape((1,) + img.shape)

    count = 0

    for batch in datagen.flow(img , batch_size = 1) :
         if count < train_limit :
                save_path = os.path.join(train_dir, symbol, f"{count}.png")

         else :
               save_path = os.path.join(test_dir, symbol, f"{count}.png")

         cv2.imwrite (save_path , batch[0])

         count += 1

         if count >= images_per_class :
              break
         
print("dataset generation complete")
              


    


