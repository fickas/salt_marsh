#All the required libraries loaded
#os library used ofr reading or writing files, creating and deleting directories or retrieving information about files.
import os

#Library importing inceptionV3 model and decode_predictions label.
from keras.applications.inception_v3 import InceptionV3, decode_predictions

#Library used to preprocess input data before feeding it to the InceptionV3 model
from keras.applications.inception_v3 import preprocess_input

#Library imports function Dense  used to define fully connected Neural Network
#Library imports function GlobalAveragePooling2D used to reduce the spatial dimensions of a 3D tensor
from keras.layers import Dense, GlobalAveragePooling2D

#Library is used to load image module and manipulate image data in TensorFlow
from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import load_img

#Library imports function that is used to convert a PIL image to NumPy array
from tensorflow.keras.preprocessing.image import img_to_array

#Library used to import model class which is used to define a neural network model
from tensorflow.keras import Model

#Library used to import generator class which is used to generate batches of augmented image data for training deep learning models.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Library used to work with arrays and matrices
import numpy as np

#Library used to import function train_test_split which is used to split a dataset into training and testing sets.
from sklearn.model_selection import train_test_split

#Library used to import function accuracy_score which is used to evaluate the accuracy of a classifier.
from sklearn.metrics import accuracy_score

#Library is used to load image module which is used for loading, manipulating and saving image files.
from PIL import Image

# This line imports specific functions from the scikit-learn library for computing evaluation metrics.
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# This line imports the tifffile library for working with TIFF image files.
import tifffile

# This line imports the pyplot module from the matplotlib library for creating plots.
import matplotlib.pyplot as plt


#This line imports the albumentations module for augmenting the data
import albumentations as A

# This line imports the cv2 module from OpenCV library.
import cv2




def RGB_Image_To_Readable(folder_path_add,output_folder_add):
  # Folder path containing the TIFF files
  folder_path = folder_path_add

  # Output folder for normalized images
  output_folder = output_folder_add

  # Create the output folder if it doesn't exist
  os.makedirs(output_folder, exist_ok=True)

  i=0

  # Process each TIFF file in the folder
  for filename in os.listdir(folder_path):
    try:
      if filename.endswith('.tif'):
          # Load the TIFF file
          tiff_path = os.path.join(folder_path, filename)
          tiff_data = tifffile.imread(tiff_path)

          # Normalize the pixel values to the range of [0, 1]
          tiff_data_normalized = tiff_data.astype(np.float32) / np.max(tiff_data)

          # Create a PIL Image from the normalized data
          image = Image.fromarray((tiff_data_normalized * 255).astype(np.uint8))

          # Save the normalized image
          output_path = os.path.join(output_folder, filename)
          image.save(output_path)



          print(f"Image '{filename}' saved successfully!")

          i=i+1


    except:
      print(f"Skipping file {filename} due to error")
      continue
  print(i)
  
def Augment_data(input_folder_1_link,input_folder_2_link,input_folder_3_link,output_dir1_link,output_dir2_link,output_dir3_link):
  input_folder_1 = input_folder_1_link
  input_folder_2 = input_folder_2_link
  input_folder_3 = input_folder_3_link

  output_dir_regular = output_dir1_link
  output_dir_irregular = output_dir2_link
  output_dir_other = output_dir3_link


  # Create the output directories if they don't exist
  if not os.path.exists(output_dir_regular):
    os.makedirs(output_dir_regular)
  if not os.path.exists(output_dir_irregular):
    os.makedirs(output_dir_irregular)
  if not os.path.exists(output_dir_other):
    os.makedirs(output_dir_other)

  # Define the augmentations
  transform = A.Compose([    A.HorizontalFlip(p=1),    A.VerticalFlip(p=1)])

  # Process input folder 1
  i = 0
  for image_file in os.listdir(input_folder_1):
      image_path = os.path.join(input_folder_1, image_file)
      image = tifffile.imread(image_path)
      augmented = transform(image=image)
      image_1 = augmented["image"]  # Original image
      image_2 = np.fliplr(image_1)  # Flipped horizontally
      image_3 = np.flipud(image_1)  # Flipped vertically
      image_4 = np.flipud(image_2)  # Flipped both horizontally and vertically
      # Save the four distinct images in output folder 1
      cv2.imwrite(os.path.join(output_dir_regular, str(i) + '_' + image_file), image_1)
      cv2.imwrite(os.path.join(output_dir_regular, str(i) + '_'  + "hflip_" + image_file), image_2)
      cv2.imwrite(os.path.join(output_dir_regular, str(i) + '_' + "vflip_" + image_file), image_3)
      cv2.imwrite(os.path.join(output_dir_regular, str(i) + '_' + "hvflip_" + image_file), image_4)
      i = i + 1

  # Process input folder 2
  i = 0
  for image_file in os.listdir(input_folder_2):

      image_path = os.path.join(input_folder_2, image_file)
      image = tifffile.imread(image_path)
      augmented = transform(image=image)
      image_1 = augmented["image"]  # Original image
      image_2 = np.fliplr(image_1)  # Flipped horizontally
      image_3 = np.flipud(image_1)  # Flipped vertically
      image_4 = np.flipud(image_2)  # Flipped both horizontally and vertically
      # Save the four distinct images in output folder 2
      cv2.imwrite(os.path.join(output_dir_irregular, str(i) + '_' + image_file), image_1)
      cv2.imwrite(os.path.join(output_dir_irregular, str(i) + '_'  + "hflip_" + image_file), image_2)
      cv2.imwrite(os.path.join(output_dir_irregular, str(i) + '_' + "vflip_" + image_file), image_3)
      cv2.imwrite(os.path.join(output_dir_irregular, str(i) + '_' + "hvflip_" + image_file), image_4)
      i = i + 1



  # Process input folder 3
  i = 0
  for image_file in os.listdir(input_folder_3):

      image_path = os.path.join(input_folder_3, image_file)
      image = tifffile.imread(image_path)
      augmented = transform(image=image)
      image_1 = augmented["image"]  # Original image
      image_2 = np.fliplr(image_1)  # Flipped horizontally
      image_3 = np.flipud(image_1)  # Flipped vertically
      image_4 = np.flipud(image_2)  # Flipped both horizontally and vertically
      # Save the four distinct images in output folder 2
      cv2.imwrite(os.path.join(output_dir_other, str(i) + '_' + image_file), image_1)
      cv2.imwrite(os.path.join(output_dir_other, str(i) + '_'  + "hflip_" + image_file), image_2)
      cv2.imwrite(os.path.join(output_dir_other, str(i) + '_' + "vflip_" + image_file), image_3)
      cv2.imwrite(os.path.join(output_dir_other, str(i) + '_' + "hvflip_" + image_file), image_4)
      i = i + 1
      
      
def test_augmentation(output_dir1_link,output_dir2_link,output_dir3_link): 
  # Define the output directories
  output_dir_regular = output_dir1_link
  output_dir_irregular = output_dir2_link
  output_dir_other = output_dir3_link


  image_count = len([f for f in os.listdir(output_dir_regular) if f.endswith('.tif')])
  print(f'Number of images in augmented folder: {image_count}')


  image_count = len([f for f in os.listdir(output_dir_irregular) if f.endswith('.tif')])
  print(f'Number of images in augmented folder: {image_count}')


  image_count = len([f for f in os.listdir(output_dir_other) if f.endswith('.tif')])
  print(f'Number of images in augmented folder: {image_count}')

def extract_features(directory):

    model = InceptionV3(weights='imagenet', pooling='avg', include_top=False)
    feature_dict = {}

    labels = os.listdir(directory)
    print(labels)

    for label in labels:
        label_dir = os.path.join(directory, label)
        tifs = [f for f in os.listdir(label_dir) if f.lower().endswith('.tif')]
        n = len(tifs)

        for i, f in enumerate(tifs):

            print(f'working on {f}: {i} of {n}.')
            image_path = os.path.join(label_dir, f)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, inception_size)
            x = np.array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            output = model.predict(x)
            feat = output.flatten()
            feature_dict[f] = [feat, label]

    return feature_dict

def image_feature_site(direc):
   model = InceptionV3(weights='imagenet', pooling='avg', include_top=False)
   feature_dict = {}
   tifs = [f for f in os.listdir(direc) if f.lower().endswith('.tif')]
   n = len(tifs)

   for i, f in enumerate(tifs):
      print(f'working on {f}: {i} of {n}.')
      image_path = os.path.join(direc, f)
      img = cv2.imread(image_path, cv2.IMREAD_COLOR)
      img = cv2.resize(img, inception_size)
      x = np.array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      output = model.predict(x)
      feat = output.flatten()
      feature_dict[f] = [feat, -1]
   return feature_dict

def threshold_results(thresh_list, actuals, predicted,average_value):
  result_df = pd.DataFrame(columns=['threshold', 'precision', 'recall', 'f1', 'accuracy'])
  for t in thresh_list:
    yhat = [1 if v >=t else 0 for v in predicted]
    #note: where TP=0, the Precision and Recall both become 0
    precision = precision_score(actuals, yhat, zero_division=0,average = average_value)
    recall = recall_score(actuals, yhat, zero_division=0,average = average_value)
    f1 = f1_score(actuals, yhat,average = average_value)
    accuracy = accuracy_score(actuals, yhat)
    result_df.loc[len(result_df)] = {'threshold':t, 'precision':precision, 'recall':recall, 'f1':f1, 'accuracy':accuracy}

  result_df = result_df.round(2)

  #Next bit fancies up table for printing. See https://betterdatascience.com/style-pandas-dataframes/
  #Note that fancy_df is not really a dataframe. More like a printable object.
  headers = {
    "selector": "th:not(.index_name)",
    "props": "background-color: #800000; color: white; text-align: center"
  }
  properties = {"border": "1px solid black", "width": "65px", "text-align": "center"}

  fancy_df = result_df.style.format(precision=2).set_properties(**properties).set_table_styles([headers])
  return (result_df, fancy_df)
