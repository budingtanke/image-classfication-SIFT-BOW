import os
import pandas as pd
import numpy as np

"""
cd to google-images-download and run the below commands to download from google images:
python3 bing_scraper.py --search “poodles” --limit 500 --download --chromedriver /Users/linhan/Desktop/Han/Learning/Python/Softwares/chromedriver
python3 bing_scraper.py --search “fried chicken” --limit 500 --download --chromedriver /Users/linhan/Desktop/Han/Learning/Python/Softwares/chromedriver
python3 bing_scraper.py --search “blueberry muffin” --limit 500 --download --chromedriver /Users/linhan/Desktop/Han/Learning/Python/Softwares/chromedriver

This script renames the file name, combines them with provided 3000 images
"""

#path = os.getcwd()
path = '/Users/linhan/Desktop/Han/Learning/Project/Image Classification with SIFT BOW SVM/data/google_images/'

# all file extension: {'jpg', 'gif', 'jpeg', 'DS_Store', 'png'}
# file_extension = set()
# for img_folder in os.listdir(path):
#     if os.path.isdir(path + img_folder):
#         this_folder = path + img_folder
#         for file in os.listdir(this_folder):
#             extension = file.split('.')[-1]
#             file_extension.add(extension)
# print(file_extension)

df_labels = pd.read_csv('/Users/linhan/Desktop/Han/Learning/Project/Image Classification with SIFT BOW SVM/data/label_train.csv')

i = 3001 # image index starts from 3001, because already have 3000 provided images
for img_folder in os.listdir(path):
    this_folder_index = -2
    if os.path.isdir(path + img_folder):
        this_folder = path + img_folder
        if img_folder == 'blueberry_muffin':
            this_folder_index = 0
        elif img_folder == 'fried_chicken':
            this_folder_index = 1
        elif img_folder == 'poodles':
            this_folder_index = 2

        for file in os.listdir(this_folder):
            if file.endswith('jpg') | file.endswith('png') | file.endswith('jpeg'):
                # rename original file name to image index name
                old_name = os.path.join(this_folder, file)
                new_name = os.path.join(this_folder, 'img_{}.jpg'.format(i))
                os.rename(old_name, new_name)

                # add labels
                df_labels = pd.concat([df_labels, pd.DataFrame(np.array([i, this_folder_index]).reshape(1, 2), columns=df_labels.columns)])
                i += 1
            else:
                # remove files if not ith desired format
                os.remove(os.path.join(this_folder, file))


df_labels.to_csv('/Users/linhan/Desktop/Han/Learning/Project/Image Classification with SIFT BOW SVM/data/new_labels.csv', index=False)





