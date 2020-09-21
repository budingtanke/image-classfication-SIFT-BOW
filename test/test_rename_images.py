import os
import pandas as pd
import numpy as np

print(os.getcwd())
#path = os.getcwd()
test_path = '/Users/linhan/Desktop/Han/Learning/Project/Image Classification with SIFT BOW SVM/test/test_images/'
path = '/Users/linhan/Desktop/Han/Learning/Project/Image Classification with SIFT BOW SVM/data/google_images/'

# all file extension: {'jpg', 'gif', 'jpeg', 'DS_Store', 'png'}
file_extension = set()
for img_folder in os.listdir(path):
    if os.path.isdir(path + img_folder):
        this_folder = path + img_folder
        for file in os.listdir(this_folder):
            extension = file.split('.')[-1]
            file_extension.add(extension)
print(file_extension)

df_labels = pd.read_csv('/Users/linhan/Desktop/Han/Learning/Project/Image Classification with SIFT BOW SVM/test/label_train.csv')
print(df_labels)

i = 3001
'img_3000.jpg'
for img_folder in os.listdir(test_path):
    this_folder_index = -2
    if os.path.isdir(test_path + img_folder):
        this_folder = test_path + img_folder
        if img_folder == 'blueberry muffin':
            this_folder_index = 0
        elif img_folder == 'fried chicken':
            this_folder_index = 1
        elif img_folder == 'poodles':
            this_folder_index = 2

        for file in os.listdir(this_folder):
            if file.endswith('jpg') | file.endswith('png'):
                old_name = os.path.join(this_folder, file)
                new_name = os.path.join(this_folder, 'img_{}.jpg'.format(i))
                os.rename(old_name, new_name)

                df_labels = pd.concat([df_labels, pd.DataFrame(np.array([i,this_folder_index]).reshape(1,2), columns=df_labels.columns)])
                i += 1
df_labels.to_csv('./new_labels.csv', index=False)





