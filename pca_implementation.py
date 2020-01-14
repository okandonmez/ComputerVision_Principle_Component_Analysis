#region IMPORT AREA
import cv2
import numpy as np
import glob
import operator
from sklearn.decomposition import PCA
import os
from sklearn.preprocessing import StandardScaler
import math
#endregion

#region READING THE FACE DATAS AND SET FORMAT +
labels_of_data = []
face_datas = []
celeb_names = []

for face_name_path in sorted(glob.glob("inputs/VGG-Faces/CelebFaces/*")):

    celeb_name = face_name_path.split("/")[-1]
    celeb_names.append(celeb_name)

    for path in glob.glob(os.path.join(face_name_path, "*.jpg")):
        face = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        face_datas.append(face)
        labels_of_data.append(celeb_name)

face_datas = np.array(face_datas)
labels_of_data = np.array(labels_of_data)

dict_name_to_id = {
    x :i for i ,x in enumerate(np.unique(labels_of_data))
}



name_ids = np.array([dict_name_to_id[x] for x in labels_of_data])
#endregion

#region APPLY THE PCA PROCESS TO THE DATA TO BE TRAINED

temp_for_sca = StandardScaler()
flatten_images = temp_for_sca.fit_transform([i.flatten() for i in face_datas])

tool_pca = PCA(n_components = 100)
result_set = tool_pca.fit_transform(flatten_images)
train_face = np.column_stack((result_set ,name_ids))
#endregion
