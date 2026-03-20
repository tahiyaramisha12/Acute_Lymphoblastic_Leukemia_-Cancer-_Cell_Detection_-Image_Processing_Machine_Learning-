import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

preprocessed_dir = 'preprocessed_dataset'
segmented_dir = 'segmented_dataset'

output_file = 'features.csv'

categories = {
    'cancer': ['Benign', 'Early Malignant', 'Pre Malignant', 'Pro Malignant'],
    'normal': []
}

radius = 1
n_points = 8 * radius

data = []

def extract_features(pre_img_path, seg_img_path, main_label, sub_label=None):
   
    pre_img = cv2.imread(pre_img_path, cv2.IMREAD_GRAYSCALE)
    seg_img = cv2.imread(seg_img_path, cv2.IMREAD_GRAYSCALE)
    
    if pre_img is None or seg_img is None:
        return None
    
    mean_val = np.mean(pre_img)
    var_val = np.var(pre_img)
    
    try:
        glcm = graycomatrix(pre_img, [1], [0], 256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0][0]
        energy = graycoprops(glcm, 'energy')[0][0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0][0]
    except:
        contrast = energy = homogeneity = 0
    
    try:
        lbp = local_binary_pattern(pre_img, n_points, radius, method='uniform')
        lbp_mean = np.mean(lbp)
        lbp_std = np.std(lbp)
    except:
        lbp_mean = lbp_std = 0
    
    _, thresh = cv2.threshold(seg_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if perimeter != 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
    else:
        area = perimeter = circularity = 0
    
    if main_label == 'normal':
        final_label = 'normal'
    else:
        final_label = sub_label
    
    return [mean_val, var_val, contrast, energy, homogeneity,
            lbp_mean, lbp_std, area, perimeter, circularity, final_label]

for category, subcategories in categories.items():
    if subcategories:
        for sub in subcategories:
            pre_path = os.path.join(preprocessed_dir, category, sub)
            seg_path = os.path.join(segmented_dir, category, sub)
            
            if os.path.exists(pre_path) and os.path.exists(seg_path):
                for img_name in os.listdir(pre_path):
                    pre_img_path = os.path.join(pre_path, img_name)
                    seg_img_path = os.path.join(seg_path, img_name)
                    
                    features = extract_features(pre_img_path, seg_img_path, category, sub)
                    if features:
                        data.append(features)
    else:
        pre_path = os.path.join(preprocessed_dir, category)
        seg_path = os.path.join(segmented_dir, category)
        
        if os.path.exists(pre_path) and os.path.exists(seg_path):
            for img_name in os.listdir(pre_path):
                pre_img_path = os.path.join(pre_path, img_name)
                seg_img_path = os.path.join(seg_path, img_name)
                
                features = extract_features(pre_img_path, seg_img_path, category)
                if features:
                    data.append(features)

columns = [
    'mean', 'variance',
    'contrast', 'energy', 'homogeneity',
    'lbp_mean', 'lbp_std',
    'area', 'perimeter', 'circularity',
    'label'
]

df = pd.DataFrame(data, columns=columns)
df.to_csv(output_file, index=False)

print(f"Feature extraction complete. Saved to {output_file}")
print(f"Total samples: {len(df)}")
print(f"Class distribution:\n{df['label'].value_counts()}")