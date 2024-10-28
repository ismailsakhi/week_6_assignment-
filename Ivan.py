import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

# Load data - replace with actual file paths
true_boxes = pd.read_excel("training.xlsx")
predicted_boxes = pd.read_excel("predictions_training.xlsx")

# IOU Calculation function
def calculate_iou(true_boxes, predicted_boxes):
    intersections = []
    unions = []

    for i in range(len(true_boxes)):
        true_box = true_boxes.iloc[i]
        pred_box = predicted_boxes.iloc[i]
        
        # Compute the intersection area
        x_overlap = max(0, min(true_box['max_c'], pred_box['max_c']) - max(true_box['min_c'], pred_box['min_c']))
        y_overlap = max(0, min(true_box['max_r'], pred_box['max_r']) - max(true_box['min_r'], pred_box['min_r']))
        intersection = x_overlap * y_overlap

        # Compute the union area
        true_area = (true_box['max_c'] - true_box['min_c']) * (true_box['max_r'] - true_box['min_r'])
        pred_area = (pred_box['max_c'] - pred_box['min_c']) * (pred_box['max_r'] - pred_box['min_r'])
        union = true_area + pred_area - intersection
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0
        intersections.append(intersection)
        unions.append(union)

    mean_iou = np.mean([inter / uni for inter, uni in zip(intersections, unions)])
    return mean_iou, [inter / uni if uni > 0 else 0 for inter, uni in zip(intersections, unions)]

# Calculate mean IOU
mean_iou, iou_vector = calculate_iou(true_boxes, predicted_boxes)
print("Mean IOU:", mean_iou)

# Fit Logistic Regression Model
X = np.array(iou_vector).reshape(-1, 1)
y = true_boxes['category'].values

model = LogisticRegression()
model.fit(X, y)

# Compute R-squared
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print("R-squared:", r2)
#3b 
#Testing with a test set ensures the model performs well on new data, 
# not just the training data, helping prevent overfitting 
# and confirming its reliability for future predictions.