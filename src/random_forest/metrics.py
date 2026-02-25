import numpy as np
import os

#> ---------------------------------------------------------------------------------------
def Cross_Entropy(predicted_y_prob, y_array, eps=1e-12):
    #> Note: Penalizes confident wrong predictions brutally & Rewards calibrated probabilities
    p = np.clip(predicted_y_prob, eps, 1 - eps)
    y = y_array
    # --- Return ---
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    
#> ---------------------------------------------------------------------------------------
def Confusion_Matrix(predicted_y, y_array):
    #> Note: The raw reality of our model’s decisions
    # --- Creating a Parallel Y Array ---
    y_pred = np.round(predicted_y).astype(int)
    y_true = y_array.astype(int)
    # --- Calculating TP,FP,TN,FN ---
    #> Note: Format: (True Positive, False Positive, True Negetive, False Negetive)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    # --- Return ---
    return np.array([[TP, FP],
                     [TN, FN]])

#> ---------------------------------------------------------------------------------------
def Accuracy(predicted_y, y_array):
    #> Note: Fraction of correct predictions
    y_pred = np.round(predicted_y)
    return np.mean(y_pred == y_array)

#> ---------------------------------------------------------------------------------------
def Precision(predicted_y, y_array):
    #> Note: Of the predicted positives, how many were correct?
    y_pred = np.round(predicted_y)
    # --- Process ---
    TP = np.sum((y_pred == 1) & (y_array == 1))
    FP = np.sum((y_pred == 1) & (y_array == 0))
    # --- Return ---
    return TP / (TP + FP) if (TP + FP) != 0 else 0

#> ---------------------------------------------------------------------------------------
def Recall(predicted_y, y_array):
    #> Note: Of the true positives, how many did we catch?
    y_pred = np.round(predicted_y)
    # --- Process ---
    TP = np.sum((y_pred == 1) & (y_array == 1))
    FN = np.sum((y_pred == 0) & (y_array == 1))
    # --- Return ---
    return TP / (TP + FN) if (TP + FN) != 0 else 0

#> ---------------------------------------------------------------------------------------
def F1_Score(predicted_y, y_array):
    #> Note: Harmonic mean of precision and recall
    precision=Precision(predicted_y, y_array)
    recall=Recall(predicted_y, y_array)
    # --- If ---
    if (precision + recall)!=0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else: 
        f1_score=0
    # --- Return ---
    return f1_score

#> ---------------------------------------------------------------------------------------
def Brier_Score(predicted_y, y_array):
    #> Note: Mean squared error of probabilities vs truth
    return (1/len(y_array))*(((y_array-predicted_y)**2).sum())

#> ---------------------------------------------------------------------------------------
def Model_Evaluation(predicted_y, y_test, save=False, RFModel=None, text_file_name="metrics.txt", default_path=r"..\results\metrics"):
    # --- Getting Information ---
    r1, r2, r3, = Cross_Entropy(predicted_y, y_test), Confusion_Matrix(predicted_y, y_test), Accuracy(predicted_y, y_test)
    r4, r5, r6 = Precision(predicted_y, y_test), Recall(predicted_y, y_test), F1_Score(predicted_y, y_test)
    r7 = Brier_Score(predicted_y, y_test)
    # --- Writing Information ---
    r1=f"Cross_Entropy: {r1}"
    r2=f"Confusion Matrix: \n{r2}\n"
    r3=f"Accuracy: {r3}"
    r4=f"Precision: {r4}"
    r5=f"Recall: {r5}"
    r6=f"F1_Score: {r6}"
    r7=f"Brier_Score: {r7}"
    if RFModel != None:
        mr1, mr2, mr3, mr4, mr5 = RFModel.n_trees, RFModel.tree_depth , RFModel.trees_node_min_purity, RFModel.n_samples, RFModel.time_taken
        mr1=f"Total Tree Objects in the Forest: {mr1}"
        mr2=f"Depth of Each Tree Object: {mr2}"
        mr3=f"Minimum Purity of Each Leaf Node in Tree Objects: {mr3}"
        mr4=f"Model Training on {mr4} Samples."
        mr5=f"Total Time Taken for Training the Model: {mr5}"
    # --- Repersenting Information ---
    print(r1)
    print(r2)
    print(r3)
    print(r4)
    print(r5)
    print(r6)
    print(r7)
    # --- Saving Metrics ---
    if save:
        path = os.path.join(default_path, text_file_name)
        tree_details = [mr1, mr2, mr3, mr4, mr5]
        details = [r1,r2,r3,r4,r5,r6,r7]
        with open(path, "w") as f:
            f.write("--- Model Info ---\n")
            if RFModel != None:
                for item in tree_details:
                    f.write("> " + item + "\n")
            f.write("\n--- Model Metrics ---\n")
            for item in details:
                f.write("> " + item + "\n")

#> ---------------------------------------------------------------------------------------
