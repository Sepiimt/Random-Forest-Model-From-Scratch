import numpy as np
import datetime as dt
from tree import Node

#> ---------------------------------------------------------------------------------------   
class RandomForest:
    def __init__(self):
        self.tree_list=None #> a list or an array to store Tree Objects.
        self.n_trees=None #> How many trees shall be created from data.
        self.n_features=None #> How many columns did our x array had.
        self.n_samples=None #> How many samples have been provided for training.
        self.is_fitted=False #> Is trained or not.
        self.tree_depth=None #> Each Tree's depth.
        self.trees_node_min_purity=None #> Minimum leaf node purity.
        self.time_taken=None #> Total time taken to train.

    def __str__(self):
        return(f"""\n--- Random Forest Object ---
        Details:
        > Trained Status: {self.is_fitted}
        > Number of Trees: {self.n_trees}
        """)

    def input_validator(self, x_array, y_array, true_y_value=1, false_y_value=0):
        # --- Attempt Conversion to Numpy.Array ---
        if not isinstance(x_array, np.ndarray):
            try:
                x_array = np.asarray(x_array)
            except Exception:
                raise TypeError(f'Could not convert input of type "{type(x_array)}" to "numpy.array".')
        if not isinstance(y_array, np.ndarray):
            try:
                y_array = np.asarray(y_array)
            except Exception:
                raise TypeError(f'Could not convert input of type "{type(y_array)}" to "numpy.array".')    
        # --- Checking the Inputs Logic ---
        rows, columns = np.shape(x_array)
        if (columns>rows):
            x_array=x_array.T
        y_array=y_array.ravel()
        # --- Fixing the Inputs type ---
        if false_y_value!=0 or true_y_value!=1:
            y_array=np.where(y_array == true_y_value, 1, 0)
        # --- return ---
        return x_array, y_array

    def data_bootstrapper(self, x_array, y_array, rng):
        # --- Getting the Shape of X Array ---
        n_rows, n_columns= np.shape(x_array)
        # --- Choosing Random Rows ---
        random_rows = rng.choice(n_rows, size=n_rows)
        # --- Applying the Random Rows Mask ---
        if n_columns!=1:
            x_array_boot=np.array(x_array[random_rows,:])
        else: 
            x_array_boot=np.array(x_array[random_rows])
        y_array_boot=np.array(y_array[random_rows])
        # --- Returning ---
        return x_array_boot, y_array_boot
        
    def fit(self, x_array, y_array, n_trees=5, trees_max_depth=5, min_leaf_purity=0.95, true_y_value=1, false_y_value=0):
        # --- Initelizing Related Tools ---
        rng = np.random.default_rng()
        # --- Validating Input ---
        x_array, y_array=self.input_validator(x_array, y_array, true_y_value, false_y_value)
        # --- Storing Information ---
        self.n_samples, self.n_features = np.shape(x_array)
        self.n_trees=n_trees
        self.tree_depth=trees_max_depth
        self.trees_node_min_purity=min_leaf_purity
        # --- Printing Details ---
        print("-- Random Forest Model Training Live Info --")
        print("> Model Tuning Info:")
        print(f"Forest Model Consists of {self.n_trees} Tree Objects.")
        print(f"Each Tree Object Has {self.tree_depth} Depth.")
        print(f"Model Teaining on {self.n_samples} Samples.")
        print(f"Minimum Purity for each Leaf Node is {self.trees_node_min_purity}")
        print("\n> Live Update:")
        # --- Iterating to create Trees ---
        self.tree_list=[]
        # --- Saving the First Parent Time-Stamp ---
        parent_time_stamp_1=dt.datetime.now()
        for i in range(n_trees):
            # --- Saving the First Time-Stamp ---
            time_stamp_1=dt.datetime.now()
            # --- Creating Bootstraped Data ---
            x_array_boot, y_array_boot=self.data_bootstrapper(x_array, y_array, rng)
            # --- Creating the Tree ---
            Tree=Node()
            Tree.fit(x_array_boot, y_array_boot, trees_max_depth, min_leaf_purity, rng)
            # --- Saving the Tree ---
            self.tree_list.append(Tree)
            # --- Saving the First Time-Stamp ---
            time_stamp_2=dt.datetime.now()
            # --- Printing Details ---
            print(f"{i+1}st Tree Has Been Trained. Time Taken: {time_stamp_2-time_stamp_1}")
        # --- Saving the Second Parent Time-Stamp ---
        parent_time_stamp_2=dt.datetime.now()
        self.time_taken=parent_time_stamp_2-parent_time_stamp_1
        # --- Printing Details ---
        print(f"\nModel has been trained successfully.\nTotal Time Taken: {self.time_taken}")
        # --- Chagning the Flag ---
        self.is_fitted=True

    def predict(self, x_array, detailed=True):
        # --- Quick Validation of Input ---
        n_rows, n_columns = np.shape(x_array)
        if n_columns!=self.n_features:
            raise TypeError(f"Entered X_Array's features does not match the trained.")
        # --- If Input has 1 Row ---
        if n_rows==1:
            # --- If the output should be Detailed ---
            if detailed:
                return self.sub_predict_row(x_array)
            else:
                return np.round(self.sub_predict_row(x_array))
        # --- If Input has n Rows ---
        else:
            # --- If the output should be Detailed ---
            if detailed:
                return self.sub_predict_n_rows(x_array)
            else:
                return np.round(self.sub_predict_n_rows(x_array))

    def sub_predict_row(self, row):
        # --- Defining temp Variable ---
        predicted_y_probabilities = 0
        # --- Iterating over Trees ---
        for Tree in self.tree_list:
            predicted_y_probabilities+=Tree.predict(row)
        # --- Calculating Prediction Probability and Returning---
        return (predicted_y_probabilities/self.n_trees if self.n_trees!=0 else 0)

    def sub_predict_n_rows(self, x_array):
        # --- Defining Saving List ---
        predicted_y_probabilities_array=[]
        # --- Iterating Over Rows ---
        for row in x_array:
            # --- Defining temp Variable ---
            predicted_y_probabilities = 0
            # --- Iterating over Trees ---
            for Tree in self.tree_list:
                predicted_y_probabilities+=Tree.predict(row)
            # --- Calculating the Prediction and Prediction Probability ---
            predicted_y_probabilities_mean = (predicted_y_probabilities/self.n_trees if self.n_trees!=0 else 0)
            predicted_y_probabilities_array.append(predicted_y_probabilities_mean)
        # --- Correcting Return Format ---
        predicted_y_probabilities_array=np.array(predicted_y_probabilities_array)
        return predicted_y_probabilities_array.ravel()
#> ---------------------------------------------------------------------------------------   