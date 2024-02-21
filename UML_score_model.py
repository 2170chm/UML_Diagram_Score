import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer

labels_dir = 'UML_Data/UML_YOLOv8/train/labels'
ratings_file = 'UML_Data/rating_images.csv'

def calculate_pairwise_distances(centers):
    num_centers = len(centers)
    distances = np.zeros((num_centers, num_centers))

    for i in range(num_centers):
        for j in range(num_centers):
            diff = centers[i] - centers[j]  # Difference in x and y coordinates
            distances[i, j] = np.sqrt(np.sum(diff ** 2))  # Euclidean distance

    return distances

def calculate_additional_features(labels):
        labels['class_id'] = labels['class_id'].astype(int)
        labels['area'] = labels['width'] * labels['height']
        labels['aspect_ratio'] = labels['width'] / labels['height']

        centers = labels[['x_center', 'y_center']].values
        distances = calculate_pairwise_distances(centers)

        class_counts = labels['class_id'].value_counts().to_dict()


        min_width = labels['width'].min()
        max_width = labels['width'].max()
        min_height = labels['height'].min()
        max_height = labels['height'].max()

        labels['min_edge_dist_x'] = labels['x_center'].apply(lambda x: min(x, 1 - x))
        labels['min_edge_dist_y'] = labels['y_center'].apply(lambda y: min(y, 1 - y))
        min_edge_distance = labels[['min_edge_dist_x', 'min_edge_dist_y']].min(axis=1).mean()

        total_area = labels['area'].sum()
        density = total_area / 1

        avg_x_center = labels['x_center'].mean()
        avg_y_center = labels['y_center'].mean()
        
        labels['xmin'] = labels['x_center'] - labels['width']/2
        labels['ymin'] = labels['y_center'] - labels['height']/2
        labels['xmax'] = labels['x_center'] + labels['width']/2
        labels['ymax'] = labels['y_center'] + labels['height']/2
        
        x_extremes = (labels['xmin'].min(), labels['xmax'].max())
        y_extremes = (labels['ymin'].min(), labels['ymax'].max())

        return labels, distances, class_counts, min_width, max_width, \
        min_height, max_height, min_edge_distance, density, avg_x_center, \
        avg_y_center, x_extremes, y_extremes
   
def process_dataset(labels_dir, ratings_file):
    ratings_df = pd.read_csv(ratings_file, header=None, names=['index', 'image_name', 'rating'], skiprows=1)
    ratings_df['image_id'] = ratings_df['image_name'].str.split('.').str[0].astype(int)
    ratings_df.drop('image_name', axis=1, inplace=True)

    data = []

    for filename in os.listdir(labels_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(labels_dir, filename)
            
            with open(file_path, 'r') as file:
                labels = [line.strip().split() for line in file.readlines()]
                labels = pd.DataFrame(labels, dtype=float)
                labels.columns = ['class_id', 'x_center', 'y_center', 'width', 'height']

                num_crosses = (labels['class_id'] == 2).sum()

                labels = labels[labels['class_id'] != 2]

                if labels.empty:
                    continue
                
                labels, distances, class_counts, min_width, max_width, min_height, \
                max_height, min_edge_distance, density, avg_x_center, avg_y_center, \
                x_extremes, y_extremes = calculate_additional_features(labels)

                image_id = int(filename.split('_')[0])

                features = {
                    'image_id': image_id,
                    'num_objects': len(labels),
                    'num_crosses': num_crosses,
                    'mean_width': labels['width'].mean(),
                    'mean_height': labels['height'].mean(),
                    'std_width': labels['width'].std(),
                    'std_height': labels['height'].std(),
                    'mean_area': labels['area'].mean(),
                    'mean_aspect_ratio': labels['aspect_ratio'].mean(),
                    'std_area': labels['area'].std(),
                    'std_aspect_ratio': labels['aspect_ratio'].std(),
                    'mean_distance': np.mean(distances),
                    'std_distance': np.std(distances),
                    'min_width': min_width,
                    'max_width': max_width,
                    'min_height': min_height,
                    'max_height': max_height,
                    'min_edge_distance': min_edge_distance,
                    'xmin_extreme': x_extremes[0],
                    'xmax_extreme': x_extremes[1],
                    'ymin_extreme': y_extremes[0],
                    'ymax_extreme': y_extremes[1],
                    'density': density,
                    'avg_x_center': avg_x_center,
                    'avg_y_center': avg_y_center,
                }
                
                for cls, count in class_counts.items():
                    features[f'class_{cls}_count'] = count

                data.append(features)

    features_df = pd.DataFrame(data)

    final_df = pd.merge(features_df, ratings_df, on='image_id')
    final_df.drop('index', axis=1, inplace=True)
    
    return final_df


def get_accuracy_f1_per_class(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    return accuracy, f1_per_class

train_df = process_dataset('UML_Data/UML_YOLOv8/train/labels', 'UML_Data/rating_images.csv')
test_df = process_dataset('UML_Data/UML_YOLOv8/test/labels', 'UML_Data/rating_images.csv')

X_train = train_df.drop(['image_id', 'rating'], axis=1)
y_train = train_df['rating']

X_test = test_df.drop(['image_id', 'rating'], axis=1)
y_test = test_df['rating']

imputer = SimpleImputer(strategy='median')

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

selector = SelectFromModel(RandomForestClassifier())
X_train_selected = selector.fit_transform(X_train_imputed, y_train)
X_test_selected = selector.transform(X_test_imputed)

param_grid = {
    'n_estimators': [10, 50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    }
    
rf_clf = RandomForestClassifier()
grid_search = GridSearchCV(rf_clf, param_grid, cv=10, scoring='accuracy', return_train_score=True)
grid_search.fit(X_train_selected, y_train)

best_model = grid_search.best_estimator_
best_model.fit(X_train_selected, y_train)

test_accuracy, test_f1_per_class = get_accuracy_f1_per_class(best_model, X_test_selected, y_test)

print(f'Test Accuracy: {test_accuracy}')
print(f'Test F1 Scores for Each Class: {test_f1_per_class}')