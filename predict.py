"""Predict plant types from plant common name."""
import csv
import itertools
import numpy
from sklearn.cross_validation import train_test_split
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
import tflearn


# Build an array of unique words from the data set
# unique_words: Array of unique words for all common names
# data: Array of raw plant data
# col_index_common_name: Column index of the common name in the plant data
#
# returns - Array: len(unique_words) X len(data)
def build_name_words_array(unique_words, data, col_index_common_name):
    """Build an array of activated unique words."""
    data_words = numpy.zeros([len(data), len(unique_words)])
    for row_idx, row in enumerate(data):
        plant_common_name_words = row[col_index_common_name].split()
        for word in plant_common_name_words:
            data_words[row_idx][unique_words.index(word)] = 1
    return data_words


# Mapping for ID to Plant Type Name
plant_type_names = {
 1: "Perennial", 2: "Rhododendron", 3: "Shrub", 5: "Groundcover", 6: "Annual",
 7: "Tree", 8: "Ornamental Grass", 9: "Cactus/Succulent",
 10: "Vine - Requires Support", 11: "Camellia", 12: "Conifer", 13: "Magnolia",
 15: "Bamboo", 16: "Fern", 17: "Peony", 18: "Palm",
 19: "Vine - Self-clinging", 20: "Citrus"}

# Read in plant data from CSV
data = None
header_row = None
with open('./plants.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
    header_row = data.pop(0)

col_index_common_name = header_row.index('Common Name')
common_names = [d[col_index_common_name] for d in data]
common_name_words = [n.split() for n in common_names]
common_name_words = list(itertools.chain(*common_name_words))
common_name_words_unique = list(set(common_name_words))
common_name_words_unique.sort()

# Find all plants that include a Plant Type
col_index_plant_type_id = header_row.index('Plant Type ID')
data_with_plant_type = [d for d in data if len(d[col_index_plant_type_id]) > 0]

# Build a unique array of  all of the words used in the Common Name field
# len(common_name_words_unique) X len(data_with_plant_type)
data_words = build_name_words_array(
    common_name_words_unique, data_with_plant_type, col_index_common_name)

# Build the output array of plant types
plant_type_ids = list(
    set([int(d[col_index_plant_type_id]) for d in data_with_plant_type]))
data_plant_types = numpy.zeros(
    [len(data_with_plant_type), len(plant_type_ids)])
for row_idx, row in enumerate(data_with_plant_type):
    data_plant_types[row_idx][
        plant_type_ids.index(int(row[col_index_plant_type_id]))] = 1

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data_words, data_plant_types, test_size=0.33, random_state=42)

# Build a neural network
network = input_data(shape=[None, len(data_words[0])])
network = fully_connected(network, 2048, activation='relu')
network = fully_connected(
    network, len(data_plant_types[0]), activation='softmax')

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0003)
model = tflearn.DNN(network, tensorboard_verbose=0)

# Train the network
model.fit(X_train, y_train, n_epoch=10, shuffle=True,
          validation_set=(X_test, y_test),
          show_metric=True, batch_size=25, run_id='specific_cnn')

# Find all plants that do not include a Plant Type
# Build up a words array in the same format as our training set
col_index_plant_type_id = header_row.index('Plant Type ID')
data_without_plant_type = [
    d for d in data if len(d[col_index_plant_type_id]) == 0]

data_words = build_name_words_array(common_name_words_unique,
                                    data_without_plant_type,
                                    col_index_common_name)

# Predict plant type
predictions = model.predict(data_words)

# Write predictions to CSV
prediction_ids = [
    plant_type_ids[
        prediction.index(max(prediction))] for prediction in predictions]
prediction_names = [
    plant_type_names[plant_type_id] for plant_type_id in prediction_ids]
with open('./predictions.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    for row_idx, row in enumerate(data_without_plant_type):
        writer.writerow(
            [row[col_index_common_name],
             prediction_names[row_idx],
             prediction_ids[row_idx]])
