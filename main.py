from tensorflow import keras
from raw_data_generation_and_processing import read_all_data, data_split
from model import input_generatation, densenet_model, densenet_train, densenet_test, plot_train_valid, cnn_test
# ======================================================================================================================
# Data preprocessing If you don't wanna use generated image uncomment this section to get image from wav.file
# path = './raw_dataset/'
# seg_sigs, seg_labels = read_all_data(path)
# X_train, y_train, X_valid, y_valid, X_test, y_test = data_split(seg_sigs, seg_labels)
# Trainpath = './Raw_Image/Train'
# plot_raw_image(X_train, y_train, Trainpath)
# Validpath = './Raw_Image/Validation'
# plot_raw_image(X_valid, y_valid, Validpath)
# Testpath = './Raw_Image/Test'
# plot_raw_image(X_test, y_test, Testpath)
# ======================================================================================================================
# For Densenet Based CNN, model needs to be trained, no saved model since too large

path  = './STFT/' # Specify the image data path
train_generator, valid_generator, test_generator = input_generatation(path) # Generate compatible input for TF CNN
model = densenet_model() # Use pretrained model 
model.summary()
history = densenet_train(model, train_generator, valid_generator) #Train densenet 
plot_train_valid(history, './Figure/Densenet_STFT_train_valid_curve.png') # Plot train and valid curve
densenet_test(model, test_generator, './Figure/Densenet_STFT_Test_Heatmap.png') #Get testset metrics

# ======================================================================================================================
# For Custom Based CNN, model is loaded to directly evaluate test data
model_path = './Result/model/cnn_stft_Raw_densenet.h5'
model = keras.models.load_model(model_path)
path = './STFT/'
figpath = './Result/Figure/cnn_stft_test_metrics.png'
cnn_test(model, path, figpath)