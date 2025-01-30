#define USING_OPENCV

#include <Eigen/Dense>
#include <iostream>
#include <iterator>
#include <flare/flare.hpp>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace fl;
using namespace std;
namespace fs = filesystem;

void saveModel(const fl::Sequential& model, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file for saving!" << endl;
        return;
    }

    // Save the number of layers
    file << model.layers.size() << endl;

    // Save each layer
    for (size_t i = 0; i < model.layers.size(); ++i) {
        
        string layer_filename = filename + "_layer_" + to_string(i) + ".dat";
        try{    
            model.layers[i]->Save(layer_filename);
            file << layer_filename << endl; // Save the layer's filename
        } catch(const exception& e) {
            cerr << "Skipping layer " << i << ": " << e.what() << endl;
            file << "non-trainable" << endl;
        }
    }
    file.close();
    cout << "Model saved to " << filename << endl;
}

void loadModel(fl::Sequential& model, const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file for loading!" << endl;
        return;
    }

    size_t num_layers;
    file >> num_layers;

    for (size_t i = 0; i < num_layers; ++i) {
        string layer_filename;
        file >> layer_filename;

        if (layer_filename != "non-trainable") {
            try {
                model.layers[i]->Load(layer_filename);
            } catch (const exception& e) {
                cerr << "Error loading layer " << i << ": " << e.what() << endl;
            }
        }
    }

    file.close();
    cout << "Model loaded from " << filename << endl;
}


bool getBoolFromUser(const string& questionString){
  cout << questionString << endl;

  bool setValue;
  if (cin >> setValue) return setValue;  // Boolean read correctly

  // Badly formed input: failed to read a bool
  cout << "Wrong value. Only 1 or 0 is accepted." << endl;
  cin.clear();                // Clear the failed state of the stream
  cin.ignore(1000000, '\n');  // Extract and discard the bad input

  return getBoolFromUser(questionString);  // Try again
}

bool endsWith(const string& input, const string& query){
    if (input.length() >= query.length()){
        return input.substr(input.length() - query.length()) == query;
    }else{
        return false;
    }
}

void addConvLayer(Sequential& model, int filters, int strides, int kernel_size) {
    model.Add(new Conv2D<ReLU>(
        filters,                  // Number of filters
        3,                        // Input channels (assume RGB images)
        Kernel(kernel_size, kernel_size),
        Padding::PADDING_SAME     // Same padding
    ));

    model.Add(new BatchNormalization<2, 1>(
        Dims<1>(-1)               // Normalize over the channel dimension
    ));

    model.Add(new MaxPooling2D(
        PoolSize(2, 2),           // Pooling size
        Stride(strides, strides), // Strides
        Padding::PADDING_SAME     // Same padding
    ));

    model.Add(new Dropout<4>(0.2)); // Dropout with 0.2 rate
}

int main() {

    cout << "Starting application" << endl;

    cout << "Enter dataset location: ";
    string path;
    cin >> path;

    // check if valid path
    bool ret = endsWith(path, "/");
    if (!ret){
        path.append(string(1, fs::path::preferred_separator));
    }

    Dataset dataset(Dims<3>(64, 64, 3), Dims<1>(10));

    for (int i = 0; i < 10; i++) {
        Tensor<1> label(10);
        label.setZero();
        label(i) = 1;

        for (const auto &entry: fs::directory_iterator(path + to_string(i))) {
            dataset.Add(entry.path(), label);
        }
    }

    dataset.Batch(16, true, false);

    for (auto &image: dataset.training_samples) {
        image = image / 255.0;
    }

    cout << "Total number of batched samples: "
                  << dataset.training_samples.size() << "\n";

    cout << "Defining classifier" << endl; 
    
    
    /*Sequential model {
        new Conv2D<ReLU>(64, 3, Kernel(3, 3), Padding::PADDING_SAME),
        new BatchNormalization<4, 1>(Dims<1>(64)),
??? from here until ???END lines may have been inserted/deleted
        new MaxPooling2D(PoolSize(2, 2), Stride(2, 2), Padding::PADDING_SAME),
        new Dropout<4>(0.2),
        new Flatten<4>(),
        new Dense<Softmax>(64*32*32, 10, false),
    };*/


    // Define the model
    Sequential model;

    // Add convolutional layers
    addConvLayer(model, 48, 2, 5);
    addConvLayer(model, 64, 1, 5);
    addConvLayer(model, 128, 2, 5);
    addConvLayer(model, 160, 1, 5);
    addConvLayer(model, 192, 2, 5);
    addConvLayer(model, 192, 1, 5);
    addConvLayer(model, 192, 2, 5);
    addConvLayer(model, 192, 1, 5);

    // Add dense layers
    model.Add(new Flatten<4>()); // Flatten input
    model.Add(new Dense<Linear>(3072, 3072)); // Fully connected layer
    model.Add(new Dense<Linear>(3072, 3072)); // Another dense layer

    // Final output layer with softmax activation
    model.Add(new Dense<Softmax>(3072, 10));

    // Assuming you have input data `x` with shape (batch_size, 64, 64, 3)
    Tensor<4> x(/* shape= */ 64, 64, 3, /* batch_size= */ 1);
    Tensor<2> y = model.Forward(x); // Perform forward pass

    cout << "Output shape: " << y.Shape() << endl;

    CategoricalCrossEntropy<2> loss;
    Adam opt;

    model.Fit(dataset.training_samples, dataset.training_labels, 15, loss, opt);


    cout << model.Predict<2>(dataset.training_samples.front()) << "\nexpect\n"
                << dataset.training_labels.front() << "\n";
    cout << model.Predict<2>(dataset.training_samples.back()) << "\nexpect\n"
                << dataset.training_labels.back() << "\n\n";


    saveModel(model, "model.flr");
    cout << "saved trained model" << endl;
    
    return 0;
}
???END
