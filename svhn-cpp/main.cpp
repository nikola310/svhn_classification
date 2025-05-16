#define USING_OPENCV

#include <Eigen/Dense>
#include <iostream>
#include <iterator>
#include <unistd.h>
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
    } else {
        return false;
    }
}


int main() {

    cout << "Starting application" << endl;

    // Define a buffer 
    const size_t size = 1024; 
    // Allocate a character array to store the directory path
    char cwd[size];        
    fs::path path;
    // Call getcwd to get the current working directory and store it in buffer
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        path = fs::path(cwd);

        cout << path << endl;
        path = path.parent_path().parent_path();

        path /= "resized-data";
        path /= "train";

        cout << "Dataset path is " << path << endl;
    } else {
        perror("getcwd() error");
        return 1;
    }

    Dataset dataset(Dims<3>(32, 32, 3), Dims<1>(10));

    for (int i = 0; i < 10; i++) {
        Tensor<1> label(10);
        label.setZero();
        label(i) = 1;

        for (const auto &entry: fs::directory_iterator(path / to_string(i))) {
            dataset.Add(entry.path(), label);
        }
    }

    dataset.Batch(16, true, false);

    for (auto &image: dataset.training_samples) {
        image = image / 255.0;
    }

    cout << "Total number of batched samples: " << dataset.training_samples.size() << "\n";

    cout << "Defining classifier" << endl; 

    // Define the model
    Sequential model {
        // Conv Block 1
        new Conv2D<ReLU>(48, 3, Kernel(5, 5), Padding::PADDING_SAME),
        new MaxPooling2D(PoolSize(2, 2), Padding::PADDING_SAME),
        new Dropout<4>(0.2),

        // Conv Block 2
        new Conv2D<ReLU>(64, 48, Kernel(5, 5), Padding::PADDING_SAME),
        new MaxPooling2D(PoolSize(2, 2), Padding::PADDING_SAME),
        new Dropout<4>(0.2),

        // Conv Block 3
        new Conv2D<ReLU>(128, 64, Kernel(5, 5), Padding::PADDING_SAME),
        new MaxPooling2D(PoolSize(2, 2), Padding::PADDING_SAME),
        new Dropout<4>(0.2),

        // Conv Block 4
        new Conv2D<ReLU>(160, 128, Kernel(5, 5), Padding::PADDING_SAME),
        new MaxPooling2D(PoolSize(2, 2), Padding::PADDING_SAME),
        new Dropout<4>(0.2),

        // Conv Block 5
        new Conv2D<ReLU>(192, 160, Kernel(5, 5), Padding::PADDING_SAME),
        new MaxPooling2D(PoolSize(2, 2), Padding::PADDING_SAME),
        new Dropout<4>(0.2),

        // Conv Block 6
        new Conv2D<ReLU>(192, 192, Kernel(5, 5), Padding::PADDING_SAME),
        new MaxPooling2D(PoolSize(2, 2), Padding::PADDING_SAME),
        new Dropout<4>(0.2),

        // Conv Block 7
        new Conv2D<ReLU>(192, 192, Kernel(5, 5), Padding::PADDING_SAME),
        new MaxPooling2D(PoolSize(2, 2), Padding::PADDING_SAME),
        new Dropout<4>(0.2),

        // Conv Block 8
        new Conv2D<ReLU>(192, 192, Kernel(5, 5), Padding::PADDING_SAME),
        new MaxPooling2D(PoolSize(2, 2), Padding::PADDING_SAME),
        new Dropout<4>(0.2),

        // Dense Layers
        new Flatten<4>(),
        new Dense<ReLU>(192, 3072, false),
        new Dense<ReLU>(3072, 3072, false),
        new Dense<Softmax>(3072, 10, false)
    };
    cout << "Classifier defined" << endl;

    CategoricalCrossEntropy<2> loss;
    Adam opt;
    cout << "Training started" << endl;
    model.Fit(dataset.training_samples, dataset.training_labels, 15, loss, opt);
    cout << "Training over!" << endl;
    cout << model.Predict<2>(dataset.training_samples.front()) << "\nexpect\n"
                << dataset.training_labels.front() << "\n\n";
    cout << model.Predict<2>(dataset.training_samples.back()) << "\nexpect\n"
                << dataset.training_labels.back() << "\n\n";

    saveModel(model, "classifier.flr");
    cout << "Training model saved!" << endl;
    cout << "Program finished" << endl;
    return 0;
}
