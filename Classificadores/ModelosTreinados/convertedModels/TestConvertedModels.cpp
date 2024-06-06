#include <iostream>
#include <fstream>
#include <vector>
#include <xgboost/c_api.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

// Function to load and preprocess image for EfficientNet model
torch::Tensor preprocess_image(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::resize(image, image, cv::Size(224, 224));
    image.convertTo(image, CV_32F, 1.0 / 255);
    auto tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kFloat);
    tensor = tensor.permute({0, 3, 1, 2});
    tensor = torch::data::transforms::Normalize<>(std::vector<float>{0.485, 0.456, 0.406}, std::vector<float>{0.229, 0.224, 0.225})(tensor);
    return tensor;
}

// Function to load XGBoost model and predict
float predict_xgboost(const std::string& model_path, const std::vector<float>& input) {
    BoosterHandle booster;
    XGBoosterCreate(nullptr, 0, &booster);
    XGBoosterLoadModel(booster, model_path.c_str());

    DMatrixHandle dmatrix;
    XGDMatrixCreateFromMat(input.data(), 1, input.size(), -1, &dmatrix);
    bst_ulong out_len;
    const float* out_result;
    XGBoosterPredict(booster, dmatrix, 0, 0, 0, &out_len, &out_result);
    float prediction = out_result[0];

    XGDMatrixFree(dmatrix);
    XGBoosterFree(booster);
    return prediction;
}

// Function to load EfficientNet model and predict
std::pair<int, int> predict_efficientnet(const std::string& binary_model_path, const std::string& multi_model_path, const std::string& image_path) {
    torch::jit::script::Module binary_model = torch::jit::load(binary_model_path);
    torch::jit::script::Module multi_model = torch::jit::load(multi_model_path);

    binary_model.eval();
    multi_model.eval();

    torch::Tensor image = preprocess_image(image_path);

    // Binary model prediction
    torch::Tensor output_binary = binary_model.forward({image}).toTensor();
    float prediction_binary = torch::sigmoid(output_binary).item<float>();
    int binary_class = prediction_binary >= 0.5 ? 1 : 0;

    // Multi-class model prediction
    torch::Tensor output_multi = multi_model.forward({image}).toTensor();
    int multi_class = output_multi.argmax(1).item<int>();

    return {binary_class, multi_class};
}

int main() {
    // Paths to models and image
    std::string xgboost_binary_model_path = "convertedModels/xgboostBinary_model.json";
    std::string xgboost_multi_model_path = "convertedModels/xgboostMulti_model.json";
    std::string effnet_binary_model_path = "convertedModels/Effnet_Binary_Weights.pt";
    std::string effnet_multi_model_path = "convertedModels/Effnet_Multi_Weights.pt";
    std::string image_path = "5.png";

    // Example input for XGBoost models
    std::vector<float> xgboost_input = {7.248304655095292, 21.17803684411382, 28.20973364173325, 29.135240694328274, -59.46663875649005,
                                        -39.72426859380878, 57.82618002775308, 6.466844614986726, 18.84047735308029, 25.448004897625164,
                                        26.12834139099954, -53.525266476300445, 35.628335712793984, 51.93695413514918, 4.4785120396360245,
                                        13.629832611574178, 15.391241327616664, 15.287393274888451, 32.26078688725662, 22.44070673768876,
                                        30.646120858935387, 7.272455733398132, 21.793772112872936, 28.858041551777507, 29.674772852086072,
                                        -60.84398169444117, -40.57291770094505, 58.95242852493078};

    // XGBoost binary model prediction
    float binary_prediction = predict_xgboost(xgboost_binary_model_path, xgboost_input);
    std::cout << "XGBoost Binary Model Prediction: " << binary_prediction << std::endl;

    // XGBoost multi-class model prediction
    float multi_prediction = predict_xgboost(xgboost_multi_model_path, xgboost_input);
    std::cout << "XGBoost Multi-Class Model Prediction: " << multi_prediction << std::endl;

    // EfficientNet models prediction
    auto [binary_class, multi_class] = predict_efficientnet(effnet_binary_model_path, effnet_multi_model_path, image_path);
    std::cout << "EfficientNet Binary Model Prediction: " << binary_class << std::endl;
    std::cout << "EfficientNet Multi-Class Model Prediction: " << multi_class << std::endl;

    return 0;
}
