# Cells Recognition in Pap Smear Exams

## Project Overview

This project implements image analysis and processing techniques to classify cells in Pap smear exams. The application provides an intuitive graphical interface allowing users to load images of cell clusters obtained from Pap smear exams for individual cell analysis.

### Key Features
- **Image Loading**: Users can load images of cell clusters for analysis.
- **Histogram Generation**: The application generates histograms to analyze the tonal distribution within the images.
- **Shape Feature Calculation**: Hu's invariant moments are calculated to capture shape features invariant to image transformations.
- **Co-occurrence Matrices**: Spatial relationships between pixels are analyzed using co-occurrence matrices.

### Classification Techniques
The application utilizes both shallow and deep learning models for cell classification:
- **Shallow Models**: Based on XGBoost from the Scikit-Learn library.
- **Deep Learning Models**: Pre-trained EfficientNet models from Pytorch, fine-tuned for this application.

### Dataset
The dataset used for training the models consists of 100x100 images of cells extracted from Pap smear images, with labels provided.

### Results
- **XGBoost Multiclass**: 95.16% accuracy.
- **EfficientNet Multiclass**: 80.64% accuracy.
- **Binary Classification**: XGBoost: 84.09%, EfficientNet: 88.82%.

### Conclusion
The classifiers show strong performance in identifying severe cases like ASC-H, HSIL, and SCC, but there is room for improvement in classifying less severe abnormalities and benign lesions.

## Interface
![interface](https://github.com/Andreixzc/Pap-smear-CellClassification/assets/90361670/239637d4-6a65-474d-9e00-a8ef6842d894)

## Authors
- Andrei Gonçalves Rohlfs Massaini
- João Gabriel Polonio Teixeira
- João Pedro Torres de Souza Silva

## References
- Bradski, G. (2000). The opencv library.
- Chen, T. and Guestrin, C. (2016). Xgboost: A scalable tree boosting system.
- Harris, C. R., et al. (2020). Array programming with numpy.
- Hunter, J. D. (2007). Matplotlib: A 2d graphics environment.
- Lemaître, G., et al. (2017). Imbalanced-learn: A python toolbox to tackle the curse of imbalanced datasets in machine learning.
- Limited, R. C. (2021). Pyqt6 documentation.
- McKinney, W. (2010). Data structures for statistical computing in python.
- Paszke, A., et al. (2019). Pytorch: An imperative style, high-performance deep learning library.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in python.
- Rezende, M., et al. (2021). Cric searchable image database as a public platform for conventional pap smear cytology data.
