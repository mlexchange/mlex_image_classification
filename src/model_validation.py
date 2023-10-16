from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List


model_list = {
    'xception' : 299,
    'vgg16' : 224,
    'vgg19' : 224,
    'resnet101' : 224,
    'resnet152' : 224,
    'resnet50v2' : 224,
    'resnet50': 224,
    'resnet152v2' : 224,
    'inceptionv3' : 299,
    'densenet201' : 224,
    'nasnetlarge' : 331,
    'inceptionresnetv2' : 299,
    'densenet169' : 224,
}


class NNModel(str, Enum):
    xception = 'Xception'
    vgg16 = 'VGG16'
    vgg19 = 'VGG19'
    resnet101 = 'ResNet101'
    resnet152 = 'ResNet152'
    resnet50v2 = 'ResNet50V2'
    resnet50 = 'ResNet50'
    resnet152v2 = 'ResNet152V2'
    inceptionv3 = 'InceptionV3'
    densenet201 = 'DenseNet201'
    nasnetlarge = 'NASNetLarge'
    inceptionresnetv2 = 'InceptionResNetV2'
    densenet169 = 'DenseNet169'


class Optimizer(str, Enum):
    Adadelta = "Adadelta"
    Adagrad = "Adagrad"
    Adam = "Adam"
    Adamax = "Adamax"
    Ftrl = "Ftrl"
    Nadam = "Nadam"
    RMSprop = "RMSprop"
    SGD = "SGD"


class Weights(str, Enum):
    none = 'None'
    imagenet = 'imagenet'


class LossFunction(str, Enum):
    binary_crossentropy = "binary_crossentropy"
    binary_focal_crossentropy = "binary_focal_crossentropy"
    categorical_crossentropy = "categorical_crossentropy"
    categorical_hinge = "categorical_hinge"
    cosine_similarity = "cosine_similarity"
    hinge = "hinge"
    huber = "huber"
    log_cosh = "log_cosh"
    kullback_leibler_divergence = "kullback_leibler_divergence"
    mean_absolute_error = "mean_absolute_error"
    mean_absolute_percentage_error = "mean_absolute_percentage_error"
    mean_squared_error = "mean_squared_error"
    mean_squared_logarithmic_error = "mean_squared_logarithmic_error"
    poisson = "poisson"
    sparse_categorical_crossentropy = "sparse_categorical_crossentropy"
    squared_hinge = "squared_hinge"


class ImageFlip(str, Enum):
    none = 'None'
    vertical = 'vertical'
    horizontal = 'horizontal'
    horizontal_and_vertical = 'horizontal_and_vertical'


class DataAugmentationParams(BaseModel):
    image_flip: ImageFlip
    batch_size: int = Field(description='batch size')
    rotation_angle: Optional[int] = Field(description='rotation angle')
    val_pct: Optional[int] = Field(description='validation percentage')
    shuffle: Optional[bool] = Field(description='shuffle data')
    seed: Optional[int] = Field(description='random seed', default=42)


class TrainingParams(DataAugmentationParams):
    weights: Weights
    optimizer: Optimizer
    loss_function: LossFunction
    learning_rate: float = Field(description='learning rate')
    epochs: int = Field(description='number of epochs')
    nn_model: NNModel


class TransferLearningParams(DataAugmentationParams):
    epochs: int = Field(description='number of epochs')
    init_layer: int = Field(description='initial layer')
    nn_model: Optional[NNModel]