from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

model_list_size = {
    "xception": 299,
    "vgg16": 224,
    "vgg19": 224,
    "resnet101": 224,
    "resnet152": 224,
    "resnet50v2": 224,
    "resnet50": 224,
    "resnet152v2": 224,
    "inceptionv3": 299,
    "densenet201": 224,
    "nasnetlarge": 331,
    "inceptionresnetv2": 299,
    "densenet169": 224,
}


model_list_preprocess = {
    "xception": "xception",
    "vgg16": "vgg16",
    "vgg19": "vgg19",
    "resnet101": "resnet",
    "resnet152": "resnet",
    "resnet50v2": "resnet_v2",
    "resnet50": "resnet",
    "resnet152v2": "resnet_v2",
    "inceptionv3": "inception_v3",
    "densenet201": "densenet",
    "nasnetlarge": "nasnet",
    "inceptionresnetv2": "inception_resnet_v2",
    "densenet169": "densenet",
}


class NNModel(str, Enum):
    xception = "Xception"
    vgg16 = "VGG16"
    vgg19 = "VGG19"
    resnet101 = "ResNet101"
    resnet152 = "ResNet152"
    resnet50v2 = "ResNet50V2"
    resnet50 = "ResNet50"
    resnet152v2 = "ResNet152V2"
    inceptionv3 = "InceptionV3"
    densenet201 = "DenseNet201"
    nasnetlarge = "NASNetLarge"
    inceptionresnetv2 = "InceptionResNetV2"
    densenet169 = "DenseNet169"


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
    none = "None"
    imagenet = "imagenet"


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
    none = "None"
    vertical = "vertical"
    horizontal = "horizontal"
    horizontal_and_vertical = "horizontal_and_vertical"


class DataAugmentationParams(BaseModel):
    image_flip: ImageFlip
    batch_size: int = Field(description="batch size")
    rotation_angle: Optional[int] = Field(description="rotation angle", default=None)
    val_pct: Optional[int] = Field(description="validation percentage", default=None)
    shuffle: Optional[bool] = Field(description="shuffle data", default=None)
    seed: Optional[int] = Field(description="random seed", default=42)
    log: Optional[bool] = Field(description="bool flag to log transform the data")


class TrainingParams(DataAugmentationParams):
    weights: Weights
    optimizer: Optimizer
    loss_function: LossFunction
    learning_rate: float = Field(description="learning rate")
    epochs: int = Field(description="number of epochs")
    nn_model: NNModel
