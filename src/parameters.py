from enum import Enum
from typing import List, Optional

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


class DataType(str, Enum):
    tiled = "tiled"
    file = "file"


class IOParameters(BaseModel):
    data_uris: List[str] = Field(description="directory containing the data")
    data_type: DataType = Field(description="type of data")
    root_uri: str = Field(description="root URI containing the data")
    uid_save: str = Field(description="uid to save models, metrics and etc")
    uid_retrieve: Optional[str] = Field(
        description="optional, uid to retrieve models for inference"
    )
    data_tiled_api_key: Optional[str] = Field(description="API key for data tiled")
    labels_tiled_uri: Optional[str] = Field(description="labels tiled uri")
    labels_tiled_api_key: Optional[str] = Field(description="labels tiled api key")
    results_tiled_uri: str = Field(description="tiled uri to save results to")
    results_tiled_api_key: Optional[str] = Field(description="tiled api key")
    models_dir: str = Field(description="directory containing the model")
    results_dir: str = Field(description="directory to save the results")
    mask_tiled_uri: Optional[str] = Field(description="mask tiled uri")
    mask_tiled_api_key: Optional[str] = Field(
        description="detector tiled api key", default=None
    )


class PreProcessingParameters(BaseModel):
    image_flip: ImageFlip
    rotation_angle: Optional[int] = Field(description="rotation angle", default=None)
    log: Optional[bool] = Field(description="bool flag to log transform the data")
    low_percentile: Optional[float] = Field(description="low percentile")
    high_percentile: Optional[float] = Field(description="high percentile")


class TrainingParameters(PreProcessingParameters):
    nn_model: NNModel
    weights: Weights
    optimizer: Optimizer
    loss_function: LossFunction
    learning_rate: float = Field(description="learning rate")
    epochs: int = Field(description="number of epochs")
    batch_size: int = Field(description="batch size")
    val_pct: Optional[int] = Field(description="validation percentage", default=None)
    shuffle: Optional[bool] = Field(description="shuffle data", default=None)
    seed: Optional[int] = Field(description="random seed", default=42)


class InferenceParameters(PreProcessingParameters):
    batch_size: int = Field(description="batch size")
