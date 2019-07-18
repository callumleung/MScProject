from keras.layers.normalisation import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K

# https://towardsdatascience.com/implementing-a-resnet-model-from-scratch-971be7193718


class ResNet:
    @staticmethod
    # data input to the residual module
    # K number of filters that will be learned by the final Conv layer, first two layers learn K/4
    # stride controls the stride of the convolution, reduces spatial dimension without maz pooling
    # chanDim defines the axis which will perfoirm batch normalisation
    # red (reduce) will control whether we are reducing spatial dimensions(true) or not (false) as
    #    not all residual models will reduce dimensions of our spatial volume
    # applies regularisation strength for all Conv layers in the residual module
    # bnEps controls the epsilon responsible for avoiding division by zero errors when normlising
    # bnMom controls the momentum for the moving average
    def residual_module(data, k, stride, chanDim, red=False, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        # shortcut branch of the ResNet module should be initialised as the input data
        shortcut = data

        # first block of ResNet module are the 1x1 CONVs
        # ResNet module follows BN => ReLu => Conv => pattern
        # Conv layer uses 1x1 convolutions by K/4 filters
        # bias terms not in Conv layer as BN layers already have bias terms
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1,1), use_bias=False, kernel_regularizer=12(reg))(act1)

        # second block of ResNet model are 3x3 Convs
        # k/4 layers
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False,
                       kernel_regularizer=12(reg)(act2))

        # third block of the ResNet module is another set of 1x1 CNVs
        # increases dimensionality again, applying K filters with dimensions 1x1
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=12(reg))(act3)

        # to avoid using max pooling, check if reducing spatial dimensions is necessary
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=12(reg))(act1)

        # If reduction of spatial dimensions is called, a convolutional layer with stride < 1 is applied to the shortcut
        # add the shortcut and the final conv
        x = add([conv3, shortcut])

        # finally add the shortcut and the final conv layer to create the output
        # return as the output of the resnet module
        # will for the building bock for deep residual network
        return x

    # params stages and filters are both lists.
    # stack N residual modules on top of each other.
    # each residual module in the same stage learns te same number of filters.
    # after a layer learns filters, this is followed by dimensionality reduction
    # we repeat this until the average pooling layer and softmax classifier are applied
    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=0.0001, bnEps=2e-5, bnMom=0.9):

        # Initialise inputShape and chanDim based on channels last or channels first
        # initialise the input shape the be "channels last" and the channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # if using channels first update input shape and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # set input and apply BN
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(inputs)

        # ResNet uses BN as the first layer as an added level of normalisation to the input
        # apply Conv => BN => Act => POOL to reduce spatial size
        x = Conv2D(filters[0], (5, 5), use_bias=False, padding="same", kernel_regularizer=12(reg))(x)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # Stack residual layers on top of each other
        # loop over number of stages
        for i in range(0, len(stages)):
            # initialise the stride and apply a residual module used to reduce spatial size of the input volume
            # to reduce volume size without pooling layers change stride of convolution
            # first entry in the stage has stride of (1,1) ie no downsampling
            # every stage after stride is (2, 2) which decreases volume size
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride, chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                x = ResNet.residual_module(x, filters[i + 1], (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

        # avoid dense fully connected layers by applying average pooling to reduce volume size to 1x1xclasses
        # apply BN => Act => POOL
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        # create dense layer for total number of classes to learn then apply
        # softmax activation to generate final output probs
        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=12(reg))(x)
        x = Activation("softmax")(x)

        # create model
        model = Model(inputs, x, name="resnet")

        # return the constructed network architecture
        return model








