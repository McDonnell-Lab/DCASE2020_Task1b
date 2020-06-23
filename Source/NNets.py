#Author: Mark McDonnell, mark.mcdonnell@unisa.edu.au
import tensorflow.keras
from tensorflow.keras.layers import Input, ReLU,Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, AveragePooling2D, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from binary_layers_tf import BinaryConv2D

#network definition
def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,learn_bn = True,wd=1e-4,use_relu=True,binarise_weights=False):

    x = inputs
    x = BatchNormalization(center=learn_bn, scale=learn_bn,epsilon=1e-5)(x)
    if use_relu:
        x = Activation('relu')(x)
    if binarise_weights == False:
        x = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer='he_normal',
                  kernel_regularizer=l2(wd),use_bias=False)(x)
    else:
        x = BinaryConv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer='he_normal',
                  kernel_regularizer=l2(wd),use_bias=False)(x)
    return x


def Freq_split(x,chans):
    return x[:,chans[0]:chans[1],:,:] 


def model_resnet_DCASE2020_Task1b(num_classes,input_shape =[128,None,6], num_filters =24,wd=1e-3,binarise_weights=False,NotHybrid=True):
    
    My_wd = wd 
    num_res_blocks=2
    
    inputs = Input(shape=input_shape)
    
    
    #split up frequency into two branches
    Split1=  Lambda(Freq_split,arguments={'chans':[0,int(input_shape[0]/2)]})(inputs)
    Split2=  Lambda(Freq_split,arguments={'chans':[int(input_shape[0]/2),input_shape[0]]})(inputs)

    ResidualPath1 = resnet_layer(inputs=Split1,
                     num_filters=num_filters,
                     strides=[1,2],
                     learn_bn = True,
                     wd=My_wd,
                     use_relu = False,
                     binarise_weights=NotHybrid)
    
    ResidualPath2 = resnet_layer(inputs=Split2,
                     num_filters=num_filters,
                     strides=[1,2],
                     learn_bn = True,
                     wd=My_wd,
                     use_relu = False,
                     binarise_weights=NotHybrid)

    # Instantiate the stack of residual units
    for stack in range(4):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = [1,2]  # downsample
            ConvPath1 = resnet_layer(inputs=ResidualPath1,
                             num_filters=num_filters,
                             strides=strides,
                             learn_bn = False,
                             wd=My_wd,
                             use_relu = True,
                             binarise_weights=binarise_weights)
            ConvPath2 = resnet_layer(inputs=ResidualPath2,
                             num_filters=num_filters,
                             strides=strides,
                             learn_bn = False,
                             wd=My_wd,
                             use_relu = True,
                             binarise_weights=binarise_weights)
            ConvPath1 = resnet_layer(inputs=ConvPath1,
                             num_filters=num_filters,
                             strides=1,
                             learn_bn = False,
                             wd=My_wd,
                             use_relu = True,
                             binarise_weights=binarise_weights)
            ConvPath2 = resnet_layer(inputs=ConvPath2,
                             num_filters=num_filters,
                             strides=1,
                             learn_bn = False,
                             wd=My_wd,
                             use_relu = True,
                             binarise_weights=binarise_weights)
            if stack > 0 and res_block == 0:  
                # first layer but not first stack: this is where we have gone up in channels and down in feature map size
                #so need to account for this in the residual path
                
                #average pool and downsample the residual path then zero-pad channels
                ResidualPath1 = AveragePooling2D(pool_size=(3, 3), strides=[1,2], padding='same')(ResidualPath1)
                ResidualPath2 = AveragePooling2D(pool_size=(3, 3), strides=[1,2], padding='same')(ResidualPath2)
                ResidualPath1 = tensorflow.keras.layers.concatenate([ResidualPath1,Lambda(K.zeros_like)(ResidualPath1)])
                ResidualPath2 = tensorflow.keras.layers.concatenate([ResidualPath2,Lambda(K.zeros_like)(ResidualPath2)])
               
            ResidualPath1 = tensorflow.keras.layers.add([ConvPath1,ResidualPath1])
            ResidualPath2 = tensorflow.keras.layers.add([ConvPath2,ResidualPath2])
            
        #double the number of filters   
        if stack <=2:
            num_filters *= 2
        

    ResidualPath = tensorflow.keras.layers.add([ResidualPath1,ResidualPath2])#tensorflow.keras.layers.concatenate([ResidualPath1,ResidualPath2],axis=1)
    
    OutputPath = resnet_layer(inputs=ResidualPath,
                             num_filters=192,
                              kernel_size=3,
                             strides=1,
                             learn_bn = False,
                             wd=My_wd,
                             use_relu = True,
                             binarise_weights=binarise_weights)
        
    #output layers after last sum
    OutputPath = resnet_layer(inputs=OutputPath,
                     num_filters=num_classes,
                     strides = 1,
                     kernel_size=1,
                     learn_bn = False,
                     wd=My_wd,
                     use_relu=True,
                     binarise_weights=binarise_weights)
    OutputPath = BatchNormalization(center=True, scale=True,epsilon=1e-5)(OutputPath)
    OutputPath = GlobalAveragePooling2D()(OutputPath)
    OutputPath = Activation('softmax')(OutputPath)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=OutputPath)
    return model


