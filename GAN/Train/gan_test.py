
#MUST BE FIRST IMPORT
from DeepJetCore.training.training_base import training_base

import keras
from keras.models import Model
from keras.layers import RepeatVector,GaussianNoise,LeakyReLU, Dense, Conv2D,Conv1D, Flatten, BatchNormalization, Reshape, Concatenate #etc


    

from DeepJetCore.Layers import ScalarMultiply, Print, ReplaceByNoise, FeedForward, ReduceSumEntirely
def d_model(Inputs):
    
    x = Inputs[0] 
    x_sum = ReduceSumEntirely()(x)
    x1 = Inputs[1] #coordinates
    x2 = Inputs[2] #rest global
    x2 = RepeatVector(int(x.shape[1])*int(x.shape[1]))(x2)
    x2 = Reshape((int(x.shape[1]),int(x.shape[2]),int(x2.shape[-1])))(x2)
    
    x = Concatenate()([x,x1])
    x = Concatenate()([x,x2])
    x = Conv2D(16,(1,1),padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(16,(1,1),padding='same')(x)
    
    #x = Conv2D(16,(8,8),padding='valid')(x)
    #x = LeakyReLU()(x)
    #x = Conv2D(16,(8,8),padding='valid')(x)
    #x = LeakyReLU()(x)
    #x = Conv2D(16,(4,4),padding='valid')(x)
    #x = LeakyReLU()(x)
    x = Conv2D(1, (4,4),padding='valid')(x)
    x = LeakyReLU()(x)
    #x_noise = Inputs[1] 
    #only_for_gan=Inputs[2]
    
    x = Flatten()(x)
    x = Concatenate()([x,x_sum])
    x = Dense(16,activation='relu')(x) #discriminator
    x = Dense(1,activation='sigmoid')(x) #discriminator
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions,name='discriminator')

def g_model(Inputs):
        
    feed_forward = [FeedForward()(Inputs[1]),FeedForward()(Inputs[2])]    
    
    x = Inputs[0] 
    x = ReplaceByNoise()(x)
    x1 = Inputs[1] #coordinates
    x = Concatenate()([x1,x])
    
    x2 = Inputs[2] #rest global
    x2 = RepeatVector(int(x.shape[1])*int(x.shape[1]))(x2)
    x2 = Reshape((int(x.shape[1]),int(x.shape[2]),int(x2.shape[-1])))(x2)
    x2 = ScalarMultiply(1./(24.*24.))(x2)
    
    x = Concatenate()([x2,x])
    x = Conv2D(16,(1,1),padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(16,(1,1),padding='same')(x)
    x = LeakyReLU()(x)
    #x = Conv2D(16,(8,8),padding='same')(x)
    #x = LeakyReLU()(x)
    #x = Conv2D(16,(8,8),padding='same')(x)
    #x = LeakyReLU()(x)
    x = Conv2D(1,(4,4), padding='same')(x)
    x = LeakyReLU()(x)
    
    #use multiply with scalaer to avoid fed and fetched problem
    return Model(inputs=Inputs, outputs=[x]+feed_forward,name='generator')


train=training_base(testrun=False,testrun_fraction=0.05, resumeSilently=False,renewtokens=True)

train.setGANModel(g_model, d_model)
    
    #for regression use a different loss, e.g. mean_squared_error instead of categorical_crossentropy
#add some loss scaling factors here
train.compileModel(learningrate=0.0003, print_models=True,
                   discr_loss_weights=None,#[2.],
                   gan_loss_weights=None)#[1.]) 

train.trainGAN_exp(nepochs=1,
                   batchsize=500,
                   verbose=1,
                   gan_skipping_factor=1,
                   discr_skipping_factor=1)

train.change_learning_rate(0.0000001)

train.trainGAN_exp(nepochs=1,
                   batchsize=150,
                   verbose=1,
                   gan_skipping_factor=1,
                   discr_skipping_factor=1)



#make some plot after the training (really advisable to come up with metrics that can be 
#monitored during the training) or with callbacks that make plots (still under development for GANs)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotgrid(in_array, nplotsx, nplotsy, outname):
    fig,ax=plt.subplots(nplotsy,nplotsx)
    counter=0
    for i in range(nplotsy):
        for j in range(nplotsx):
            ax[i][j].imshow(in_array[counter,:,:,0])
            counter+=1
    fig.savefig(outname)
    plt.close()

import numpy as np
from DeepJetCore.TrainData import TrainData
td = TrainData()
td.readIn("/eos/home-j/jkiesele/DeepNtuples/GraphGAN_test/test/out_800.meta")
x_gen = train.generator.predict(td.x)
forplots = np.concatenate([x_gen[0][:4], td.x[0][:4]],axis=0)
plotgrid(forplots, nplotsx=4, nplotsy=2, outname=train.outputDir+"comparison.pdf")


exit()






