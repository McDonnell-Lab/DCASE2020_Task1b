import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils  import Sequence
import numpy as np
import copy

#for implementing warm restarts in learning rate
class LR_WarmRestart(tensorflow.keras.callbacks.Callback):
    
    #corrected by Mark McDonnell, 12 May 2020
    
    def __init__(self,nbatch,initial_lr,min_lr,epochs_restart):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.nbatch = nbatch
        self.currentEP=0.0
        self.startEP=1.0
        self.ThisBatch = 0.0
        self.lr_used=[]
        self.Tmult=0.0
        self.epochs_restart=epochs_restart
        
    def on_epoch_begin(self, epoch, logs={}):
        self.currentEP = self.currentEP+1.0
        self.ThisBatch = 0.0
        if np.isin(self.currentEP,self.epochs_restart):
            self.startEP=self.currentEP
            self.Tmult=self.currentEP+1.0
            K.set_value(self.model.optimizer.lr,self.initial_lr)
        print ('\n Start of Epoch Learning Rate = {:.6f}'.format(K.get_value(self.model.optimizer.lr)))

    def on_epoch_end(self, epochs, logs={}):
        print ('\n End of Epoch Learning Rate = {:.6f}'.format(self.lr_used[-1]))

        
    def on_batch_begin(self, batch, logs={}):
        
        pts = self.currentEP - self.startEP + self.ThisBatch/(self.nbatch-1.0)
        decay = 1.0+np.cos(pts/self.Tmult*np.pi)
        newlr = self.min_lr+0.5*(self.initial_lr-self.min_lr)*decay
        K.set_value(self.model.optimizer.lr,newlr)
        
        #keep track of what we  use in this batch
        self.lr_used.append(K.get_value(self.model.optimizer.lr))
        self.ThisBatch = self.ThisBatch + 1.0





class MixupGenerator(Sequence):
    
    #generator of samples for training
    #applies the following data augmentation
    
    #1. mixup (randomly adds to samples using weights, and does the same on their labels
    #2. crop: crops a random temporal window of length crop_length out of a longer spectrogram
    #3. random channel swap 
    
    #also optionally applies balanced undersamplng
    
    def __init__(self, X_train, y_train, batch_size=30, alpha=0.4, shuffle=True, crop_length=400,UseBalance=False): 
        self.X_train = X_train
        self.y_train = y_train
        self.y_labels = np.argmax(y_train,axis=-1)
        self.num_channels = X_train.shape[-1]
        
        self.UseCount = np.zeros(self.y_train.shape[0])
        
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.NewLength = crop_length
        if self.num_channels == 6:
            self.swap_inds = [1,0,3,2,5,4]
        elif self.num_channels == 4:
            self.swap_inds = [1,0,3,2]
        elif self.num_channels == 2:
            self.swap_inds = [1,0]
            
        self.UseBalance=UseBalance
        
        if self.UseBalance:
            Classes,self.ClassCounts = np.unique(self.y_labels,return_counts=True)
            self.num_classes = len(Classes)
            self.reset_length = min(int(np.floor(self.num_classes*min(self.ClassCounts)/self.batch_size)),int(np.floor(self.X_train.shape[0]/self.batch_size)))
            print('mixup samples per epoch with balanced undersampling =',self.reset_length*self.batch_size)
            self.ordering = self.ShuffleBalancedUnderSampling()
        else:
            self.ordering = np.arange(X_train.shape[0])
            np.random.shuffle(self.ordering)
        self.start_ind=0
        
    def __len__(self):
        if self.UseBalance:
            return self.reset_length
        else:
            return int(np.floor(self.X_train.shape[0]/self.batch_size))
    
  
          
    def ShuffleBalancedUnderSampling(self):

        EachClass=[]
        for i in range(self.num_classes):
            Class_indices = np.where(self.y_labels==i)[0]
            
            Class_ordering = np.arange(len(Class_indices))
            np.random.shuffle(Class_ordering)
            
            EachClass.append([Class_indices[j] for j in Class_ordering])

        SampleIndices = []
        Count=0
        while True: 
            #add one samples from each class each time through
            rand_class_order= np.arange(self.num_classes)
            np.random.shuffle(rand_class_order)
            
            for class_ind in rand_class_order:
                SampleIndices.append(EachClass[class_ind][Count])
            Count += 1
            
            if len(self.ClassCounts)*Count >= self.reset_length*self.batch_size:
                break
        return np.asarray(SampleIndices)


    def __getitem__(self, index):
        
        #mixup random numbers for each sample
        if self.alpha >0:
            mixup_weights = np.random.beta(self.alpha, self.alpha, self.batch_size)
        else:
            mixup_weights = np.zeros(self.batch_size)
        X_mixup = mixup_weights.reshape(self.batch_size, 1, 1, 1)
        y_mixup = mixup_weights.reshape(self.batch_size, 1)

        #samples for mixup combining
        batch_ids1 = self.ordering.tolist()[self.start_ind:self.start_ind+self.batch_size]
        batch_ids2 = list(reversed(self.ordering.tolist()))[self.start_ind:self.start_ind+self.batch_size]   
        X1 = copy.deepcopy(self.X_train[batch_ids1,:,:,:])
        X2 = copy.deepcopy(self.X_train[batch_ids2,:,:,:])
        
        self.UseCount[batch_ids1]+=1
        self.UseCount[batch_ids2]+=1
        
        for j in range(X1.shape[0]):
            
            #random temporal cropping
            StartLoc1 = np.random.randint(0,X1.shape[2]-self.NewLength)
            StartLoc2 = np.random.randint(0,X2.shape[2]-self.NewLength)
            X1[j,:,0:self.NewLength,:] = X1[j,:,StartLoc1:StartLoc1+self.NewLength,:]
            X2[j,:,0:self.NewLength,:] = X2[j,:,StartLoc2:StartLoc2+self.NewLength,:]
            
            #randomly swap left and right channels, if we have stereo input
            if X1.shape[-1]==2 or X1.shape[-1]==4 or X1.shape[-1]==6:
                if np.random.randint(2) == 1:
                    X1[j,:,:,:] = X1[j:j+1,:,:,self.swap_inds]
                if np.random.randint(2) == 1:
                    X2[j,:,:,:] = X2[j:j+1,:,:,self.swap_inds]
            
        #shorten to the cropped length  and apply mixup
        X = X1[:,:,0:self.NewLength,:] * X_mixup + X2[:,:,0:self.NewLength,:] * (1.0 - X_mixup)
        y = self.y_train[batch_ids1] * y_mixup +self.y_train[batch_ids2] * (1.0 - y_mixup)

        #update sample accounting
        self.start_ind = self.start_ind + self.batch_size
        if self.UseBalance:
            if self.start_ind > self.reset_length*self.batch_size-self.batch_size:
                self.ordering = self.ShuffleBalancedUnderSampling()
                self.start_ind=0
        else:
            if self.start_ind > self.X_train.shape[0]-self.batch_size:
                np.random.shuffle(self.ordering)
                self.start_ind=0
            
        return X, y