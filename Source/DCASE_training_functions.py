import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils  import Sequence
import numpy as np
import copy
import albumentations as A

#for implementing warm restarts in learning rate
class LR_WarmRestart(tensorflow.keras.callbacks.Callback):
    
    '''I. Loshchilov and F. Hutter. SGDR: stochastic gradient descent with restarts.
    http://arxiv.org/abs/1608.03983.'''
    
    def __init__(self,nbatch,initial_lr,min_lr,epochs_restart,Tmults):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.nbatch = nbatch
        self.currentEP=0.0
        self.startEP=1.0
        self.ThisBatch = 0.0
        self.lr_used=[]
        self.Tmults=Tmults
        self.Tcount=0
        self.epochs_restart=epochs_restart
        self.Init=False
        
    def on_epoch_begin(self, epoch, logs={}):
        self.currentEP = self.currentEP+1.0
        self.ThisBatch = 0.0
        if self.Init==False:
            K.set_value(self.model.optimizer.lr,self.initial_lr)
            self.Init=True
        if np.isin(self.currentEP,self.epochs_restart):
            self.startEP=self.currentEP
            #self.Tmult=self.currentEP+1.0
            self.Tmult=self.Tmults[self.Tcount]
            self.Tcount+=1
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

        
        

#data augmentations
def get_training_augmentation():
    train_transform = [
        A.RandomCrop(height=256, width=400, always_apply=True, p=1.0)
    ]
    return A.Compose(train_transform)



class MixupGenerator(tensorflow.keras.utils.Sequence):
    
    #generator of samples for training
    #applies the following data augmentation
    
    #1. mixup (randomly adds to samples using weights, and does the same on their labels
    #2. crop: crops a random temporal window of length crop_length out of a longer spectrogram
    #3. random channel swap 
   
    def __init__(self, X_train, y_train, batch_size=30, alpha=0.4, crop_length=400,UseBalance=False):

        self.augmentation1=get_training_augmentation()

        self.X_train = X_train
        self.y_train = y_train #categorical
        self.num_channels = X_train.shape[-1]
        
        self.batch_size = batch_size
        self.alpha = alpha
        self.crop_length = crop_length
        
        #initial shuffle
        self.UseBalance=UseBalance
        
        if self.UseBalance:
            self.y_labels=np.argmax(y_train,-1)
            Classes,self.ClassCounts = np.unique(self.y_labels,return_counts=True)
            self.num_classes = len(Classes)
            self.SamplesPerEpoch = self.num_classes*min(self.ClassCounts)
            self.ordering = self.ShuffleBalancedUnderSampling()
            
        else:
            self.ordering = np.arange(X_train.shape[0])
            np.random.shuffle(self.ordering)

        if alpha > 0:
            #divide by 2 because mixup would use each sample twice in an epoch otherwise
            self.len =int(np.floor(len(self.ordering)/batch_size)/2.0)
        else:
            self.len =int(np.floor(len(self.ordering)/batch_size))
            
        self.list1 = self.ordering.tolist()
        self.list2 = list(reversed(self.list1))
        
        
    def ShuffleBalancedUnderSampling(self):

        #create a list of each class, with indices randomly ordered
        EachClass=[]
        for i in range(self.num_classes):
            #find the indices for this class
            Class_indices = np.where(self.y_labels==i)[0]
            
            #order the indices for ths class randomly
            Class_ordering = np.arange(len(Class_indices))
            np.random.shuffle(Class_ordering)
            
            EachClass.append([Class_indices[j] for j in Class_ordering])

        #add one sample from each class each time through
        SampleIndices = []
        Count=0
        while Count <  min(self.ClassCounts): 
            
            #add from a randomly chosen class
            rand_class_order= np.arange(self.num_classes)
            np.random.shuffle(rand_class_order)
            
            for class_ind in rand_class_order:
                SampleIndices.append(EachClass[class_ind][Count])
            Count += 1
            
        return np.asarray(SampleIndices)
    
    
    def __len__(self):
        return self.len
    
    def on_epoch_end(self):
            
        if self.UseBalance:
            self.ordering = self.ShuffleBalancedUnderSampling()
        else:
            np.random.shuffle(self.ordering)
        self.list1 = self.ordering.tolist()
        self.list2 = list(reversed(self.list1))

    def __getitem__(self, index):
        
        #samples for mixup combining
        batch_ids1 = self.list1[index*self.batch_size:(index+1)*self.batch_size]
        if self.alpha >0:
            batch_ids2 = self.list2[index*self.batch_size:(index+1)*self.batch_size]
        
        #mixup weightings
        if self.alpha >0:
            mixup_weights = np.random.beta(self.alpha, self.alpha, self.batch_size)
        else:
            mixup_weights = np.zeros(self.batch_size)
       
        #apply augmentation and mixup. We assume here two-channel input
        X=np.zeros((self.batch_size,256,self.crop_length,self.num_channels))
        for j in range(self.batch_size):
            
            #temporal crop
            x1 = self.augmentation1(image=copy.deepcopy(self.X_train[batch_ids1[j],:,:,:]))['image']
            if self.alpha >0:
                x2 = self.augmentation1(image=copy.deepcopy(self.X_train[batch_ids2[j],:,:,:]))['image']
            
            #random channel swap
            if np.random.randint(2) == 1:
                x1 = x1[:,:,::-1]
            if self.alpha >0:
                if np.random.randint(2) == 1:
                    x2 = x2[:,:,::-1]
                
            #mixup
            if self.alpha >0:
                X[j,:,:,:] = x1 * mixup_weights[j] + x2 * (1.0 - mixup_weights[j])
            else:
                X[j,:,:,:] = x1

        #mixup for targets
        if self.alpha >0:
            y_mixup = mixup_weights.reshape(self.batch_size, 1)
            Y = self.y_train[batch_ids1] * y_mixup +self.y_train[batch_ids2] * (1.0 - y_mixup)
        else:
            Y = self.y_train[batch_ids1]
              
        return X,Y  