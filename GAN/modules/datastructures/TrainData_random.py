
from DeepJetCore.TrainData import TrainData, fileTimeOut
import numpy 

class TrainData_coord_gan(TrainData):
    def __init__(self):
        TrainData.__init__(self)

        self.treename="tree" #input root tree name
        
        self.truthclasses=[]#['isA','isB','isC'] #truth classes for classification
        self.regressiontargetclasses=['sigfrac_bgfrac_featforweights']
        
        self.weightbranchX='isA' #needs to be specified if weighter is used
        self.weightbranchY='isB' #needs to be specified if weighter is used
        
        #there is no need to resample/reweight
        self.weight=False
        self.remove=False
        #does not do anything in this configuration
        self.referenceclass='flatten'
        self.weight_binX = numpy.array([0,40000],dtype=float) 
        self.weight_binY = numpy.array([0,40000],dtype=float) 
        
        
        #self.registerBranches() #list of branches to be used 
        
        self.registerBranches(self.truthclasses)
        
        self.registerBranches(['scale','xcenter','ycenter','type','xcoords','ycoords'])
        #call this at the end
        self.reduceTruth(None)
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
    
        # this function defines how to convert the root ntuple to the training format
        # options are not yet described here
        import numpy as np
        import ROOT
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("tree")
        self.nsamples=tree.GetEntries()
        
        
        # user code, example works with the example 2D images in root format generated by make_example_data
        from DeepJetCore.preprocessing import read2DArray,readListArray
        print(filename)
        feature_image = read2DArray(filename,"tree","image2d",self.nsamples,24,24)
        
        npy_array = self.readTreeFromRootToTuple(filename)
        scale   = np.expand_dims(npy_array['scale'],axis=1)
        xcenter = np.expand_dims(npy_array['xcenter'],axis=1)
        ycenter = np.expand_dims(npy_array['ycenter'],axis=1)
        ptype   = np.expand_dims(npy_array['type'],axis=1)
        
        print('ycenter',ycenter.shape)
        
        add_features = np.concatenate([scale,xcenter,ycenter,ptype],axis=1)
        
        
        xcoords = numpy.expand_dims( numpy.array(list(npy_array['xcoords']),dtype='float32'), axis=2)
        ycoords = numpy.expand_dims( numpy.array(list(npy_array['ycoords']),dtype='float32'), axis=2)
        xcoords = numpy.reshape(xcoords, newshape=[xcoords.shape[0],24,24,1])
        ycoords = numpy.reshape(ycoords, newshape=[xcoords.shape[0],24,24,1])
        
        print('xcoords',xcoords.shape)
        
        all_coords = numpy.concatenate([xcoords,ycoords],axis=-1)
        
        #readListArray(filename,"tree","frac_at_idxs",self.nsamples,4,1)
        
        alltruth = numpy.zeros(self.nsamples)+1. #this is real data
        
        self.x = [feature_image,all_coords,add_features] 
        self.y = [alltruth]
        self.w=[]
        
    def replaceTruthForGAN(self, generated_array, original_truth): # this array will have B X 1 !
        #original_truth = generated_array
        return [generated_array]


    def formatPrediction(self, predicted_list):
        
        format_names = ['gen_image2d','all_coords','add_features']
        out_pred = predicted_list
        
        return out_pred,  format_names

