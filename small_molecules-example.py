
# coding: utf-8

# In[ ]:


import numpy as np
import sys, os  
sys.path.append(os.path.join(os.environ["HOME"], "python_utils"))
sys.path.append(os.environ['soap_dir'])
import instruct

#sys.path.append(os.path.join(os.environ['HOME'], 'epfl/epfl/glosim'))
#sys.path.append(os.path.join(os.environ['HOME'], 'epfl/epfl_arjuna/epfl/glosim'))
#sys.path.append(os.path.join(os.environ['HOME'], 'epfl/epfl_nmpc_normalized_kernel/epfl/glosim'))
#sys.path.append(os.path.join(os.environ['HOME'], 'epfl/epfl_nmpc_normalized_kernel/epfl_arjuna/glosim'))
import quippy as qp


# In[ ]:


from ase.io import read,write
from ase.visualize import view


# In[ ]:


sys.path.insert(0,'../../../')
sys.path.insert(0,'../../../tools')
from GlobalSimilarity import get_environmentalKernels, get_globalKernel
from CV import CrossValidation
from krr import KRR,dump_json,load_json,dump_data,load_data,score

# # Compute a kernel and save it

class glosim2():
    def __init__(self):
        '''
        Interprets the instruction conf file
        
        '''
        inst_path = sys.argv[-1]
        inst = instruct.Instruct()
        inst.load_instruction_from_file(inst_path)

        sname = 'cross_validation'
        self.all_xyz_structs_fpath = inst.get(sname, 'all_xyz_structs_fpath')
        self.all_structs_property_fname = inst.get_with_default(sname, 'all_structs_property_fname', 'all_structs_property_file.npy')
        self.train_num_structs = inst.get_eval(sname, 'train_num_structs')
        self.test_num_structs = inst.get_eval(sname, 'test_num_structs')
        self.selection_methods = inst.get_eval(sname, 'selection_methods')
        self.num_replicas_with_constant_train_set = inst.get_eval(sname, 'num_replicas_with_constant_train_set')
        self.num_replicas_with_constant_test_set = inst.get_eval(sname, 'num_replicas_with_constant_test_set')


        sname = 'train_kernel'
        self.train_xyz_fname = inst.get_get_with_default(sname, 'train_xyz_fname', 'train.xyz')
        self.train_xyz_fname = inst.get_get_with_default(sname, 'test_xyz_fname', 'test.xyz')
        self.kernel_matrix_fname_prefix = inst.get_with_default(sname, kernel_matrix_fname_prefix, 'kernel_matrix')
        self.train_property_fname = inst.get_with_default(sname, 'train_property_fname', 'train_property_file.npy')
        self.test_property_fname = inst.get_with_default(sname, 'test_property_fname', 'test_property_file.npy')

        #SOAP params
        self.cutoff = inst.get_eval(sname, 'cutoff')
        self.nmax = inst.get_eval(sname, 'nmax')
        self.lmax = inst.get_eval(sname, 'lmax')
        self.gaussian_width = inst.get_eval(sname, 'gaussian_width')
        self.cutoff_transition_width = inst.get_eval(sname, 'cutoff_transition_width')
        self.centerweight = inst.get_eval(sname, 'centerweight')
        self.nocenters = inst.get_eval(sname, 'nocenters')
        self.chem_channels = inst.get_eval(sname, 'chem_channels')
        self.chemicalKernel = inst.get_with_default(sname, 'chemicalKernel', None)
        self.chemicalProjection = inst.get_with_default(sname, 'chemicalProjection', None)
        self.dispbar = inst.get_eval(sname, 'dispbar')
        self.is_fast_average = inst.get_eval(sname, 'is_fast_average')
        self.islow_memory = inst.get_eval(sname, 'islow_memory')
        self.nthreads = inst.get_eval(sname, 'nthreads')
        self.nchunks = inst.get_eval(sname, 'nchunks')
        self.nprocess = inst.get_eval(sname, 'nprocess')
        #more kernel params
        self.kernel_type = inst.get_eval(sname, 'kernel_type')
        self.zeta = inst.get_eval(sname, 'zeta')
        self.gamma = inst.get_eval(sname, 'gamma ')
        self.eps = inst.get_eval(sname, 'eps')
        self.normalize_global_kernel = inst.get_eval(sname, 'normalize_global_kernel')
        #self. = inst.get_eval(sname, '')

        sname = 'krr'
        self.krr_weights_fname = inst.get_with_default(sname, 'krr_weights_fname', 'krr_model_output.json')
        self.krr_sigma = inst.get_eval(sname, 'krr_sigma')
        self.krr_csi = inst.get_eval(sname, 'krr_csi')
        self.krr_sampleWeights = inst.get_with_default(sname, 'krr_sampleWeights', None)
        self.krr_memory_eff = inst.get_eval(sname, 'krr_memory_eff')

    def train_kernel(self):
        


def main():
    pass
    

# In[ ]:
#num_train is the number of structures to 
num_train = 30
frames = qp.AtomsList('./small_molecules.xyz',stop=num_train)


# In[ ]:


soap_params = dict(cutoff=3, nmax=6, lmax=6, gaussian_width=0.4,
                    cutoff_transition_width=0.5, centerweight=1.,nocenters=[],
                   chem_channels=True, is_fast_average=False,
                   islow_memory=False,nthreads=1,nchunks=1,nprocess=1)

environmentalKernels = get_environmentalKernels(atoms=frames,**soap_params)


# In[ ]:


kernel_params = dict(kernel_type='average', zeta=2, normalize_global_kernel=True)
globalKernel = get_globalKernel(environmentalKernels,**kernel_params)


# In[ ]:


prefix = './'
fn = 'my_kernel_matrix'
metadata = dict(soap_params=soap_params,fn=fn+'.npy',
                kernel_params=kernel_params)
dump_data(prefix+fn+'.json',metadata,globalKernel)


# # Train and save a KRR model

# In[ ]:


params, Kmat = load_data('./my_kernel_matrix.json')

train = range(num_train)

Kmat_train = Kmat[np.ix_(train,train)]
y_train = np.load('./small_molecules-dHf_peratom.npy')[train]


# In[ ]:


model = KRR(sigma=1e-1,csi=1)
model.train(Kmat_train,y_train)


# In[ ]:


state = model.pack()
dump_json('./my_krr_model.json',state)

# # Predict with saved model

# In[ ]:


params, Kmat = load_data('./my_kernel_matrix.json')

train = range(num_train)
test = range(int(num_train / 2),num_train)

Kmat_test = Kmat[np.ix_(train,test)]

y_test = np.load('./small_molecules-dHf_peratom.npy')[test]

model_state = load_json('./my_krr_model.json')
model = KRR().unpack(model_state)


# In[ ]:


y_pred = model.predict(Kmat_test)

print 'MAE={:.3e} RMSE={:.3e} SUP={:.3e} R2={:.3e} CORR={:.3e}'.format(*score(y_pred,y_test))

# # Kfold CV

# In[ ]:


params, Kmat = load_data('./my_kernel_matrix.json',mmap_mode=None)
y = np.load('./small_molecules-dHf_peratom.npy')[:num_train]
params = dict(sigma=1e-1,csi=1)


# In[ ]:


scoreTest, err_scoreTest = CrossValidation(Kmat,y,params,Nfold=4,seed=10)
print 'MAE={:.3e} RMSE={:.3e} SUP={:.3e} R2={:.3e} CORR={:.3e}'.format(*scoreTest)

