
/*

    TODO:   
        - Fix param proposal near 0,1, for fb
        - Fix grammar proposal with abs --> Just reject proposal
        - Add parameter priors
*/
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <getopt.h>
#include <string.h>
#include <vector>
#include "cuPrintf.cu" //!

// for randomness
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

// for loading matrices 
#include "hdf5.h"


using namespace std;

const int BLOCK_SIZE = 1024;
const int N_BLOCKS = 1024; // set below

// Macro for defining arrays with name device_name, and then copying name over
#define DEVARRAY(type, nom, size) \
        type* device_ ## nom;\
        cudaMalloc((void **) &device_ ## nom, size); \
	cudaMemcpy(device_ ## nom, nom, size, cudaMemcpyHostToDevice);

// macro for reading hdf5 call "nom" in the data to the variable nom
#define LOAD_HDF5(nom, type) \
    dset = H5Dopen(file, #nom, H5P_DEFAULT);\
    status = H5Dread (dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, nom);\
    assert(status==0);\
    status = H5Dclose (dset);\
    assert(status==0);
	
boost::mt19937 rng; // I don't seed it on purpouse (it's not relevant)
boost::normal_distribution<> nd(0.0, 1.0);
boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);
boost::uniform_01<boost::mt19937> random_real(rng);
        
/*
 * To add: prior offset
 */

__global__ void compute_prior(float* lpx, int* counts, float* to, int Nhyp, int Nrules) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Nhyp) { return; }
    
    float p = 0.0;
    for(int r=0;r<Nrules;r++) {
        p += counts[idx*Nrules+r]*logf(lpx[r]);
    }    
    
    to[idx] = p;    
}



/*
 * output[h][d] gives the output for the dth point
 * prior[h]
 * likelihood[h][d]
 * output[h][d]
 * human_yes[d]
 * human_no[d]
 * to[d]
 
 We might be able to make this faster by parallelizing over hypotheses rather than data points
 since there are more hyps. Then we'd have to sum up with separate kernels
 */
__global__ void compute_human_likelihood(float alpha, float beta, float pt, float lt,
                                         float* prior, float* likelihood, float* output, 
                                         int* human_yes, int* human_no, float* to, int Nhyp, int Ndata) {
 
    // now, idx will run over data points and each thread will add its own hypotheses
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Ndata) { return; }
    
    // logsumexp normalizing constant Z 
    float mx = -9e99;
    for(int h=0;h<Nhyp;h++) {
        float v = prior[h]/pt+likelihood[idx*Nhyp + h]/lt;
        if(v>mx) { mx=v; }
    }
    float sm = 0.0;
    for(int h=0;h<Nhyp;h++) {
        sm += expf(prior[h]/pt+likelihood[idx*Nhyp + h]/lt - mx);
    }
    float Z = mx+logf(sm);    /// normalizing constant

    // now compute the p human data
    float pyes=0.0;
    for(int h=0;h<Nhyp;h++) {
        pyes += output[idx*Nhyp + h] * expf(prior[h]/pt+likelihood[idx*Nhyp + h]/lt - Z); // weighted average over hypotheses
    }
    
    to[idx] = logf(      pyes*alpha  + (1.0-alpha)*beta)*human_yes[idx] + \
              logf( (1.0-pyes)*alpha + (1.0-alpha)*(1.0-beta))*human_no[idx];
}




int main(int argc, char** argv) {    
    
    hid_t file = H5Fopen("/home/piantado/Desktop/datasets/data-50concept.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dset; // used in macros for reading each data set 
    herr_t status; // used in macro for checking
    
    // first load the specs to unpack all of our variable dimensions
    int specs[5];
    LOAD_HDF5(specs, H5T_NATIVE_INT)
    int NHYP   = specs[0];
    int NRULES = specs[1];
    int NDATA  = specs[2];
    int NNT    = specs[3];

    cout << "# Loaded with " << NHYP << " hypotheses, " << NRULES << " rules, " << NDATA << " data, and " << NNT << " nonterminals." << endl;
    
    int ntlen[NNT]; // now load the lengths of each nonterminal
    LOAD_HDF5(ntlen, H5T_NATIVE_INT)

    // Now load the real data
    int counts_size = NHYP*NRULES*sizeof(int);
    int* counts = new int[NHYP*NRULES];
    LOAD_HDF5(counts, H5T_NATIVE_INT)
    DEVARRAY(int, counts, counts_size)

    int output_size = NHYP*NDATA*sizeof(int);
    float* output = new float[NHYP*NDATA];
    LOAD_HDF5(output, H5T_NATIVE_FLOAT)
    DEVARRAY(float, output, output_size)
    
    int human_yes_size = NDATA*sizeof(int);
    int* human_yes = new int[NDATA];
    LOAD_HDF5(human_yes, H5T_NATIVE_INT)
    DEVARRAY(int, human_yes, human_yes_size)
        
    int human_no_size = NDATA*sizeof(int);
    int* human_no = new int[NDATA];
    LOAD_HDF5(human_no, H5T_NATIVE_INT)    
    DEVARRAY(int, human_no, human_no_size)
    
    int likelihood_size = NHYP*NDATA*sizeof(float);
    float* likelihood = new float[NHYP*NDATA];
    LOAD_HDF5(likelihood, H5T_NATIVE_FLOAT)
    DEVARRAY(float, likelihood, likelihood_size)
    

    /////////////////////////////////////////
    // Set up local variables 
    /////////////////////////////////////////
    cout << "# Setting up MCMC variables" << endl;
    
    float X[NRULES]; // probabilities
    for(int i=0;i<NRULES;i++) 
        X[i] = 1.0;
    float X_size =  NRULES*sizeof(float);
    DEVARRAY(float, X, X_size)
    float oldX[NRULES]; // used for copying and saving old version
    
    // other parameters
    const int NPARAMS = 4;
    float params[NPARAMS] = {0.75, 0.5, 1.0, 1.0}; // alpha, beta, priortemp, lltemp
    float oldparams[NPARAMS];
    
    float prior_size =  NHYP*sizeof(float);
    float prior[NHYP];
    DEVARRAY(float, prior, prior_size)
    
    float human_ll[NDATA];
    int human_ll_size = NDATA*sizeof(float);
    DEVARRAY(float, human_ll, human_ll_size)

    /////////////////////////////////////////
    // MCMC
    /////////////////////////////////////////
    
    double current = -99e99; // proposals to a single dimension of X (for simplicity)
    int proposal_i = 0;
    
    cout << "# Starting MCMC" << endl;
    for(int steps=0;steps<100;steps++) {
        
        // decide whether to propose to x or something else
        int proposetoX = (rng() % 2)==0; 
        if(proposetoX) {
            
            proposal_i = rng()%NRULES;
        
            memcpy(oldX, X, X_size);
            
            X[proposal_i] = abs(X[proposal_i] + 0.01*var_nor());
            
            // renormalize                                          TODO: CHECK F/B
            int xi=0;
            for(int nt=0;nt<NNT;nt++) {
                float sm = 0.0;
                for(int i=0;i<ntlen[nt];i++) 
                    sm += X[xi+i];
                
                for(int i=0;i<ntlen[nt];i++) 
                    X[xi+i] = X[xi+i] / sm;            
                
                xi += ntlen[nt];
            }
            assert(xi == NRULES);
            
            cudaMemcpy(device_X, X, X_size, cudaMemcpyHostToDevice);
        } else {
            int i = rng() % NPARAMS;
            
            memcpy(oldparams, params, NPARAMS*sizeof(float));
            
            params[i] = params[i] + 0.05*var_nor();
            
            // enforce some bounds
            if(params[0] < 0.01) params[0] = 0.01;
            if(params[0] > 0.99) params[0] = 0.99;
            if(params[1] < 0.01) params[1] = 0.01;
            if(params[1] > 0.99) params[1] = 0.99;
            if(params[2] < 0.01) params[2] = 0.01;
            if(params[3] < 0.01) params[3] = 0.01;       
        }
        
        // TODO: ENSURE THAT BLOCKS ARE SET CORRECTLY AND OPTIMALLY
        
        assert(BLOCK_SIZE*N_BLOCKS > NHYP);
        compute_prior<<<N_BLOCKS,BLOCK_SIZE>>>(device_X, device_counts, device_prior, NHYP, NRULES);
        cudaMemcpy(prior, device_prior, prior_size, cudaMemcpyDeviceToHost);
//             for(int h=0;h<10;h++) {  cout << prior[h] << " "; }
        
        assert(BLOCK_SIZE*BLOCK_SIZE > NDATA);
        compute_human_likelihood<<<N_BLOCKS,BLOCK_SIZE>>>(params[0], params[1], params[2], params[3], device_prior, device_likelihood, device_output, device_human_yes, device_human_no, device_human_ll, NHYP, NDATA);
        
        cudaMemcpy(human_ll, device_human_ll, human_ll_size, cudaMemcpyDeviceToHost);
        
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            break;
        }	
        
        double proposal = 0.0;
        for(int d=0;d<NDATA;d++) {
//             cout << human_ll[d] << " " ;
            proposal += human_ll[d];
        }
        
        // decide whether to accept
        if(proposal > current || random_real() < exp(proposal - current)){
            current = proposal;  // update current, that's all
        }
        else {
            // restore what we had
            if(proposetoX) {        
                memcpy(X, oldX, X_size);
            } else {
                memcpy(params, oldparams, NPARAMS*sizeof(float));
            }            
        }
        
        cout << steps << "\t" << current << "\t" << proposal << "\t";
        for(int i=0;i<4;i++) { cout << params[i] << " "; }
        cout << "\t";
        for(int i=0;i<NRULES;i++) { cout << X[i] << " "; }
        cout << endl;
    }
    

}
