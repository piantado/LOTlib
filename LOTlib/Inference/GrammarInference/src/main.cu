/*
    Todo: Add prior offset
*/
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <getopt.h>
#include <string.h>

// for CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>
#include "cuPrintf.cu" //!

// for randomness
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

// for loading matrices 
#include "hdf5.h"

// our kernels for prior, likelihood
#include "kernels.cu"

using namespace std;

const int BLOCK_SIZE = 1024;
const int N_BLOCKS = 1024; // set below
const int NPARAMS = 4;

// priors on temperatures // NOTE: These values are not appropriate to fit because the normalizing constants are not computed in the posteriors below
const float TEMPERATURE_k = 2.0;
const float TEMPERATURE_theta = 0.5; 

// the k and theta on grammar productions x
const float X_k = 2.0;
const float X_theta = 0.5;

// Default parameters
int STEPS = 1000000;
int THIN  = 1000;
int WHICH_GPU = 0;
string in_file_path = "data.h5"; 
bool DO_RR = 0; // do rational rules or pcfg?

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
        



double lgammapdf(float x, float k, float theta) {
    // Does not bother with normalizing constant
    return log(x)*(k-1.0) - x/theta;
}




static struct option long_options[] =
    {   
        {"in",           required_argument,    NULL, 'd'},
        {"steps",        required_argument,    NULL, 's'},
        {"thin",         required_argument,    NULL, 't'},
        {"gpu",          required_argument,    NULL, 'g'},
        {"rr",           no_argument,    NULL, 'r'},
        {"pcfg",         no_argument,    NULL, 'p'},
        {NULL, 0, 0, 0} // zero row for bad arguments
    };  

int main(int argc, char** argv) {    
    
    // -----------------------------------------------------------------------
    // Parse command line
    // -----------------------------------------------------------------------
    
    int option_index = 0, opt=0;
    while( (opt = getopt_long( argc, argv, "bp", long_options, &option_index )) != -1 )
            switch( opt ) {
                    case 'd': in_file_path = optarg; break;
                    case 's': STEPS = atoi(optarg); break;
                    case 't': THIN = atoi(optarg); break;
                    case 'g': WHICH_GPU = atoi(optarg); break;
                    case 'r': DO_RR = 1; break;
                    case 'p': DO_RR = 0; break;
                    default: return 1; // unspecified
            }
    
    // -----------------------------------------------------------------------
    // Initialize the GPU
    // -----------------------------------------------------------------------
     
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if(WHICH_GPU <= deviceCount) {
        cudaSetDevice(WHICH_GPU);
    }
    else {
        cerr << "Invalid GPU device " << WHICH_GPU << endl;
        return 1;
    }
        
    // -----------------------------------------------------------------------
    // Process the HDF5 file
    // -----------------------------------------------------------------------
    
    hid_t file = H5Fopen(in_file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dset; // used in macros for reading each data set 
    herr_t status; // used in macro for checking
    
    // first load the specs to unpack all of our variable dimensions
    int specs[5];
    LOAD_HDF5(specs, H5T_NATIVE_INT)
    int NHYP   = specs[0];
    int NRULES = specs[1];
    int NDATA  = specs[2];
    int NNT    = specs[3];

    cout << "# Loaded with " << NHYP << " hypotheses, " << 
                              NRULES << " rules, " << 
                               NDATA << " data, and " << 
                                 NNT << " nonterminals." << endl;
    cout << "# Allocating memory " << endl;
    
    int ntlen[NNT]; // now load the lengths of each nonterminal
    LOAD_HDF5(ntlen, H5T_NATIVE_INT)
    DEVARRAY(int, ntlen, NNT*sizeof(int))
    
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
    
    // -----------------------------------------------------------------------
    // Local variables
    // -----------------------------------------------------------------------
    
    cout << "# Setting up MCMC variables" << endl;
    
    float X[NRULES]; // probabilities
    for(int i=0;i<NRULES;i++) 
        X[i] = 1.0;
    float X_size =  NRULES*sizeof(float);
    DEVARRAY(float, X, X_size)
    float oldX[NRULES]; // used for copying and saving old version
    
    // other parameters
    float params[NPARAMS] = {0.5, 0.5, 1.0, 1.0}; // alpha, beta, priortemp, lltemp
    float oldparams[NPARAMS];
    
    float prior_size =  NHYP*sizeof(float);
    float* prior = new float[NHYP];
    DEVARRAY(float, prior, prior_size)
    
    float* human_ll = new float[NDATA];
    int human_ll_size = NDATA*sizeof(float);
    DEVARRAY(float, human_ll, human_ll_size)
    
    // -----------------------------------------------------------------------
    // Actual MCMC
    // -----------------------------------------------------------------------    
    
    double current = -INFINITY; // the current posterior
    double proposal; // store the proposal value 
    cudaError_t error;
    
    cout << "# Starting MCMC" << endl;
    for(int steps=0;steps<STEPS;steps++) {
		
        int proposetoX = (rng() % 2)==0;   // decide whether to propose to x or something else
        if(proposetoX) {
            
            int j = rng()%NRULES; // who do I propose to?
            
            float v = X[j] + (DO_RR ? 0.1 : 0.01) *var_nor(); // make a proposal
            
            if(v < 0.0) goto REJECT_SAMPLE; // 
            
            memcpy(oldX, X, X_size); // save the old version
            
            X[j] = v;
                        
            if(!DO_RR) { // renormalize if we are doing a pdcfg   TODO: CHECK F/B
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
            }
            
            // and copy to the device
            cudaMemcpy(device_X, X, X_size, cudaMemcpyHostToDevice);
        
        } else { // otherwise we propose to the parameters
            int i = rng() % NPARAMS;
                        
            float v = params[i] + 0.01*var_nor();
            
            // deal with bounds
            if( i <= 1 && (v > 0.99 || v < 0.01) ) goto REJECT_SAMPLE;
            if( i >= 2 && (v < 0.01) )             goto REJECT_SAMPLE;

            memcpy(oldparams, params, NPARAMS*sizeof(float)); // save old version
            
            params[i] = v;
        }
        
        assert(BLOCK_SIZE*N_BLOCKS > NHYP);
        if(DO_RR) compute_RR_prior<<<N_BLOCKS,BLOCK_SIZE>>>(device_X, device_counts, device_prior, NHYP, NRULES, NNT, device_ntlen);
        else 	  compute_PCFG_prior<<<N_BLOCKS,BLOCK_SIZE>>>(device_X, device_counts, device_prior, NHYP, NRULES);
        cudaMemcpy(prior, device_prior, prior_size, cudaMemcpyDeviceToHost);

        assert(BLOCK_SIZE*N_BLOCKS > NDATA);
        compute_human_likelihood<<<N_BLOCKS,BLOCK_SIZE>>>(params[0], params[1], params[2], params[3], device_prior, device_likelihood, device_output, device_human_yes, device_human_no, device_human_ll, NHYP, NDATA);
        cudaMemcpy(human_ll, device_human_ll, human_ll_size, cudaMemcpyDeviceToHost);
        
        error = cudaGetLastError();
        if(error != cudaSuccess) {  printf("CUDA error: %s\n", cudaGetErrorString(error)); cerr << error << endl; break; }	
        
        // -------------------------------------------------------------------
        // Now compute the posterior
        proposal = 0.0; // the proposal probability

        // compute the prior on the params
        proposal += lgammapdf(params[2], TEMPERATURE_k, TEMPERATURE_theta) + 
	                lgammapdf(params[3], TEMPERATURE_k, TEMPERATURE_theta);
        
        // add up the prior on X
        for(int i=0;i<NRULES;i++) {
            proposal += lgammapdf(X[i], X_k, X_theta);
        }
        
        // now compute the human ll from what CUDA returned
        for(int d=0;d<NDATA;d++) {
//             cout << human_ll[d] << " " ;
            proposal += human_ll[d];
        }
        
        // --------------------------------------------------------------------
        // decide whether to accept via MH rule
        
        if(proposal > current || random_real() < exp(proposal - current)){
            current = proposal;  // update current, that's all since X and params are already set
        }
        else {
            // restore what we had for whatever was proposed to
            if(proposetoX) memcpy(X, oldX, X_size);
            else           memcpy(params, oldparams, NPARAMS*sizeof(float));
        }
        
        // -----------------------------------------------------------------------
        // Print out
        // -----------------------------------------------------------------------
    
        
REJECT_SAMPLE: // we'll skip the memcpy back since X was never set

        if(steps % THIN == 0) {
            cout << steps << "\t" << current << "\t" << proposal << "\t";
            for(int i=0;i<4;i++) { cout << params[i] << " "; }
            cout << "\t";
            for(int i=0;i<NRULES;i++) { cout << X[i] << " "; }
            cout << endl;
        }
    }
    

}
