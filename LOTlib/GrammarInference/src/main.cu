/*
 *  Steve Piantadosi, October 2016
    Todo: 
        -Add prior offset
        -Should the prior be renormalized or not?
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
//#include "cuPrintf.cu" //!

// for randomness
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

// for loading matrices 
#include "hdf5.h"

// our kernels for prior, likelihood
#include "kernels.cu"

using namespace std;

// -----------------------------------------------------------------------
// CUDA setup
// -----------------------------------------------------------------------     

const int BLOCK_SIZE = 1024;
const int N_BLOCKS = 1024; // set below

// -----------------------------------------------------------------------
// Hyperparameters
// -----------------------------------------------------------------------     

// NOTE: If a future change lets us fit these, we need to add normalizing constants to the code below
const float TEMPERATURE_k = 10.0; // mean is k*theta, variance is k*theta^2
const float TEMPERATURE_theta = 0.1; 

// the k and theta on grammar productions x
const float RR_GAMMA_K = 2.0;
const float RR_GAMMA_THETA = 0.5;

float PCFG_DIRICHLET_ALPHA = 1.0;

const int NPARAMS = 4; // number of free parameters in fitting (alpha, beta, priortemp, lltemp)

// -----------------------------------------------------------------------
// Command line parameters
// -----------------------------------------------------------------------     

// Default parameters
int STEPS = 5000000;
int THIN  = 50;
int NCHAINS = 4;
int WHICH_GPU = 0;
float X_PROPOSAL_SCALE = 0.1; // scale of proposals to X
float PARAM_PROPOSAL_SCALE = 0.1; // scale of proposals to all parameters
string in_file_path = "data.h5"; 
bool DO_RR = 0; // do rational rules or pcfg?
bool start1 = 0; // should we start at 1 or randomly initialize?

static struct option long_options[] = {   
        {"in",           required_argument,    NULL, 'd'},
        {"steps",        required_argument,    NULL, 's'},
        {"thin",         required_argument,    NULL, 't'},
        {"chains",       required_argument,    NULL, 'c'},
        {"xscale",       required_argument,    NULL, 'x'},
        {"gpu",          required_argument,    NULL, 'g'},
        {"start1",       no_argument,    NULL, '1'}, // should we start at 1?        
        {"rr",           no_argument,    NULL, 'r'},
        {"pcfg",         no_argument,    NULL, 'p'},
        {NULL, 0, 0, 0} // zero row for bad arguments
};  

// -----------------------------------------------------------------------
// Macros
// -----------------------------------------------------------------------     

// Macro for defining arrays with name device_name, and then copying name over
#define DEVARRAY(type, nom, size) \
        type* device_ ## nom;\
        cudaMalloc((void **) &device_ ## nom, size); \
	cudaMemcpy(device_ ## nom, nom, size, cudaMemcpyHostToDevice);\
	error = cudaGetLastError();\
    if(error != cudaSuccess) {  printf("CUDA error on allocating " #nom ": %s\n", cudaGetErrorString(error)); } 
   

// macro for reading hdf5 call "nom" in the data to the variable nom
#define LOAD_HDF5(nom, type) \
    dset = H5Dopen(file, #nom, H5P_DEFAULT);\
    status = H5Dread (dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, nom);\
    assert(status==0);\
    status = H5Dclose (dset);\
    assert(status==0);
	
// -----------------------------------------------------------------------
// RNG
// -----------------------------------------------------------------------     

boost::mt19937 rng; // I don't seed it on purpouse (it's not relevant)
boost::normal_distribution<> nd(0.0, 1.0);
boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);
boost::uniform_01<boost::mt19937> random_real(rng);

// -----------------------------------------------------------------------
// Helpful functions
// -----------------------------------------------------------------------     

double lgammapdf(float x, float k, float theta) {
    // Does not bother with normalizing constant
    return log(x)*(k-1.0) - x/theta;
}

double ldirichletpdf(float x, float alpha) {
    // ignores normalizing constant
    return log(x)*(alpha-1.0);
}

void normalize_by_nonterminals(float* x, int NRULES, int NNT, int* ntlen){
    // renormalize a vector by its 
    int xi=0;
    for(int nt=0;nt<NNT;nt++) {
            float sm = 0.0;
            for(int i=0;i<ntlen[nt];i++) 
                    sm += x[xi+i];
            
            for(int i=0;i<ntlen[nt];i++) 
                    x[xi+i] = x[xi+i] / sm;            
            
            xi += ntlen[nt];
    }
    assert(xi == NRULES);   
}

// ############################################################################################
// # Main
// ############################################################################################

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
                    case 'c': NCHAINS = atoi(optarg); break;
                    case 'x': X_PROPOSAL_SCALE = atof(optarg); break;
                    case '1': start1 = 1; break;
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
    // Process the HDF5 file, make device arrays
    // -----------------------------------------------------------------------
    
    hid_t file = H5Fopen(in_file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dset; // used in macros for reading each data set 
    herr_t status; // used in macro for checking
    cudaError_t error; // used to error check after each allocation in case we are too big (thus assumed in the macro DEVARRAY)
    
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
    
    int* ntlen = new int[NNT]; // now load the lengths of each nonterminal
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
    
    char* names_str  = new char[1000]; // load the names for each column, tab separated already from python
    int* names = new int[1000]; // load as ints since chars are bad
    LOAD_HDF5(names, H5T_NATIVE_INT)
    for(int i=0;i<1000;i++) { names_str[i] = (char)names[i]; }
    
    // -----------------------------------------------------------------------
    // Local variables
    // -----------------------------------------------------------------------
    
    cout << "# Setting up MCMC variables" << endl;
    
    float X[NCHAINS][NRULES]; // the local copies of X, one for each chain; little x is the variable for the current chain
    for(int c=0;c<NCHAINS;c++) {
        for(int i=0;i<NRULES;i++) {
            X[c][i] = start1 ? 1.0 : abs(1.0+var_nor()); // start at normal value near 1
        }
        if(!DO_RR) normalize_by_nonterminals(X[c], NRULES, NNT, ntlen);
    }
    float x_size =  NRULES*sizeof(float);
    float* device_x;
    cudaMalloc((void **) &device_x, x_size); 
    cudaMemcpy(device_x, X[0], x_size, cudaMemcpyHostToDevice);
    error = cudaGetLastError();
    if(error != cudaSuccess) {  printf("CUDA error on allocating x: %s\n", cudaGetErrorString(error)); } 
   
    // other parameters
    float PARAMS[NCHAINS][NPARAMS];
    for(int c=0;c<NCHAINS;c++) {
        PARAMS[c][0] = start1 ? 0.5 : 1.0/(1.0+exp(-var_nor())); // alpha
        PARAMS[c][1] = start1 ? 0.5 : 1.0/(1.0+exp(-var_nor())); // beta
        PARAMS[c][2] = start1 ? 1.0 : abs(1.0+var_nor()); // priortemp
        PARAMS[c][3] = start1 ? 1.0 : abs(1.0+var_nor()); // lltemp
    }
    
    float prior_size =  NHYP*sizeof(float);
    float* prior = new float[NHYP];
    DEVARRAY(float, prior, prior_size)
    
    float* human_ll = new float[NDATA];
    int human_ll_size = NDATA*sizeof(float);
    DEVARRAY(float, human_ll, human_ll_size)

    float tmp_size =  NHYP*NDATA*sizeof(double);
    double* tmp = new double[NHYP*NDATA];
    DEVARRAY(double, tmp, tmp_size)
    
    // save old values when we propose
    float oldx[NRULES]; // used for copying and saving old version
    float oldparams[NPARAMS]; // temporary store for old values
    
    // -----------------------------------------------------------------------
    // Actual MCMC
    // -----------------------------------------------------------------------    
    
    double current[NCHAINS]; // the current posterior
    for(int c=0;c<NCHAINS;c++) current[c] = -INFINITY; // the current posterior
    
    double proposal; // store the proposal value 
   
    cout << "# Starting MCMC" << endl;
    
    // print a header
    cout << "steps\tchain\tposterior\tproposal\talpha\tbeta\tpriot.temp\tll.temp\t" << names_str << endl;
            
    for(int steps=0;steps<STEPS;steps++) {
        for(int chain=0;chain<NCHAINS;chain++) {
            float* x = X[chain]; // which chain are we on?
            float* params = PARAMS[chain];
            params[2] = 1.0;
            params[3] = 1.0;
            
            int proposetoX = steps%10!=0; //(rng() % 2)==0;   // decide whether to propose to x or something else
            if(proposetoX) {
                
                int j = rng()%NRULES; // who do I propose to?
                
                float v = x[j] + X_PROPOSAL_SCALE * var_nor(); // make a proposal
                if(v < 0.0) goto REJECT_SAMPLE; 
                
                memcpy(oldx, x, x_size); // save the old version
                
                x[j] = v;
                            
                if(!DO_RR) { // renormalize if we are doing a pcfg
                   normalize_by_nonterminals(X[chain], NRULES, NNT, ntlen);
                }
                
                            
            } else { // otherwise we propose to the parameters
                int i = rng() % NPARAMS;
                            
                float v = params[i] + PARAM_PROPOSAL_SCALE*var_nor();
                if( i <= 1 && (v > 0.999 || v < 0.001) ) goto REJECT_SAMPLE; // deal with bounds
                if( i >= 2 && (v < 0.001) )              goto REJECT_SAMPLE;

                memcpy(oldparams, params, NPARAMS*sizeof(float)); // save old version
                
                params[i] = v;
            }
            
            // and copy to the device (no matter the proposal, thanks to multiple chains)
            cudaMemcpy(device_x, x, x_size, cudaMemcpyHostToDevice);
        
            assert(BLOCK_SIZE*N_BLOCKS > NHYP);
            if(DO_RR) compute_RR_prior<<<N_BLOCKS,BLOCK_SIZE>>>(device_x, device_counts, device_prior, NHYP, NRULES, NNT, device_ntlen);
            else 	  compute_PCFG_prior<<<N_BLOCKS,BLOCK_SIZE>>>(device_x, device_counts, device_prior, NHYP, NRULES);
            cudaMemcpy(prior, device_prior, prior_size, cudaMemcpyDeviceToHost);
            
            error = cudaGetLastError();
            if(error != cudaSuccess) {  printf("CUDA error (in prior): %s\n", cudaGetErrorString(error)); break; } 
            
            assert(BLOCK_SIZE*N_BLOCKS > NDATA);
            compute_human_likelihood<<<N_BLOCKS,BLOCK_SIZE>>>(params[0], params[1], params[2], params[3], device_prior, device_likelihood, device_output, device_human_yes, device_human_no, device_human_ll, device_tmp, NHYP, NDATA);
            cudaMemcpy(human_ll, device_human_ll, human_ll_size, cudaMemcpyDeviceToHost);
        
            error = cudaGetLastError();
            if(error != cudaSuccess) {  printf("CUDA error (in likelihood): %s\n", cudaGetErrorString(error)); break; }	
/*
            for(int i=0;i<NDATA;i++) {
                cout << i <<" " << human_ll[i] << " " << endl;
            }
            return 0;*/
            
            // -------------------------------------------------------------------
            // Now compute the posterior
            proposal = 0.0; // the proposal probability

            // compute the prior on the params
            proposal += lgammapdf(params[2], TEMPERATURE_k, TEMPERATURE_theta) + 
                        lgammapdf(params[3], TEMPERATURE_k, TEMPERATURE_theta);
            
            // add up the prior on X
            for(int i=0;i<NRULES;i++) {
                if(DO_RR)
                    proposal += lgammapdf(x[i], RR_GAMMA_K, RR_GAMMA_THETA);
                else
                    proposal += ldirichletpdf(x[i], PCFG_DIRICHLET_ALPHA);
            }
            
            // now compute the human ll from what CUDA returned
            for(int d=0;d<NDATA;d++) {
    //             cout << human_ll[d] << " " ;
                proposal += human_ll[d];
            }
            
            // --------------------------------------------------------------------
            // decide whether to accept via MH rule
            // --------------------------------------------------------------------
            
            if(proposal > current[chain] || random_real() < exp(proposal - current[chain])){
                current[chain] = proposal;  // update current, that's all since X and params are already set
            }
            else {
                // restore what we had for whatever was proposed to
                if(proposetoX) memcpy(x, oldx, x_size);
                else           memcpy(params, oldparams, NPARAMS*sizeof(float));
            }
            
            // -----------------------------------------------------------------------
            // Print out
            // -----------------------------------------------------------------------
            
    REJECT_SAMPLE: // we'll skip the memcpy back since x was never set
    
            
            if(steps % THIN == 0) {
                cout << steps << " " << chain << "\t" << current[chain] << "\t" << proposal << "\t";
                for(int i=0;i<NPARAMS;i++) { cout << params[i] << " "; }
                cout << "\t";
                for(int i=0;i<NRULES;i++) { cout << x[i] << " "; }
                cout << endl;
            }
            
        } //chains
    } // steps

} // main
