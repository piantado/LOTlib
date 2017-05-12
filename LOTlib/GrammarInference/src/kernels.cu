
__global__ void compute_PCFG_prior(float* x, int* counts, float* to, int Nhyp, int Nrules) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Nhyp) { return; }
    
    float lp = 0.0;
    for(int r=0;r<Nrules;r++) {
        lp += counts[idx*Nrules+r]*logf(x[r]);
    }    
    
    to[idx] = lp;    
}

__global__ void compute_RR_prior(float* x, int* counts, float* to, int Nhyp, int Nrules, int NNT, int* ntlen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Nhyp) { return; }

    float lp = 0.0;
    
    int xi=0;
	for(int nt=0;nt<NNT;nt++) {
		
		float sumacnt = 0.0; // all the alphas plus counts
		float suma = 0.0; // all the alphas
		for(int i=0;i<ntlen[nt];i++){
			float a = x[xi+i]; // the current alpha
			float c = float(counts[idx*Nrules+(xi+i)]);
			sumacnt += c+a;
			suma += a;
			lp += lgamma(c + a) - lgamma(a);
		}
		lp += lgamma(suma) - lgamma(sumacnt);
		
		xi += ntlen[nt];
	}
    
    to[idx] = lp;    
}

__device__ double logsumexp(int idx, int ndata, double* x, const int n) {
    double mx = -1.0/0.0; // the max
    for(int i=0;i<n;i++) {
        if(x[ndata*i+idx]>mx) { mx=x[ndata*i+idx]; }
    }
    
    // TODO: Should be sorted
    double sum = 0.0;
    for(int i=0;i<n;i++) {
        sum += exp((double)x[ndata*i+idx]-mx);
    }  
    return log(sum)+mx;

}

__global__ void compute_human_likelihood(float alpha, float beta, float pt, float lt,
                                         float* prior, float* likelihood, float* output, 
                                         int* human_yes, int* human_no, float* to,
                                         double* tmp, int Nhyp,int Ndata) {
    
    // now, idx will run over data points and each thread will add its own hypotheses
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= Ndata) { return; }
    
    for(int h=0;h<Nhyp;h++) 
        tmp[Ndata*h+idx] = prior[h]/pt;
    float priorZ = logsumexp(idx,Ndata, tmp,Nhyp);    /// normalizing constant
    
    // the log posterior normalizing constant 
    for(int h=0;h<Nhyp;h++) 
        tmp[Ndata*h+idx] = prior[h]/pt-priorZ+likelihood[Ndata*h + idx]/lt;
    float Z = logsumexp(idx,Ndata, tmp, Nhyp);
    
    // now compute the p human data
    float pyes=0.0;
    for(int h=0;h<Nhyp;h++) {
        pyes += output[Ndata*h + idx] * expf(prior[h]/pt-priorZ + likelihood[Ndata*h + idx]/lt - Z); // weighted average over hypotheses
    }
    
//     to[idx] = pyes;
    to[idx] = human_yes[idx] * logf(      pyes*alpha  + (1.0-alpha)*beta) + \
              human_no[idx]  * logf( (1.0-pyes)*alpha + (1.0-alpha)*(1.0-beta));
}
