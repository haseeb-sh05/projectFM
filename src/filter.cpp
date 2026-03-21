/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#include "dy4.h"
#include "filter.h"
#include <cmath>


// Function to compute the impulse response "h" based on the sinc function
void impulseResponseLPF(real Fs, real Fc, unsigned short int num_taps, std::vector<real> &h)
{
	// Allocate memory for the impulse response
	h.clear();
	h.resize(num_taps, 0.0);

	int center = (num_taps - 1)/2;
	real scale_factor = 0.0;
	int n;

	for (int k = 0; k < (int)h.size(); k++){
		n = k - center;
		if (n == 0)
			h[k] = 2*Fc/Fs;
		
		else
			h[k] = (std::sin(2*PI*Fc*n/Fs))/(PI*n);

		h[k] = h[k] * (0.5 - (0.5)*std::cos(2*PI*k/(num_taps - 1)));
		scale_factor = scale_factor + h[k];
	}
	
	for (int k = 0; k < (int)h.size(); k++){
		h[k] = h[k]/scale_factor;
	}
}

// Function to compute the filtered output "y" by doing the convolution
// of the input data "x" with the impulse response "h"
void convolveFIR(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h)
{
	y.clear();
	y.resize(x.size() + h.size() - 1, 0.0);

	for (int n = 0; n < (int)y.size(); n++){
		for (int k = 0; k < (int) h.size(); k++) {
			if (n >= k && (n - k) < (int)x.size())
				y[n] += h[k] * x[n - k];
		}
	}
}

//////////////////////////////////////////////////////////////
// New code as part of benchmarking/testing and the project
//////////////////////////////////////////////////////////////

void convolveFIR_inefficient(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h) {
    y.clear();
    y.resize(int(x.size() + h.size()-1), 0.0);
    for (int n = 0; n < (int)y.size(); n++) {
        for (int k = 0; k < (int)x.size(); k++) {
            if ((n - k >= 0) && (n - k) < (int)h.size())
                y[n] += x[k] * h[n - k];
        }
    }
}

void convolveFIR_reference(std::vector<real> &y, const std::vector<real> &x, const std::vector<real> &h) {
    y.clear();
    y.resize(int(x.size() + h.size()-1), 0.0);

    for (int n = 0; n < (int)y.size(); n++) {
        for (int k = 0; k < (int)h.size(); k++) {
            if ((n - k >= 0) && (n - k) < (int)x.size())
                y[n] += h[k] * x[n - k];
        }
    }
}

void blockConvolve_Decimate(std::vector<real> &yb, const std::vector<real> &xb, const std::vector<real> &h, std::vector<real> &state, std::vector<real> &combined,  int decimation)
{
	int M = h.size();
	int unrolled_limit = M - (M % 4);


	for (int i = 0; i < M - 1; i++)
		combined[i] = state[i];

	for (int i = 0; i < (int)xb.size(); i++)
		combined[i + (M-1)] = xb[i];

	int idx = 0;
    for (int n = 0; n < (int)xb.size(); n+=decimation){
		int base  = n + (M-1);
		yb[idx] = 0;

        for (int k = 0; k < unrolled_limit; k+=4)
		{
                yb[idx] += h[k] * combined[base - k] + h[k + 1] * combined[base - (k+1)] + h[k + 2] * combined[base - (k+2)] + h[k + 3] * combined[base - (k+3)];
		}

		for (int k = unrolled_limit; k < M; k++)
			yb[idx] += h[k] * combined[base - k];
			
		idx++;
	}

	
    state.assign(xb.end() - (M-1), xb.end());
}

void fmDemodNoArctan(const std::vector<real> &I, const std::vector<real> &Q, real &previous_I, real &previous_Q, std::vector<real> &fm_demod)
{
	real current_I = 0.0;
	real current_Q = 0.0;
	real diff_I = 0.0;
	real diff_Q = 0.0;
	real numerator = 0.0;
	real denominator = 0.0;
	for (int k = 0; k < (int)I.size(); k++)
	{
		current_I = I[k];
		current_Q = Q[k];

		diff_Q = current_Q - previous_Q;
		diff_I = current_I - previous_I;

		numerator = (current_I * diff_Q) - (current_Q * diff_I);
		denominator = (current_I * current_I) + (current_Q * current_Q);

		if (denominator != 0)
			fm_demod[k] = numerator / denominator;
		else
			fm_demod[k] = 0.0;

		previous_I = current_I;
		previous_Q = current_Q;
	}

}
