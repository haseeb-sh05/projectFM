/*
Comp Eng 3DY4 (Computer Systems Integration Project)

Department of Electrical and Computer Engineering
McMaster University
Ontario, Canada
*/

#include "dy4.h"
#include "fourier.h"
#include "genfunc.h"
#include "iofunc.h"
#include "logfunc.h"
#include "filter.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>



int main(int argc, char* argv[])
{

	int mode = 0;
	char farthest_path = 'm';
	int filter_taps = 101;

    if (argc < 2) {
        std::cerr << "Operating in default mode 0 - mono path - filtertaps = 101" << std::endl;
    } 
	
	else if (argc == 3 || argc == 4) 
	{
        mode = atoi(argv[1]);
		farthest_path = argv[2][0];
		
		if (argc == 4)
			filter_taps = atoi(argv[3]);

        if (mode > 3)
		{
            std::cerr << "Wrong mode " << mode << std::endl;
            exit(1);
        }

    } 

	else 
	{
        std::cerr << "Usage: " << argv[0] << std::endl;
        std::cerr << "or " << std::endl;
        std::cerr << "Usage: " << argv[0] << " <mode>" << std::endl;
        std::cerr << "\t\t <mode> is a value from 0 to 3" << std::endl;
        exit(1);
    }

    std::cerr << "Operating in mode: " << mode << ", Farthest Path: " << farthest_path << ", Filter Taps: " << filter_taps <<  std::endl;

	std::cerr << "Working with reals on " << sizeof(real) << " bytes" << std::endl;

	//parameter definitions
	real rf_Fs, If_Fs, audio_Fs, rf_Fc, audio_Fc, block_duration_ms;
	int rf_decim, audio_decim, audio_up, rfblocksize, block_IQ_size, If_block_size, audio_block_size;

	audio_Fc = 16e3;
	rf_Fc = 100e3;

	if (mode == 0)
	{
		rf_Fs = 2400e3;
		If_Fs = 240e3;
		audio_Fs = 48e3;
		block_duration_ms = 40;
		rf_decim = 10;
		audio_decim = 5;
		audio_up = 1;
	}

	else if (mode == 1)
	{
		rf_Fs = 2304e3;
		If_Fs = 288e3;
		audio_Fs = 36e3;
		block_duration_ms = 56;
		rf_decim = 8;
		audio_decim = 8;
		audio_up = 1;
	}

	else if (mode == 2)
	{
		rf_Fs = 2400e3;
		If_Fs = 240e3;
		audio_Fs = 44.1e3;
		block_duration_ms = 60;
		rf_decim = 10;
		audio_decim = 800;
		audio_up = 147;
	}
	
	else
	{
		rf_Fs = 1800e3;
		If_Fs = 360e3;
		audio_Fs = 44.1e3;
		block_duration_ms = 30;
		rf_decim = 5;
		audio_decim = 400;
		audio_up = 49;
	}

	rfblocksize = rf_Fs*(block_duration_ms/1000); //num of samples in each block
	block_IQ_size = rfblocksize * 2;
	If_block_size = rfblocksize / rf_decim;
	audio_block_size = (If_block_size*audio_up)/audio_decim;
		
	std::vector<real> rf_coeff, audio_coeff;
	
	impulseResponseLPF(rf_Fs, rf_Fc, filter_taps, rf_coeff);
	impulseResponseLPF(If_Fs, audio_Fc, filter_taps, audio_coeff);

	std::vector<real> I_filter_state(filter_taps - 1, 0.0);
	std::vector<real> Q_filter_state(filter_taps - 1, 0.0);
	std::vector<real> audio_filter_state(filter_taps - 1, 0.0);
	real I_demod_state = 0.0;
	real Q_demod_state = 0.0;

	std::vector<real> pilot_coeff, stereo_coeff, stereo_lpf_coeff;
	std::vector<real> pilot_state(filter_taps - 1, 0.0);
	std::vector<real> stereo_bpf_state(filter_taps - 1, 0.0);
	std::vector<real> stereo_lpf_state(filter_taps - 1, 0.0);
	std::vector<real> pilot_filtered(If_block_size, 0.0);
	std::vector<real> stereo_filtered(If_block_size, 0.0);
	std::vector<real> nco_out(If_block_size, 0.0);
	std::vector<real> mixed(If_block_size, 0.0);
	std::vector<real> stereo_audio(audio_block_size, 0.0);
	std::vector<real> combined_pilot(filter_taps - 1 + If_block_size, 0.0);
	std::vector<real> combined_stereo_bpf(filter_taps - 1 + If_block_size, 0.0);
	std::vector<real> combined_stereo_lpf(filter_taps - 1 + If_block_size, 0.0);
	std::vector<short int> final_stereo_data(2 * audio_block_size);
	PllState pll_state;

	int mono_delay_len = (filter_taps - 1) / 2;
	std::vector<real> mono_delay_state(mono_delay_len, 0.0);
	std::vector<real> delayed_fm(If_block_size, 0.0);

	if (farthest_path == 's') {
		impulseResponseBPF(If_Fs, 18.5e3, 19.5e3, filter_taps, pilot_coeff);
		impulseResponseBPF(If_Fs, 22e3, 54e3, filter_taps, stereo_coeff);
		impulseResponseLPF(If_Fs, audio_Fc, filter_taps, stereo_lpf_coeff);
		for (auto &c : stereo_lpf_coeff) c *= 2.0;
	}

	int block_id = 0;

	std::vector<real> block_data(block_IQ_size,0.0);
	std::vector<real> block_data_I(rfblocksize,0.0);
	std::vector<real> block_data_Q(rfblocksize,0.0);
	std::vector<real> block_data_If_I(If_block_size,0.0);
	std::vector<real> block_data_If_Q(If_block_size,0.0);
	std::vector<real> fm_demod_data(If_block_size,0.0);

	std::vector<real> audio_data(audio_block_size,0.0);

	std::vector<real> combinedRF(filter_taps - 1 + rfblocksize,0.0);
	std::vector<real> combinedaudio(filter_taps - 1 + If_block_size,0.0);

	std::vector<short int> final_audiodata(audio_data.size());

	std::vector<char> raw_IQ_data(block_IQ_size);

	while (true)
	{
		readStdinBlockData(block_IQ_size, block_data, raw_IQ_data);
		if ((std::cin.rdstate()) != 0)
		{
			std::cerr << "End of input stream reached" << std::endl;
			exit(1);
		}

		if (block_id % 50 == 0)
			std::cerr << "Read block " << block_id << std::endl;

		UnInterleave_IQ(block_data, block_data_I, block_data_Q);

		blockConvolve_Decimate(block_data_If_I, block_data_I, rf_coeff, I_filter_state,combinedRF, rf_decim);
		blockConvolve_Decimate(block_data_If_Q, block_data_Q, rf_coeff, Q_filter_state,combinedRF, rf_decim);

		fmDemodNoArctan(block_data_If_I, block_data_If_Q, I_demod_state, Q_demod_state, fm_demod_data);

		if (farthest_path == 's') {
			for (int k = 0; k < mono_delay_len; k++)
				delayed_fm[k] = mono_delay_state[k];
			for (int k = 0; k < If_block_size - mono_delay_len; k++)
				delayed_fm[k + mono_delay_len] = fm_demod_data[k];
			for (int k = 0; k < mono_delay_len; k++)
				mono_delay_state[k] = fm_demod_data[If_block_size - mono_delay_len + k];
			blockConvolve_Decimate(audio_data, delayed_fm, audio_coeff, audio_filter_state, combinedaudio, audio_decim);
		} else {
			blockConvolve_Decimate(audio_data, fm_demod_data, audio_coeff, audio_filter_state, combinedaudio, audio_decim);
		}

		if (farthest_path == 's') {
			blockConvolve_Decimate(pilot_filtered, fm_demod_data, pilot_coeff, pilot_state, combined_pilot, 1);
			pllBlock(pilot_filtered, 19e3, If_Fs, 2.0, 0.0, 0.02, pll_state, nco_out);
			blockConvolve_Decimate(stereo_filtered, fm_demod_data, stereo_coeff, stereo_bpf_state, combined_stereo_bpf, 1);
			for (int k = 0; k < If_block_size; k++) mixed[k] = stereo_filtered[k] * nco_out[k];
			blockConvolve_Decimate(stereo_audio, mixed, stereo_lpf_coeff, stereo_lpf_state, combined_stereo_lpf, audio_decim);

			for (int k = 0; k < audio_block_size; k++) {
				real left  = 0.5 * (audio_data[k] + stereo_audio[k]);
				real right = 0.5 * (audio_data[k] - stereo_audio[k]);
				final_stereo_data[2 * k]     = std::isnan(left)  ? 0 : static_cast<short int>(left  * 16384);
				final_stereo_data[2 * k + 1] = std::isnan(right) ? 0 : static_cast<short int>(right * 16384);
			}
			fwrite(&final_stereo_data[0], sizeof(short int), final_stereo_data.size(), stdout);

		} else {
			for (unsigned int k = 0; k < audio_data.size(); k++) {
				if (std::isnan(audio_data[k]))
					final_audiodata[k] = 0;
				else
					final_audiodata[k] = static_cast<short int>(audio_data[k] * 16384);
			}
			fwrite(&final_audiodata[0], sizeof(short int), final_audiodata.size(), stdout);
		}
		block_id++;
	}






	


	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	/*
	
	// Binary files can be generated through the
	// Python models from the "../model/" sub-folder
	const std::string in_fname = "../data/fm_demod_10.bin";
	std::vector<real> bin_data;
	readBinData(in_fname, bin_data);

	// Generate an index vector to be used by logVector on the X axis
	std::vector<real> vector_index;
	genIndexVector(vector_index, bin_data.size());
	// Log time data in the "../data/" subfolder in a file with the following name
	// note: .dat suffix will be added to the log file in the logVector function
	logVector("demod_time", vector_index, bin_data);

	// Take a slice of data with a limited number of samples for the Fourier transform
	// note: NFFT constant is actually just the number of points for the
	// Fourier transform - there is no FFT implementation ... yet
	// unless you wish to wait for a very long time, keep NFFT at 1024 or below
	std::vector<real> slice_data = \
		std::vector<real>(bin_data.begin(), bin_data.begin() + NFFT);
	// note: make sure that binary data vector is big enough to take the slice

	// Declare a vector of complex values for DFT
	std::vector<std::complex<real>> Xf;
	// ... In-lab ...
	// Compute the Fourier transform
	// the function is already provided in fourier.cpp

	// Compute the magnitude of each frequency bin
	// note: we are concerned only with the magnitude of the frequency bin
	// (there is no logging of the phase response)
	std::vector<real> Xmag;
	// ... In-lab ...
	// Compute the magnitude of each frequency bin
	// the function is already provided in fourier.cpp

	// Log the frequency magnitude vector
	vector_index.clear();
	genIndexVector(vector_index, Xmag.size());
	logVector("demod_freq", vector_index, Xmag); // Log only positive freq

	// For your take-home exercise - repeat the above after implementing
	// your own function for PSD based on the Python code that has been provided
	// note the estimate PSD function should use the entire block of "bin_data"
	//
	// ... Complete as part of the take-home ...
	//

	// If you wish to write some binary files, see below example
	//
	// const std::string out_fname = "../data/outdata.bin";
	// writeBinData(out_fname, bin_data);
	//
	// output files can be imported, for example, in Python
	// for additional analysis or alternative forms of visualization

	// Naturally, you can comment the line below once you are comfortable to run GNU plot
	std::cout << "Run: gnuplot -e 'set terminal png size 1024,768' ../data/example.gnuplot > ../data/example.png\n";
	*/

	return 0;
}
