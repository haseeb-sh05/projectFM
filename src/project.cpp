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

	rfblocksize = rf_Fs*(block_duration_ms/1000);
	block_IQ_size = rfblocksize * 2;
	If_block_size = rfblocksize / rf_decim;
	audio_block_size = (If_block_size*audio_up)/audio_decim;

	int audio_M = filter_taps * audio_up;

	std::vector<real> rf_coeff, audio_coeff;

	impulseResponseLPF(rf_Fs, rf_Fc, filter_taps, rf_coeff, 1.0);
	impulseResponseLPF(If_Fs * audio_up, audio_Fc, audio_M, audio_coeff, (real)audio_up);

	std::vector<real> I_filter_state(filter_taps - 1, 0.0);
	std::vector<real> Q_filter_state(filter_taps - 1, 0.0);
	std::vector<real> audio_filter_state(audio_M - 1, 0.0);
	real I_demod_state = 0.0;
	real Q_demod_state = 0.0;

	std::vector<real> pilot_coeff, stereo_coeff, stereo_lpf_coeff;
	std::vector<real> pilot_state(filter_taps - 1, 0.0);
	std::vector<real> stereo_bpf_state(filter_taps - 1, 0.0);
	std::vector<real> stereo_lpf_state(audio_M - 1, 0.0);
	std::vector<real> pilot_filtered(If_block_size, 0.0);
	std::vector<real> stereo_filtered(If_block_size, 0.0);
	std::vector<real> nco_out(If_block_size, 0.0);
	std::vector<real> mixed(If_block_size, 0.0);
	std::vector<real> stereo_audio(audio_block_size, 0.0);
	std::vector<real> combined_pilot(filter_taps - 1 + If_block_size, 0.0);
	std::vector<real> combined_stereo_bpf(filter_taps - 1 + If_block_size, 0.0);
	std::vector<real> combined_stereo_lpf(audio_M - 1 + If_block_size, 0.0);
	std::vector<short int> final_stereo_data(2 * audio_block_size);
	PllState pll_state;

	int mono_delay_len = (filter_taps - 1) / 2;
	std::vector<real> mono_delay_state(mono_delay_len, 0.0);
	std::vector<real> delayed_fm(If_block_size, 0.0);

	if (farthest_path == 's') {
		impulseResponseBPF(If_Fs, 18.5e3, 19.5e3, filter_taps, pilot_coeff);
		impulseResponseBPF(If_Fs, 22e3, 54e3, filter_taps, stereo_coeff);
		impulseResponseLPF(If_Fs * audio_up, audio_Fc, audio_M, stereo_lpf_coeff, (real)audio_up * 2.0);
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
	std::vector<real> combinedaudio(audio_M - 1 + If_block_size,0.0);

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

		blockConvolve_DecimateFast(block_data_If_I, block_data_I, rf_coeff, I_filter_state, combinedRF, rf_decim);
		blockConvolve_DecimateFast(block_data_If_Q, block_data_Q, rf_coeff, Q_filter_state, combinedRF, rf_decim);

		fmDemodNoArctan(block_data_If_I, block_data_If_Q, I_demod_state, Q_demod_state, fm_demod_data);

		if (farthest_path == 's') {
			for (int k = 0; k < mono_delay_len; k++)
				delayed_fm[k] = mono_delay_state[k];
			for (int k = 0; k < If_block_size - mono_delay_len; k++)
				delayed_fm[k + mono_delay_len] = fm_demod_data[k];
			for (int k = 0; k < mono_delay_len; k++)
				mono_delay_state[k] = fm_demod_data[If_block_size - mono_delay_len + k];

			if (audio_up > 1)
				blockConvolve_ResampleFast(audio_data, delayed_fm, audio_coeff, audio_filter_state, combinedaudio, audio_decim, audio_up);
			else
				blockConvolve_DecimateFast(audio_data, delayed_fm, audio_coeff, audio_filter_state, combinedaudio, audio_decim);
		} else {
			if (audio_up > 1)
				blockConvolve_ResampleFast(audio_data, fm_demod_data, audio_coeff, audio_filter_state, combinedaudio, audio_decim, audio_up);
			else
				blockConvolve_DecimateFast(audio_data, fm_demod_data, audio_coeff, audio_filter_state, combinedaudio, audio_decim);
		}

		if (farthest_path == 's') {
			blockConvolve_DecimateFast(pilot_filtered, fm_demod_data, pilot_coeff, pilot_state, combined_pilot, 1);
			pllBlock(pilot_filtered, 19e3, If_Fs, 2.0, 0.0, 0.02, pll_state, nco_out);
			blockConvolve_DecimateFast(stereo_filtered, fm_demod_data, stereo_coeff, stereo_bpf_state, combined_stereo_bpf, 1);
			for (int k = 0; k < If_block_size; k++) mixed[k] = stereo_filtered[k] * nco_out[k];

			if (audio_up > 1)
				blockConvolve_ResampleFast(stereo_audio, mixed, stereo_lpf_coeff, stereo_lpf_state, combined_stereo_lpf, audio_decim, audio_up);
			else
				blockConvolve_DecimateFast(stereo_audio, mixed, stereo_lpf_coeff, stereo_lpf_state, combined_stereo_lpf, audio_decim);

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

	return 0;
}
