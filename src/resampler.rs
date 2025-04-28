
use std::{cmp::min, sync::Arc, fmt::{self, Debug, Formatter}};
use rustfft::{FftPlanner, Fft, num_complex::Complex};

#[derive(Debug, Clone)]
pub enum ResamplerError {
    SizeError(String),
}

/// How the Resampler works
/// For audio stretching:
///   1. The input audio remains its original length, and zero-padding is applied at the end to reach the target length.
///   2. Perform FFT transformation to obtain the frequency domain.
///   3. In the frequency domain, scale down the frequency values proportionally (shift them lower).
///   4. Perform inverse FFT to obtain the stretched audio.
/// 
/// For audio compression:
///   1. Take the input audio.
///   2. Perform FFT transformation.
///   3. In the frequency domain, scale up the frequency values proportionally (shift them higher).
///   4. Perform inverse FFT to obtain audio with increased pitch but unchanged length.
///   5. Truncate the audio to shorten its duration.
/// 
/// This implies: the FFT length must be chosen as the longest possible length involved.
#[derive(Clone)]
pub struct Resampler {
    fft_forward: Arc<dyn Fft<f64>>,
    fft_inverse: Arc<dyn Fft<f64>>,
    fft_size: usize,
    normalize_scaler: f64,
}

fn get_average(complexes: &[Complex<f64>]) -> Complex<f64> {
    let sum: Complex<f64> = complexes.iter().copied().sum();
    let scaler = 1.0 / complexes.len() as f64;
    Complex::<f64> {
        re: sum.re * scaler,
        im: sum.im * scaler,
    }
}

fn interpolate(c1: Complex<f64>, c2: Complex<f64>, s: f64) -> Complex<f64> {
    c1 + (c2 - c1) * s
}

impl Resampler {
    pub fn new(fft_size: usize) -> Self {
        let mut planner = FftPlanner::new();
        if fft_size & 1 != 0 {
            panic!("The input size and the output size must be times of 2, got {fft_size}");
        }
        Self {
            fft_forward: planner.plan_fft_forward(fft_size),
            fft_inverse: planner.plan_fft_inverse(fft_size),
            fft_size,
            normalize_scaler: 1.0 / fft_size as f64,
        }
    }

    /// `desired_length`: The target audio length to achieve, which must not exceed the FFT size.
    /// When samples.len() < desired_length, it indicates audio stretching to desired_length.
    /// When samples.len() > desired_length, it indicates audio compression to desired_length.
    pub fn resample_core(&self, samples: &[f32], desired_length: usize) -> Result<Vec<f32>, ResamplerError> {
        const INTERPOLATE_UPSCALE: bool = true;
        const INTERPOLATE_DNSCALE: bool = true;

        let input_size = samples.len();
        if input_size == desired_length {
            return Ok(samples.to_vec());
        }

        if desired_length > self.fft_size {
            return Err(ResamplerError::SizeError(format!("The desired size {desired_length} must not exceed the FFT size {}", self.fft_size)));
        }

        let mut fftbuf: Vec<Complex<f64>> = samples.iter().map(|sample: &f32| -> Complex<f64> {Complex{re: *sample as f64, im: 0.0}}).collect();

        if fftbuf.len() <= self.fft_size {
            fftbuf.resize(self.fft_size, Complex{re: 0.0, im: 0.0});
        } else {
            return Err(ResamplerError::SizeError(format!("The input size {} must not exceed the FFT size {}", fftbuf.len(), self.fft_size)));
        }

        // 进行 FFT 正向变换
        self.fft_forward.process(&mut fftbuf);

        // 准备进行插值
        let mut fftdst = vec![Complex::<f64>{re: 0.0, im: 0.0}; self.fft_size];

        let half = self.fft_size / 2;
        let back = self.fft_size - 1;
        let scaling = desired_length as f64 / input_size as f64;
        if input_size > desired_length {
            // Input size exceeds output size, indicating audio compression.
            // This implies stretching in the frequency domain (scaling up).
            for i in 0..half {
                let scaled = i as f64 * scaling;
                let i1 = scaled.trunc() as usize;
                let i2 = i1 + 1;
                let s = scaled.fract();
                if INTERPOLATE_DNSCALE {
                    fftdst[i] = interpolate(fftbuf[i1], fftbuf[i2], s);
                    fftdst[back - i] = interpolate(fftbuf[back - i1], fftbuf[back - i2], s);
                } else {
                    fftdst[i] = fftbuf[i1];
                    fftdst[back - i] = fftbuf[back - i1];
                }
            }
        } else {
            // Input size is smaller than the output size, indicating audio stretching.
            // This implies compression in the frequency domain (scaling down).
            for i in 0..half {
                let i1 = (i as f64 * scaling).trunc() as usize;
                let i2 = ((i + 1) as f64 * scaling).trunc() as usize;
                if i2 >= half {break;}
                let j1 = back - i2;
                let j2 = back - i1;
                if INTERPOLATE_UPSCALE {
                    fftdst[i] = get_average(&fftbuf[i1..i2]);
                    fftdst[back - i] = get_average(&fftbuf[j1..j2]);
                } else {
                    fftdst[i] = fftbuf[i1];
                    fftdst[back - i] = fftbuf[back - i1];
                }
            }
        }

        self.fft_inverse.process(&mut fftdst);

        fftdst.truncate(desired_length);

        Ok(fftdst.into_iter().map(|c| -> f32 {(c.re * self.normalize_scaler) as f32}).collect())
    }

    /// The processing unit size should be adjusted to work in "chunks per second", 
    /// and artifacts will vanish when the chunk count aligns with the maximum infrasonic frequency.
    /// Calling `self.get_desired_length()` determines the processed chunk size calculated based on the target sample rate.
    pub fn get_process_size(&self, orig_size: usize, src_sample_rate: u32, dst_sample_rate: u32) -> usize {
        const MAX_INFRASOUND_FREQ: usize = 20;
        if src_sample_rate == dst_sample_rate {
            min(self.fft_size, orig_size)
        } else {
            min(self.fft_size, src_sample_rate as usize / MAX_INFRASOUND_FREQ)
        }
    }

    /// Get the processed chunk size calculated based on the target sample rate.
    pub fn get_desired_length(&self, proc_size: usize, src_sample_rate: u32, dst_sample_rate: u32) -> usize {
        min(self.fft_size, proc_size * dst_sample_rate as usize / src_sample_rate as usize)
    }

    pub fn resample(&self, input: &[f32], src_sample_rate: u32, dst_sample_rate: u32) -> Result<Vec<f32>, ResamplerError> {
        if src_sample_rate == dst_sample_rate {
            Ok(input.to_vec())
        } else {
            let proc_size = self.get_process_size(self.fft_size, src_sample_rate, dst_sample_rate);
            let desired_length = self.get_desired_length(proc_size, src_sample_rate, dst_sample_rate);
            if input.len() > proc_size {
                Err(ResamplerError::SizeError(format!("To resize the waveform, the input size should be {proc_size}, not {}", input.len())))
            } else if src_sample_rate > dst_sample_rate {
                // Source sample rate is higher than the target, indicating waveform compression.
                self.resample_core(input, desired_length)
            } else {
                // Source sample rate is lower than the target, indicating waveform stretching.
                // When the input length is less than the desired length, zero-padding is applied to the end.
                input.to_vec().resize(proc_size, 0.0);
                self.resample_core(input, desired_length)
            }
        }
    }

    pub fn get_fft_size(&self) -> usize {
        self.fft_size
    }
}

impl Debug for Resampler {
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        fmt.debug_struct("Resampler")
            .field("fft_forward", &format_args!("..."))
            .field("fft_inverse", &format_args!("..."))
            .field("fft_size", &self.fft_size)
            .field("normalize_scaler", &self.normalize_scaler)
            .finish()
    }
}
