# Resampler for audio waveform processing

The purpose of this library is to help you change the sample rate of the audio file.
For example, you have a 22050 Hz sample rate audio file, but you want a 48000 Hz sample rate audio file, you need resampling.
Here is the resampler to help you losslessly stretch/compress the waveform to fit the sample rate.

## How to use

### First determine which FFT size fits you.

The best FFT size should be the minimum 2^n number of your sample rate. You have source sample rate and target sample rate, choose the largest.
For example, you want to stretch 22050 to 48000，the best FFT size should be 65536.

### Create the `Resampler` struct

```rust
let resampler = Resampler::new(65536);
```

### Ask the `resampler` for the best process size

```rust
let source_sample_rate = 22050;
let target_sample_rate = 48000;

let process_size = resampler.get_process_size(resampler.get_fft_size(), source_sample_rate, target_sample_rate);
```

### Do resampling in blocks. Each block size should be `process_size`

```rust
let source_audio = vec![0.0f32; 114514]; // Assume this is the waveform you want to resample
let target_audio = vec![0.0f32; 0];

let mut iter = source_audio.iter();
loop {
    let block: Vec<f32> = iter.by_ref().take(process_size).copied().collect();
    if block.is_empty() {
        break;
    }
    let block = resampler.resample(block, source_sample_rate, target_sample_rate).unwarp();
    target_audio.extend(block);
}
```

### After that, your `target_audio` is the resampled audio. No artifacts are introduced by the resampler.
