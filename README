# time_convolver

[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)]()
[![CLI](https://img.shields.io/badge/interface-command%20line-blue.svg)]()
[![DSP](https://img.shields.io/badge/domain-audio%20DSP-green.svg)]()
[![status](https://img.shields.io/badge/status-experimental-yellow.svg)]()

**time_convolver** is a Rust command-line tool implementing **direct time-domain convolution** between audio signals.

Unlike most convolution engines that rely on FFT partitioning, this project performs **explicit sample-domain convolution**. The goal is not maximum performance, but **algorithmic transparency, experimental flexibility, and research usability**.

The tool supports:

- direct convolution between audio files
- multi-channel audio
- automatic sample-rate alignment
- LUFS loudness analysis
- peak and loudness normalization
- recursive **staged self-convolution**

---

# Quick Start

Build the program:

```bash
cargo build --release
```

Run a standard convolution:

```bash
./target/release/time_convolver \
--input dry.wav \
--kernel ir.wav \
--output wet.wav
```

Run recursive self-convolution:

```bash
./target/release/time_convolver \
--input source.wav \
--output result.wav \
--self-stages 3
```

---

# Why Time-Domain Convolution?

Most DSP software implements convolution via FFT:

```
FFT(x) × FFT(h)
```

This project instead implements the **mathematical definition of convolution**:

```
y[n] = Σ x[k] · h[n − k]
```

Advantages:

- completely transparent algorithm
- deterministic sample-domain behavior
- easier experimentation
- supports recursive convolution processes

Tradeoff:

- computationally expensive

Time complexity:

```
O(N × M)
```

---

# Core Features

## DSP Engine

- direct convolution (no FFT)
- cubic time-domain resampling
- multi-channel audio support
- 32-bit floating-point output

## Audio Analysis

- peak detection
- RMS measurement
- integrated loudness (LUFS)
- optional true peak reporting

## Gain & Normalization

- input gain
- kernel gain
- output gain
- peak normalization
- loudness normalization

## Experimental Processing

- recursive staged self-convolution
- automatic export of intermediate stages

---

# Processing Architecture

Standard convolution pipeline:

```
Input signal
      │
      ▼
Load audio
      │
      ▼
Apply input gain
      │
      ▼
Load kernel
      │
      ▼
Sample-rate alignment
      │
      ▼
Direct time-domain convolution
      │
      ▼
Peak & LUFS analysis
      │
      ▼
Optional normalization
      │
      ▼
Write output WAV
```

Self-convolution pipeline:

```
Input
 │
 ▼
Stage 1: x * x
 │
 ▼
Stage 2: stage1 * stage1
 │
 ▼
Stage 3: stage2 * stage2
 │
 ▼
(optional intermediate files)
 │
 ▼
Final output
```

---

# Command Line Interface

## Required arguments

### Input

```
--input <FILE>
```

Path to the source audio file.

### Output

```
--output <FILE>
```

Path to the final rendered WAV file.

---

## Standard Convolution

```
--kernel <FILE>
```

Impulse response or convolution kernel.

Example:

```
--kernel cathedral_ir.wav
```

---

## Gain Controls

```
--in-gain-db
--kernel-gain-db
--out-gain-db
```

Gain values in decibels.

Example:

```
--kernel-gain-db -12
```

---

## Normalization

### Peak normalization

```
--normalize-peak <VALUE>
```

Example:

```
--normalize-peak 0.999
```

---

### Loudness normalization

```
--target-lufs <VALUE>
```

Typical targets:

| Context | LUFS |
|------|------|
Broadcast | -23 |
Podcast | -16 |
Streaming | -14 |

Example:

```
--target-lufs -16
```

---

## Loudness Reporting

```
--report-true-peak
```

Outputs estimated true peak levels.

---

# Self-Convolution Mode

Recursive self-convolution can be enabled using:

```
--self-stages N
```

Example:

```
--self-stages 3
```

Processing sequence:

```
Stage 1 = x * x
Stage 2 = stage1 * stage1
Stage 3 = stage2 * stage2
```

This produces extremely rich and rapidly expanding signals.

---

## Saving Intermediate Stages

```
--save-intermediate-stages
```

Intermediate files will be exported as:

```
output_stage_001.wav
output_stage_002.wav
output_stage_003.wav
```

Intermediate files are peak-normalized to remain listenable.

---

# Multi-Channel Behaviour

The output channel count is determined as:

```
max(input_channels, kernel_channels)
```

Channels are matched cyclically.

Examples:

| Input | Kernel | Output |
|------|------|------|
mono | stereo | stereo |
stereo | mono | stereo |
5ch | stereo | 5ch |

This is not a spatial routing matrix, but a simple and robust mapping strategy.

---

# Supported Formats

Audio decoding uses **Symphonia**, supporting formats such as:

- WAV
- FLAC
- MP3
- OGG
- AIFF
- AAC
- ALAC

Output format:

```
WAV
32-bit floating point
```

---

# Example Workflows

## Convolution Reverb

```
time_convolver \
--input voice.wav \
--kernel cathedral_ir.wav \
--output voice_reverb.wav
```

---

## Experimental Kernel Convolution

```
time_convolver \
--input texture.wav \
--kernel noise.wav \
--output filtered.wav
```

---

## Loudness-normalized output

```
time_convolver \
--input dry.wav \
--kernel ir.wav \
--target-lufs -16 \
--report-true-peak \
--output wet.wav
```

---

## Recursive Convolution Experiment

```
time_convolver \
--input sound.wav \
--self-stages 3 \
--save-intermediate-stages \
--output recursion.wav
```

---

# Limitations

## Computational cost

Direct convolution has complexity:

```
O(N × M)
```

Long signals or impulse responses can become slow.

---

## Signal growth

Self-convolution causes rapid growth in:

- signal length
- energy
- memory usage

Large numbers of stages can become impractical.

---

## Resampling quality

Current implementation uses cubic interpolation.  
Higher quality SRC methods may be implemented in future versions.

---

# Project Structure

Current layout:

```
src/
 └─ main.rs
```

Future modular architecture:

```
src/
 ├─ cli.rs
 ├─ audio_io.rs
 ├─ convolution.rs
 ├─ resampling.rs
 ├─ analysis.rs
 └─ self_convolution.rs
```

---

# Future Development

## DSP improvements

- block convolution
- optional FFT partitioned mode
- multithreaded convolution
- streaming processing
- higher-quality resampling

## Analysis tools

- limiter stage
- LUFS + true peak constrained normalization
- CSV / JSON analysis reports

## Architecture

- modular refactor
- automated tests
- performance benchmarks
- improved progress reporting

---

# Research Context

This tool intentionally exposes convolution in its **most explicit algorithmic form**.

It is therefore particularly useful for:

- DSP experimentation
- recursive signal processing
- algorithmic sound design
- educational demonstrations of convolution

If you require **real-time convolution with long impulse responses**, an FFT-partitioned convolver will generally be more appropriate.

---