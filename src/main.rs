use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use ebur128::{EbuR128, Mode};
use hound::{SampleFormat, WavSpec, WavWriter};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::path::{Path, PathBuf};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

mod gui;
mod errors;

pub use errors::GuiError;

#[derive(Subcommand, Debug, Clone)]
enum Command {
    /// Launch the GUI application
    Gui,
}

#[derive(Parser, Debug)]
#[command(version, about, arg_required_else_help = true)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    #[arg(long, value_name = "INPUT")]
    input: Option<PathBuf>,

    #[arg(long, value_name = "OUTPUT")]
    output: Option<PathBuf>,

    #[arg(long, value_name = "KERNEL")]
    kernel: Option<PathBuf>,

    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    in_gain_db: f32,

    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    kernel_gain_db: f32,

    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    out_gain_db: f32,

    #[arg(long)]
    normalize_peak: Option<f32>,

    #[arg(long, allow_hyphen_values = true)]
    target_lufs: Option<f64>,

    /// Se impostato, ignora il kernel e convolge il file input con se stesso
    /// per il numero di stadi richiesto. Ogni stadio usa come nuovo input il
    /// risultato dello stadio precedente convolto di nuovo con se stesso.
    #[arg(long)]
    self_stages: Option<u32>,

    #[arg(long, default_value_t = false)]
    save_intermediate_stages: bool,

    #[arg(long, default_value_t = false)]
    report_true_peak: bool,
}

#[derive(Debug, Clone)]
struct AudioData {
    sample_rate: u32,
    channels: usize,
    data: Vec<Vec<f32>>,
}

impl AudioData {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[derive(Debug, Clone)]
struct ChannelStats {
    peak: f32,
    rms: f32,
}

#[derive(Debug, Clone)]
struct AudioStats {
    per_channel: Vec<ChannelStats>,
    global_peak: f32,
}

#[derive(Debug, Clone)]
struct LoudnessReport {
    integrated_lufs: f64,
    true_peak_dbfs: Option<f64>,
}

fn main() -> anyhow::Result<()> {
    let mut args = Args::parse();

    // Check if GUI mode is requested
    if matches!(args.command, Some(Command::Gui)) {
        return Ok(gui::run_gui(&mut args)?);
    }

    print_usage_examples();

    // Validate required CLI arguments
    let input_path = args.input.ok_or_else(|| anyhow!("--input is required"))?;
    let output_path = args.output.ok_or_else(|| anyhow!("--output is required"))?;

    println!("Caricamento file...");

    let mut a = load_audio(&input_path)?;

    println!("Input  : sr={} ch={} len={}", a.sample_rate, a.channels, a.data[0].len());
    print_audio_stats("Input originale", &analyze_audio(&a.data));

    if args.in_gain_db != 0.0 {
        apply_gain_db(&mut a.data, args.in_gain_db);
    }

    let processed = if let Some(stages) = args.self_stages {
        if args.kernel_gain_db != 0.0 {
            println!("Avviso: --kernel-gain-db è ignorato in modalità self-convolution.");
        }

        println!("Modalità selezionata: self-convolution a stadi");
        run_self_convolution_stages(
            &a,
            stages,
            &output_path,
            args.save_intermediate_stages,
        )?
    } else {
        let kernel_path = args
            .kernel
            .as_ref()
            .ok_or_else(|| anyhow!("In modalità standard devi specificare --kernel <FILE>"))?;

        let mut b = load_audio(kernel_path)?;

        println!("Kernel : sr={} ch={} len={}", b.sample_rate, b.channels, b.data[0].len());
        print_audio_stats("Kernel originale", &analyze_audio(&b.data));

        if args.kernel_gain_db != 0.0 {
            apply_gain_db(&mut b.data, args.kernel_gain_db);
        }

        let target_sr = a.sample_rate.max(b.sample_rate);

        if a.sample_rate != target_sr {
            a = resample_audio(&a, target_sr)?;
        }

        if b.sample_rate != target_sr {
            b = resample_audio(&b, target_sr)?;
        }

        println!("Convoluzione time-domain in corso...");
        let convolved = convolve_multichannel_time_domain_with_progress(&a, &b);

        AudioData {
            sample_rate: target_sr,
            channels: convolved.len(),
            data: convolved,
        }
    };

    let target_sr = processed.sample_rate();
    let mut out = processed.data;

    print_audio_stats("Output dopo convoluzione (raw)", &analyze_audio(&out));

    let loud_before = measure_loudness(&out, target_sr, args.report_true_peak)?;
    print_loudness_report("Loudness output raw", &loud_before);

    if let Some(target_peak) = args.normalize_peak {
        normalize_peak_in_place(&mut out, target_peak);
    }

    if let Some(target_lufs) = args.target_lufs {
        normalize_to_target_lufs(&mut out, target_sr, target_lufs)?;
    }

    if args.out_gain_db != 0.0 {
        apply_gain_db(&mut out, args.out_gain_db);
    }

    print_audio_stats("Output finale", &analyze_audio(&out));

    let loud_after = measure_loudness(&out, target_sr, args.report_true_peak)?;
    print_loudness_report("Loudness output finale", &loud_after);

    write_wav_f32(&output_path, target_sr, &out)?;

    println!("Fatto: {}", output_path.display());

    Ok(())
}

fn run_self_convolution_stages(
    input: &AudioData,
    stages: u32,
    final_output_path: &Path,
    save_intermediate_stages: bool,
) -> Result<AudioData> {
    if stages == 0 {
        return Ok(input.clone());
    }

    println!(
        "Self-convolution a stadi: {} stadi (ogni stadio convolge il risultato con se stesso)",
        stages
    );

    let mut current = input.clone();

    for stage_idx in 1..=stages {
        println!("Stadio {} / {} in corso...", stage_idx, stages);

        let convolved = convolve_multichannel_time_domain_with_progress(&current, &current);

        // Normalizza per evitare overflow dopo multiple convoluzioni
        let mut normalized_data: Vec<Vec<f32>> = Vec::with_capacity(convolved.len());
        for ch in &convolved {
            let mut normalized_ch = ch.clone();
            normalize_peak_single_channel(&mut normalized_ch, 0.9);
            normalized_data.push(normalized_ch);
        }

        current = AudioData {
            sample_rate: current.sample_rate,
            channels: normalized_data.len(),
            data: normalized_data,
        };

        if save_intermediate_stages {
            let stage_path = build_stage_output_path(final_output_path, stage_idx);
            write_wav_f32(&stage_path, current.sample_rate, &current.data)?;
            println!("Salvato stadio intermedio: {}", stage_path.display());
        }
    }

    Ok(current)
}

fn build_stage_output_path(final_output_path: &Path, stage_idx: u32) -> PathBuf {
    let parent = final_output_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = final_output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let filename = format!("{}_stage_{:03}.wav", stem, stage_idx);
    parent.join(filename)
}

fn load_audio(path: &Path) -> Result<AudioData> {

    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;

    let track = format.default_track().ok_or(anyhow!("No audio track"))?;
    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let sample_rate = codec_params.sample_rate.unwrap();
    let channels = codec_params.channels.unwrap().count();

    let mut decoder = symphonia::default::get_codecs().make(
        &codec_params,
        &DecoderOptions::default(),
    )?;

    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    let mut out = vec![Vec::<f32>::new(); channels];

    loop {

        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break,
            Err(e) => return Err(anyhow!("Decode error: {e}")),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };

        if sample_buf.is_none() {

            let spec = *decoded.spec();
            let duration = decoded.capacity() as u64;

            sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
        }

        let buf = sample_buf.as_mut().unwrap();
        buf.copy_interleaved_ref(decoded);

        let samples = buf.samples();

        let frames = samples.len() / channels;

        for f in 0..frames {
            for ch in 0..channels {
                out[ch].push(samples[f * channels + ch]);
            }
        }
    }

    Ok(AudioData {
        sample_rate,
        channels,
        data: out,
    })
}

fn resample_audio(audio: &AudioData, target_sr: u32) -> Result<AudioData> {
    use rubato::{FftFixedIn, Resampler};

    let len = audio.data[0].len();
    let mut resampler = FftFixedIn::<f32>::new(
        audio.sample_rate as usize,
        target_sr as usize,
        len,
        1,
        1,
    )?;

    let input_vec: Vec<Vec<f32>> = audio.data.clone();
    let resampled = resampler.process(&input_vec, None)?;

    Ok(AudioData {
        sample_rate: target_sr,
        channels: audio.channels,
        data: resampled,
    })
}

fn convolve_multichannel_time_domain_with_progress(
    a: &AudioData,
    b: &AudioData,
) -> Vec<Vec<f32>> {

    let out_channels = a.channels.max(b.channels);

    let total_work: u64 = (0..out_channels)
        .map(|ch| {
            let x = a.data[ch % a.channels].len() as u64;
            let h = b.data[ch % b.channels].len() as u64;
            x * h
        })
        .sum();

    let pb = ProgressBar::new(total_work);

    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {percent:>3}% {pos}/{len}",
        )
        .unwrap(),
    );

    let mut out = Vec::with_capacity(out_channels);

    for ch in 0..out_channels {

        let x = &a.data[ch % a.channels];
        let h = &b.data[ch % b.channels];

        out.push(convolve_direct_with_progress(x, h, &pb));
    }

    pb.finish();

    out
}

fn convolve_direct_with_progress(
    x: &[f32],
    h: &[f32],
    pb: &ProgressBar,
) -> Vec<f32> {

    let mut y = vec![0.0; x.len() + h.len() - 1];

    for (n, &xn) in x.iter().enumerate() {

        for (k, &hk) in h.iter().enumerate() {
            y[n + k] += xn * hk;
        }

        pb.inc(h.len() as u64);
    }

    y
}

fn measure_loudness(
    channels: &[Vec<f32>],
    sr: u32,
    want_tp: bool,
) -> Result<LoudnessReport> {

    if channels.is_empty() || channels[0].is_empty() {
        return Ok(LoudnessReport {
            integrated_lufs: f64::NEG_INFINITY,
            true_peak_dbfs: if want_tp { Some(f64::NEG_INFINITY) } else { None },
        });
    }

    let frames = channels[0].len();

    let mode = if want_tp { Mode::I | Mode::TRUE_PEAK } else { Mode::I };

    let mut meter = EbuR128::new(channels.len() as u32, sr, mode)?;

    let mut interleaved = Vec::with_capacity(frames * channels.len());

    for i in 0..frames {
        for ch in channels {
            interleaved.push(ch[i]);
        }
    }

    meter.add_frames_f32(&interleaved)?;

    let lufs = meter.loudness_global()?;

    let tp = if want_tp {

        let mut max = f64::NEG_INFINITY;

        for ch in 0..channels.len() {

            let v = meter.true_peak(ch as u32)?;
            let db = 20.0 * v.log10();

            if db > max {
                max = db;
            }
        }

        Some(max)

    } else {
        None
    };

    Ok(LoudnessReport {
        integrated_lufs: lufs,
        true_peak_dbfs: tp,
    })
}

fn normalize_to_target_lufs(
    channels: &mut [Vec<f32>],
    sr: u32,
    target: f64,
) -> Result<()> {

    let report = measure_loudness(channels, sr, false)?;

    if !report.integrated_lufs.is_finite() {
        return Ok(());
    }

    let delta = target - report.integrated_lufs;
    let gain = 10f32.powf((delta as f32) / 20.0);

    for ch in channels {
        for s in ch {
            *s *= gain;
        }
    }

    Ok(())
}

fn analyze_audio(channels: &[Vec<f32>]) -> AudioStats {

    let mut per_channel = Vec::new();
    let mut global_peak: f32 = 0.0;

    for ch in channels {

        let mut peak: f32 = 0.0;
        let mut sum: f64 = 0.0;

        for &s in ch {

            peak = peak.max(s.abs());
            sum += (s * s) as f64;
        }

        let rms = if ch.is_empty() {
            0.0
        } else {
            (sum / ch.len() as f64).sqrt() as f32
        };

        global_peak = global_peak.max(peak);

        per_channel.push(ChannelStats { peak, rms });
    }

    AudioStats {
        per_channel,
        global_peak,
    }
}

fn apply_gain_db(channels: &mut [Vec<f32>], db: f32) {

    let gain = 10f32.powf(db / 20.0);

    for ch in channels {
        for s in ch {
            *s *= gain;
        }
    }
}

fn normalize_peak_in_place(channels: &mut [Vec<f32>], target: f32) {

    let peak = analyze_audio(channels).global_peak;

    if peak > 0.0 {

        let gain = target / peak;

        for ch in channels {
            for s in ch {
                *s *= gain;
            }
        }
    }
}

fn normalize_peak_single_channel(data: &mut Vec<f32>, target: f32) {
    let peak = data.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

    if peak > 0.0 {
        let gain = target / peak;
        for s in data {
            *s *= gain;
        }
    }
}

fn print_audio_stats(label: &str, stats: &AudioStats) {

    println!("\n=== {} ===", label);
    println!("Global peak: {:.6}", stats.global_peak);

    for (i, ch) in stats.per_channel.iter().enumerate() {

        let peak_db = if ch.peak > 0.0 {
            20.0 * ch.peak.log10()
        } else {
            f32::NEG_INFINITY
        };
        let rms_db = if ch.rms > 0.0 {
            20.0 * ch.rms.log10()
        } else {
            f32::NEG_INFINITY
        };

        println!(
            "Ch {:>2}: peak={:.6} ({:.2} dBFS), rms={:.6} ({:.2} dBFS)",
            i, ch.peak, peak_db, ch.rms, rms_db
        );
    }
}

fn print_loudness_report(label: &str, rep: &LoudnessReport) {
    println!("\n=== {} ===", label);

    if rep.integrated_lufs.is_finite() {
        println!("Integrated loudness: {:.2} LUFS", rep.integrated_lufs);
    } else {
        println!("Integrated loudness: non disponibile (segnale troppo corto o non valido)");
    }

    if let Some(tp) = rep.true_peak_dbfs {
        if tp.is_finite() {
            println!("True peak: {:.2} dBTP", tp);
        } else {
            println!("True peak: non disponibile");
        }
    }
}

fn write_wav_f32(path: &Path, sr: u32, channels: &[Vec<f32>]) -> Result<()> {

    let spec = WavSpec {
        channels: channels.len() as u16,
        sample_rate: sr,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut writer = WavWriter::create(path, spec)?;

    let frames = channels[0].len();

    for i in 0..frames {
        for ch in channels {
            writer.write_sample(ch[i])?;
        }
    }

    writer.finalize()?;

    Ok(())
}

fn print_usage_examples() {
    println!("Esempi:");
    println!("  time_convolver --input dry.wav --kernel ir.wav --output wet.wav");
    println!("  time_convolver --input dry.wav --output self.wav --self-stages 3 --save-intermediate-stages");
}