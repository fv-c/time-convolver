use crate::{
    analyze_audio, convolve_multichannel_time_domain_with_progress, load_audio,
    normalize_peak_in_place, write_wav_f32, AudioData, AudioStats, LoudnessReport,
};
use eframe::egui;
use eframe::egui::Widget;
use std::path::PathBuf;
use std::sync::mpsc;
use crate::GuiError;

#[derive(Debug, Clone, Copy, PartialEq)]
enum InputType {
    Input,
    Kernel,
}

#[derive(Debug, Clone)]
pub struct GuiState {
    pub input_path: Option<PathBuf>,
    pub kernel_path: Option<PathBuf>,
    pub output_path: Option<PathBuf>,
    pub in_gain_db: f32,
    pub kernel_gain_db: f32,
    pub out_gain_db: f32,
    pub normalize_peak: f32,
    pub target_lufs: f64,
    pub self_stages: u32,
    pub save_intermediate_stages: bool,
    pub report_true_peak: bool,
    pub input_audio: Option<AudioData>,
    pub kernel_audio: Option<AudioData>,
    pub processed_audio: Option<AudioData>,
    pub processing: bool,
    pub progress: f32,
    pub stats: Option<AudioStats>,
    pub loudness_report: Option<LoudnessReport>,
    pub error_message: Option<GuiError>,
    pub output_stats: Option<AudioStats>,
    pub convolution_result_tx: Option<mpsc::Sender<AudioData>>,
}

impl Default for GuiState {
    fn default() -> Self {
        Self {
            input_path: None,
            kernel_path: None,
            output_path: None,
            in_gain_db: 0.0,
            kernel_gain_db: 0.0,
            out_gain_db: 0.0,
            normalize_peak: 0.0,
            target_lufs: 0.0,
            self_stages: 0,
            save_intermediate_stages: false,
            report_true_peak: false,
            input_audio: None,
            kernel_audio: None,
            processed_audio: None,
            processing: false,
            progress: 0.0,
            stats: None,
            loudness_report: None,
            error_message: None,
            output_stats: None,
            convolution_result_tx: None,
        }
    }
}

impl GuiState {
    pub fn reset_audio(&mut self) {
        self.input_audio = None;
        self.kernel_audio = None;
        self.processed_audio = None;
        self.stats = None;
        self.output_stats = None;
        self.loudness_report = None;
    }

    pub fn load_input(&mut self, path: PathBuf) -> Result<(), GuiError> {
        self.reset_audio();
        self.input_path = Some(path.clone());

        match load_audio(&path) {
            Ok(audio) => {
                self.input_audio = Some(audio.clone());
                self.stats = Some(analyze_audio(&audio.data));
                Ok(())
            }
            Err(e) => Err(GuiError::LoadError(format!("Failed to load input: {}", e))),
        }
    }

    pub fn load_kernel(&mut self, path: PathBuf) -> Result<(), GuiError> {
        self.kernel_path = Some(path.clone());

        match load_audio(&path) {
            Ok(audio) => {
                self.kernel_audio = Some(audio.clone());
                self.stats = Some(analyze_audio(&audio.data));
                Ok(())
            }
            Err(e) => Err(GuiError::LoadError(format!("Failed to load kernel: {}", e))),
        }
    }

    pub fn validate_sample_rates(&self) -> Result<(), GuiError> {
        if let (Some(input), Some(kernel)) = (&self.input_audio, &self.kernel_audio) {
            if input.sample_rate != kernel.sample_rate {
                return Err(GuiError::SampleRateMismatch {
                    input_sr: input.sample_rate,
                    kernel_sr: kernel.sample_rate,
                });
            }
        }
        Ok(())
    }

}

pub fn run_gui(_args: &mut crate::Args) -> std::result::Result<(), eframe::Error> {
    let state = GuiState::default();

    let native_options = eframe::NativeOptions::default();

    eframe::run_native(
        "Time Convolver",
        native_options,
        Box::new(|_cc| {
            Ok(Box::new(GuiApp::new(state)))
        }),
    )
}

pub struct GuiApp {
    state: GuiState,
}

impl GuiApp {
    pub fn new(state: GuiState) -> Self {
        Self { state }
    }

    fn load_file(&mut self, file_type: InputType) {
        if let Some(path) = rfd::FileDialog::new()
            .set_title(match file_type {
                InputType::Input => "Select Input Audio",
                InputType::Kernel => "Select Kernel Audio",
            })
            .add_filter("Audio Files", &["wav", "mp3", "flac", "ogg"])
            .pick_file()
        {
            self.state.reset_audio();

            match file_type {
                InputType::Input => {
                    if let Err(e) = self.state.load_input(path.clone()) {
                        self.state.error_message = Some(GuiError::LoadError(e.to_string()));
                    }
                    self.state.output_path = Some(
                        path.with_file_name(format!("{}_convolved.wav", path.file_stem().unwrap_or_default().to_str().unwrap_or("output"))),
                    );
                }
                InputType::Kernel => {
                    if let Err(e) = self.state.load_kernel(path.clone()) {
                        self.state.error_message = Some(GuiError::LoadError(e.to_string()));
                    }
                    // Validate sample rates after loading kernel
                    let _ = self.state.validate_sample_rates();
                    if let Err(e) = self.state.validate_sample_rates() {
                        self.state.error_message = Some(GuiError::LoadError(e.to_string()));
                    }
                }
            }
        }
    }

    fn select_output_path(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .set_title("Select Output Location")
            .set_file_name("output.wav")
            .pick_file()
        {
            self.state.output_path = Some(path);
        }
    }

    fn run_convolution(&mut self) {
        if self.state.processing {
            return;
        }

        let input = match self.state.input_audio.clone() {
            Some(a) => a,
            None => {
                self.state.error_message = Some(GuiError::NoInput);
                return;
            }
        };

        let output_path = match &self.state.output_path {
            Some(p) => p.clone(),
            None => {
                self.state.error_message = Some(GuiError::NoOutputPath);
                return;
            }
        };

        let self_stages = self.state.self_stages;
        let save_intermediate = self.state.save_intermediate_stages;
        let kernel_audio = self.state.kernel_audio.clone();

        self.state.processing = true;
        self.state.progress = 0.0;
        self.state.error_message = None;

        // Run convolution in a background thread
        std::thread::spawn(move || {
            if self_stages > 0 {
                let mut current = input.clone();

                for stage_idx in 1..=self_stages {
                    let convolved = convolve_multichannel_time_domain_with_progress(&current, &current);
                    current = AudioData {
                        sample_rate: current.sample_rate,
                        channels: convolved.len(),
                        data: convolved,
                    };

                    if save_intermediate {
                        let stage_path = build_stage_output_path(&output_path, stage_idx);
                        let mut stage_data = current.data.clone();
                        normalize_peak_in_place(&mut stage_data, 0.999);
                        if let Err(e) = write_wav_f32(&stage_path, current.sample_rate, &stage_data) {
                            eprintln!("Failed to save stage {}: {}", stage_idx, e);
                        }
                    }
                }
            } else {
                let kernel = match kernel_audio {
                    Some(k) => k,
                    None => return,
                };

                let convolved = convolve_multichannel_time_domain_with_progress(&input, &kernel);
                let _result = AudioData {
                    sample_rate: input.sample_rate,
                    channels: convolved.len(),
                    data: convolved,
                };
            }
        });
    }

    fn format_stats(stats: &AudioStats) -> String {
        let mut s = format!("Global peak: {:.4}\n", stats.global_peak);
        for (i, ch) in stats.per_channel.iter().enumerate() {
            let peak_db = if ch.peak > 0.0 { 20.0 * ch.peak.log10() } else { f32::NEG_INFINITY };
            let rms_db = if ch.rms > 0.0 { 20.0 * ch.rms.log10() } else { f32::NEG_INFINITY };
            s.push_str(&format!("Ch {:>2}: peak={:.4} ({:.1} dBFS), rms={:.4} ({:.1} dBFS)\n",
                i, ch.peak, peak_db, ch.rms, rms_db));
        }
        s
    }

    fn format_loudness(rep: &LoudnessReport) -> String {
        let mut s = String::new();
        if rep.integrated_lufs.is_finite() {
            s.push_str(&format!("Integrated: {:.2} LUFS\n", rep.integrated_lufs));
        } else {
            s.push_str("Integrated: N/A\n");
        }
        if let Some(tp) = rep.true_peak_dbfs {
            if tp.is_finite() {
                s.push_str(&format!("True peak: {:.2} dBTP\n", tp));
            }
        }
        s
    }
}

fn build_stage_output_path(final_output_path: &std::path::Path, stage_idx: u32) -> std::path::PathBuf {
    let parent = final_output_path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let stem = final_output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let filename = format!("{}_stage_{:03}.wav", stem, stage_idx);
    parent.join(filename)
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.heading("Time Convolver GUI");
        });

        egui::SidePanel::left("sidebar").show(ctx, |ui| {
            ui.heading("Files");

            ui.horizontal(|ui| {
                ui.label("Input:");
                if ui.button("Load...").clicked() {
                    self.load_file(InputType::Input);
                }
            });

            if let Some(ref path) = self.state.input_path {
                ui.label(format!("{}", path.display()));
            } else {
                ui.label("(not loaded)");
            }

            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Kernel:");
                if ui.button("Load...").clicked() {
                    self.load_file(InputType::Kernel);
                }
            });

            if let Some(ref path) = self.state.kernel_path {
                ui.label(format!("{}", path.display()));
            } else {
                ui.label("(not loaded)");
            }

            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Output:");
                if ui.button("Pick...").clicked() {
                    self.select_output_path();
                }
            });

            if let Some(ref path) = self.state.output_path {
                ui.label(format!("{}", path.display()));
            } else {
                ui.label("(not set)");
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Parameters");

            ui.horizontal(|ui| {
                ui.label("Input Gain (dB):");
                ui.add(egui::Slider::new(&mut self.state.in_gain_db, -60.0..=60.0));
                ui.label(format!("{} dB", self.state.in_gain_db));
            });

            ui.horizontal(|ui| {
                ui.label("Kernel Gain (dB):");
                ui.add(egui::Slider::new(&mut self.state.kernel_gain_db, -60.0..=60.0));
                ui.label(format!("{} dB", self.state.kernel_gain_db));
            });

            ui.horizontal(|ui| {
                ui.label("Output Gain (dB):");
                ui.add(egui::Slider::new(&mut self.state.out_gain_db, -60.0..=60.0));
                ui.label(format!("{} dB", self.state.out_gain_db));
            });

            ui.separator();

            ui.label("Normalize Peak:");
            ui.add(egui::Slider::new(&mut self.state.normalize_peak, 0.1..=1.0).text("Target"));

            ui.horizontal(|ui| {
                ui.label("Target LUFS:");
                ui.add(egui::Slider::new(&mut self.state.target_lufs, -60.0..=0.0).text("Target"));
            });

            ui.separator();

            ui.label("Self-Convolution Stages:");
            ui.add(egui::Slider::new(&mut self.state.self_stages, 0..=10));

            ui.checkbox(&mut self.state.save_intermediate_stages, "Save Intermediate Stages");
            ui.checkbox(&mut self.state.report_true_peak, "Report True Peak");

            ui.separator();

            ui.horizontal_centered(|ui| {
                let input_loaded = self.state.input_audio.is_some();
                let output_set = self.state.output_path.is_some();
                let can_process = input_loaded && (self.state.self_stages > 0 || self.state.kernel_audio.is_some()) && output_set;

                if ui.add_enabled(can_process, egui::Button::new("Run Convolution")).clicked() {
                    self.run_convolution();
                }
            });

            if self.state.processing {
                ui.heading("Processing...");
                egui::ProgressBar::new(self.state.progress)
                    .show_percentage()
                    .ui(ui);
            }

            ui.separator();

            if let Some(ref err) = self.state.error_message {
                ui.colored_label(egui::Color32::RED, err.to_string());
            }

            ui.separator();

            ui.heading("Input Stats");
            if let Some(ref stats) = self.state.stats {
                ui.label(Self::format_stats(stats));
            } else {
                ui.label("(load an input file to see stats)");
            }

            ui.separator();

            ui.heading("Output Stats");
            if let Some(ref stats) = self.state.output_stats {
                ui.label(Self::format_stats(stats));
            } else {
                ui.label("(no output yet)");
            }

            if let Some(ref report) = self.state.loudness_report {
                ui.separator();
                ui.heading("Loudness Analysis");
                ui.label(Self::format_loudness(report));
            }
        });
    }
}
