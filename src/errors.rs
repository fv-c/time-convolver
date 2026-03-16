use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum GuiError {
    #[error("Failed to load file: {0}")]
    LoadError(String),
    #[error("Convolution error: {0}")]
    ConvolutionError(String),
    #[error("Write error: {0}")]
    WriteError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("No input audio loaded")]
    NoInput,
    #[error("No kernel audio loaded")]
    NoKernel,
    #[error("No output path set")]
    NoOutputPath,
    #[error("Sample rate mismatch: input {input_sr} Hz, kernel {kernel_sr} Hz")]
    SampleRateMismatch { input_sr: u32, kernel_sr: u32 },
}
