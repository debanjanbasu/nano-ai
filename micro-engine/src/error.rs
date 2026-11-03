#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Metal error: {0}")]
    Metal(#[from] micro_metal::Error),

    #[error("Invalid model format: {0}")]
    InvalidFormat(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The ANE failed mid-step; it is now off and the engine state was reset.
    #[error("ANE evaluation failed or produced non-finite values")]
    AneFailed,
}
