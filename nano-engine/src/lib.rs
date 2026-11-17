#![feature(portable_simd)]

pub mod ane_ffn;
pub mod artifact;
pub mod dspark;
pub mod error;
pub mod model;
mod prompt_cache;
pub mod quant;
pub mod sampler;
pub mod tokenizer;
pub mod vision;
#[cfg(test)]
mod tests;

pub use error::Error;
pub type Result<T, E = Error> = std::result::Result<T, E>;
