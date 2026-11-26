//! HTTP router implementations

mod coupled_deepseek;
pub mod dp_utils;
mod ds41_sampling;
mod kimi_media;
pub mod logprobs_merge;
pub mod openai_router;
pub mod pd_router;
pub mod pd_types;
mod pd_usage;
mod program_adapter;
mod response_history;
pub mod router;
mod token_balance;
pub mod vllm_pd_router;
pub mod vllm_service_discovery;
