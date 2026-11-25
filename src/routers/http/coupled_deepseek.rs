//! DeepSeek's shared HTTP frontend for a single pool of coupled workers.
use super::{kimi_media::MediaPool, response_history::ResponseHistory};
use crate::config::HistoryBackend;
use crate::protocols::spec::{ChatMessage, ContentPart, UserMessageContent};
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct CoupledDeepseek {
    pub media: MediaPool,
    pub history: Option<Arc<ResponseHistory>>,
}

impl CoupledDeepseek {
    pub async fn from_env(history: HistoryBackend) -> Result<Option<Self>, String> {
        match std::env::var("VLLM_ROUTER_DS41_COUPLED_FRONTEND").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("0") => return Ok(None),
            Ok("1") => {}
            _ => return Err("VLLM_ROUTER_DS41_COUPLED_FRONTEND must be 0 or 1".into()),
        }
        let media = MediaPool::from_env()
            .await?
            .filter(|pool| {
                pool.ds_request(&serde_json::json!({"model":"deepseek-v4.1-flash-1017"}))
            })
            .ok_or("DeepSeek coupled frontend requires VLLM_ROUTER_DS41_IMAGE_WORKERS > 0")?;
        let history =
            matches!(history, HistoryBackend::Memory).then(|| Arc::new(ResponseHistory::default()));
        Ok(Some(Self { media, history }))
    }
}

// Text requests can use the typed fast path without a second copy of the prompt.
pub(super) fn may_have_media(message: &ChatMessage) -> bool {
    let content = match message {
        ChatMessage::User { content, .. } | ChatMessage::System { content, .. } => Some(content),
        ChatMessage::Assistant { content, .. } => content.as_ref(),
        ChatMessage::Tool { content, .. } => return content.is_array() || content.is_object(),
        ChatMessage::Other(value) => return !value["content"].is_string(),
        ChatMessage::Function { .. } => None,
    };
    matches!(content, Some(UserMessageContent::Parts(parts)) if parts.iter().any(|p| !matches!(p, ContentPart::Text { .. })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn chat_fast_path_only_skips_text() {
        for content in [json!("hello"), json!([{"type":"text","text":"hello"}])] {
            let user = serde_json::from_value(json!({"role":"user","content":content})).unwrap();
            assert!(!may_have_media(&user));
        }
        for part in [
            json!({"type":"image_url","image_url":{"url":"data:image/png;base64,xyz"}}),
            json!({"type":"video_url","video_url":{"url":"https://example.invalid/video"}}),
        ] {
            let user = serde_json::from_value(json!({"role":"user","content":[part]})).unwrap();
            assert!(may_have_media(&user));
        }
    }
}
