//! Resolve Responses IDs at the PD ingress. P and D run independent API
//! processes and the publicly returned response ID belongs to D.
use axum::{
    body::{to_bytes, Body, Bytes},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures::{io::AsyncBufReadExt, stream, TryStreamExt};
use lru::LruCache;
use parking_lot::Mutex;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{io, num::NonZeroUsize, sync::Arc};

const MAX_HISTORY_BYTES: usize = 64 << 20;
const MAX_ENTRIES: usize = 4096;
type Scope = [u8; 32];

#[derive(Debug)]
struct Cache {
    entries: LruCache<(Scope, String), Bytes>,
    bytes: usize,
}

#[derive(Debug)]
pub(super) struct ResponseHistory(Mutex<Cache>);

impl Default for ResponseHistory {
    fn default() -> Self {
        Self(Mutex::new(Cache {
            entries: LruCache::new(NonZeroUsize::new(MAX_ENTRIES).unwrap()),
            bytes: 0,
        }))
    }
}

type RequestError = (StatusCode, Json<Value>);

fn invalid(status: StatusCode, param: &str, message: &str) -> RequestError {
    (
        status,
        Json(json!({"error":{"type":"invalid_request_error","param":param,"message":message}})),
    )
}

fn input_items(input: Value) -> Result<Vec<Value>, RequestError> {
    match input {
        Value::Array(items) => Ok(items),
        Value::String(text) => Ok(vec![json!({"role":"user","content":text})]),
        _ => Err(invalid(
            StatusCode::BAD_REQUEST,
            "input",
            "Expected a string or an array of input items.",
        )),
    }
}

impl ResponseHistory {
    pub(super) fn prepare(
        self: &Arc<Self>,
        headers: Option<&HeaderMap>,
        body: &mut Value,
    ) -> Result<Option<Turn>, RequestError> {
        let store = body.get("store").is_none_or(|v| v.as_bool() == Some(true));
        if !store && body["previous_response_id"].is_null() {
            return Ok(None);
        }
        let mut hash = Sha256::new();
        for name in ["authorization", "x-api-key"] {
            let value = headers
                .and_then(|h| h.get(name))
                .map_or(&[][..], |v| v.as_bytes());
            hash.update(value.len().to_be_bytes());
            hash.update(value);
        }
        let scope: Scope = hash.finalize().into();
        let previous = match body.get("previous_response_id") {
            None | Some(Value::Null) => None,
            Some(Value::String(id)) => Some(id.clone()),
            _ => {
                return Err(invalid(
                    StatusCode::BAD_REQUEST,
                    "previous_response_id",
                    "Expected a response ID string.",
                ))
            }
        };
        if let Some(id) = &previous {
            if body["previous_input_messages"]
                .as_array()
                .is_some_and(|messages| !messages.is_empty())
                || !body["conversation"].is_null()
            {
                return Err(invalid(
                    StatusCode::BAD_REQUEST,
                    "previous_response_id",
                    "Conflicting conversation history parameters.",
                ));
            }
            let saved = self
                .0
                .lock()
                .entries
                .get(&(scope, id.clone()))
                .cloned()
                .ok_or_else(|| {
                    invalid(
                        StatusCode::NOT_FOUND,
                        "previous_response_id",
                        &format!("Response with id '{id}' not found."),
                    )
                })?;
            // Entries are serialized by remember(). Each snapshot is complete
            // so evicting or branching from a parent cannot truncate a later turn.
            let mut history: Vec<Value> =
                serde_json::from_slice(&saved).expect("serialized response history");
            history.extend(input_items(body["input"].take())?);
            body["input"] = Value::Array(history);
            body.as_object_mut().unwrap().remove("previous_response_id");
        }
        let input = if store {
            Some(input_items(body["input"].clone())?)
        } else {
            None
        };
        Ok(Some(Turn {
            cache: self.clone(),
            scope,
            previous,
            input,
        }))
    }

    fn remember(&self, scope: Scope, id: &str, items: &[Value]) {
        let bytes = Bytes::from(serde_json::to_vec(items).expect("JSON input/output items"));
        let size = bytes.len() + id.len();
        let mut cache = self.0.lock();
        if size > MAX_HISTORY_BYTES {
            if let Some(((_, old_id), old)) = cache.entries.pop_entry(&(scope, id.to_owned())) {
                cache.bytes -= old.len() + old_id.len();
            }
            return;
        }
        if let Some(((_, old_id), old)) = cache.entries.push((scope, id.to_owned()), bytes) {
            cache.bytes -= old.len() + old_id.len();
        }
        cache.bytes += size;
        while cache.bytes > MAX_HISTORY_BYTES {
            let ((_, old_id), old) = cache.entries.pop_lru().unwrap();
            cache.bytes -= old.len() + old_id.len();
        }
    }
}

pub(super) struct Turn {
    cache: Arc<ResponseHistory>,
    scope: Scope,
    previous: Option<String>,
    input: Option<Vec<Value>>,
}

impl Turn {
    fn observe(&mut self, response: &mut Value, terminal: bool) {
        if response["object"] != "response" {
            return;
        }
        if let Some(parent) = &self.previous {
            response["previous_response_id"] = json!(parent);
        }
        if terminal
            && matches!(
                response["status"].as_str(),
                Some("completed" | "incomplete")
            )
        {
            if let (Some(id), Some(output), Some(mut history)) = (
                response["id"].as_str(),
                response["output"].as_array(),
                self.input.take(),
            ) {
                history.extend_from_slice(output);
                self.cache.remember(self.scope, id, &history);
            }
        }
    }

    // The vLLM encoder writes one compact JSON data line per SSE event.
    // Read whole events across arbitrary HTTP chunks, and save the final ID
    // before the caller can observe response.completed and start a follow-up.
    fn observe_event(&mut self, frame: &mut Vec<u8>) {
        if let Some(event) = frame
            .split(|b| *b == b'\n')
            .next()
            .and_then(|line| line.strip_prefix(b"event:"))
        {
            match event.trim_ascii() {
                b"response.completed" | b"response.incomplete" => {}
                b"response.created"
                | b"response.in_progress"
                | b"response.failed"
                | b"response.queued"
                    if self.previous.is_some() => {}
                _ => return,
            }
        }
        let mut offset = 0;
        for line in frame.split_inclusive(|b| *b == b'\n') {
            if let Some(data) = line.strip_prefix(b"data:") {
                if let Ok(mut event) = serde_json::from_slice::<Value>(data) {
                    let terminal = matches!(
                        event["type"].as_str(),
                        Some("response.completed" | "response.incomplete")
                    );
                    if let Some(response) = event.get_mut("response") {
                        self.observe(response, terminal);
                        if self.previous.is_some() {
                            let mut replacement = b"data: ".to_vec();
                            serde_json::to_writer(&mut replacement, &event).unwrap();
                            replacement.push(b'\n');
                            frame.splice(offset..offset + line.len(), replacement);
                        }
                    }
                }
                break;
            }
            offset += line.len();
        }
    }

    pub(super) async fn finish(mut self, response: Response) -> Response {
        if !response.status().is_success() {
            return response;
        }
        let (mut parts, body) = response.into_parts();
        parts.headers.remove(header::CONTENT_LENGTH);
        let is_sse = parts
            .headers
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("text/event-stream"));
        if is_sse {
            let reader = futures::io::BufReader::new(
                body.into_data_stream()
                    .map_err(io::Error::other)
                    .into_async_read(),
            );
            let events = stream::try_unfold((reader, self), |(mut reader, mut turn)| async move {
                let mut frame = Vec::new();
                loop {
                    let start = frame.len();
                    let len = reader.read_until(b'\n', &mut frame).await?;
                    if len == 0 || matches!(&frame[start..], b"\n" | b"\r\n") {
                        if frame.is_empty() {
                            return Ok::<_, io::Error>(None);
                        }
                        if len != 0 {
                            turn.observe_event(&mut frame);
                        }
                        return Ok(Some((Bytes::from(frame), (reader, turn))));
                    }
                }
            });
            return Response::from_parts(parts, Body::from_stream(events));
        }
        let bytes = match to_bytes(body, usize::MAX).await {
            Ok(bytes) => bytes,
            Err(error) => return (StatusCode::BAD_GATEWAY, error.to_string()).into_response(),
        };
        let mut value: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(error) => return (StatusCode::BAD_GATEWAY, error.to_string()).into_response(),
        };
        self.observe(&mut value, true);
        let result = if self.previous.is_some() {
            Bytes::from(serde_json::to_vec(&value).unwrap())
        } else {
            bytes
        };
        Response::from_parts(parts, Body::from(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stream_publishes_completed_context_before_eof() {
        let history = Arc::new(ResponseHistory::default());
        let turn = history
            .prepare(None, &mut json!({"input":"hello"}))
            .unwrap()
            .unwrap();
        let (tx, rx) = futures::channel::mpsc::unbounded::<Result<Bytes, io::Error>>();
        let response = Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(Body::from_stream(rx))
            .unwrap();
        let mut stream = turn.finish(response).await.into_body().into_data_stream();
        let follow = || json!({"previous_response_id":"resp_a","input":"next","store":false});
        tx.unbounded_send(Ok(Bytes::from_static(b"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"object\":\"response\",\"id\":\"resp_a\",\"status\":\"in_progress\"}}\n\n"))).unwrap();
        assert!(stream
            .try_next()
            .await
            .unwrap()
            .unwrap()
            .starts_with(b"event: response.created"));
        assert!(history.prepare(None, &mut follow()).is_err());
        tx.unbounded_send(Ok(Bytes::from_static(b"event: response.completed\r\ndata: {\"type\":\"response.completed\",\"response\":{\"object\":\"response\",\"id\":\"resp_a\",\"sta"))).unwrap();
        tx.unbounded_send(Ok(Bytes::from_static(b"tus\":\"completed\",\"output\":[{\"type\":\"function_call\",\"call_id\":\"call_1\"}]}}\r\n\r\n"))).unwrap();
        assert!(stream
            .try_next()
            .await
            .unwrap()
            .unwrap()
            .starts_with(b"event: response.completed"));
        let mut next = follow();
        history.prepare(None, &mut next).unwrap();
        assert_eq!(
            next["input"][1],
            json!({"type":"function_call","call_id":"call_1"})
        );
    }

    #[test]
    fn evicting_a_parent_does_not_truncate_its_child() {
        let history = Arc::new(ResponseHistory(Mutex::new(Cache {
            entries: LruCache::new(NonZeroUsize::new(1).unwrap()),
            bytes: 0,
        })));
        let mut first = history
            .prepare(None, &mut json!({"input":"one"}))
            .unwrap()
            .unwrap();
        first.observe(
            &mut json!({"object":"response","id":"one","status":"completed","output":[]}),
            true,
        );
        let mut second = history
            .prepare(
                None,
                &mut json!({"input":"two","previous_response_id":"one"}),
            )
            .unwrap()
            .unwrap();
        second.observe(
            &mut json!({"object":"response","id":"two","status":"incomplete","output":[]}),
            true,
        );
        assert!(history
            .prepare(
                None,
                &mut json!({"input":"other","previous_response_id":"one"})
            )
            .is_err());
        let mut last = json!({"input":"three","previous_response_id":"two"});
        history.prepare(None, &mut last).unwrap();
        assert_eq!(last["input"].as_array().unwrap().len(), 3);
        assert_eq!(last["input"][0]["content"], "one");
    }
}
