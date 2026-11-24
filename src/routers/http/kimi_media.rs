//! Bounded CPU workers for model-specific inline media processing before PD dispatch.
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{io, process::Stdio, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Semaphore,
    time::timeout,
};

const KIMI: &str = include_str!("../../../tokmo/kimi31-inline-media/normalize.py");
const DEEPSEEK: &str = include_str!("../../../tokmo/ds41-inline-image/normalize.py");
const MAX_FRAME: usize = 4 * (256 * 1024 * 1024usize).div_ceil(3) + 1024 * 1024;

#[derive(Debug)]
pub(super) struct MediaPool {
    deepseek: bool,
    default_ds_model: bool,
    idle: Mutex<Vec<Worker>>,
    active: Semaphore,
    admitted: Semaphore,
}

#[derive(Debug)]
struct Worker {
    child: Child,
    input: ChildStdin,
    output: ChildStdout,
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Include the worker's FFmpeg child on cancellation or watchdog expiry.
        #[cfg(unix)]
        if let Some(pid) = self.child.id() {
            // SAFETY: start() gives this child its own process group.
            unsafe { libc::kill(-(pid as i32), libc::SIGKILL) };
        }
        let _ = self.child.start_kill();
    }
}

impl Worker {
    async fn start(deepseek: bool) -> io::Result<Self> {
        let mut cmd = Command::new("python3");
        cmd.args(["-u", "-c", if deepseek { DEEPSEEK } else { KIMI }])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);
        let mut child = cmd.spawn()?;
        let mut worker = Self {
            input: child.stdin.take().unwrap(),
            output: child.stdout.take().unwrap(),
            child,
        };
        let mut ready = [0; 4];
        timeout(
            Duration::from_secs(15),
            worker.output.read_exact(&mut ready),
        )
        .await??;
        if &ready != if deepseek { b"DS4\n" } else { b"K31\n" } {
            return Err(io::Error::other("unexpected media worker greeting"));
        }
        Ok(worker)
    }

    async fn exchange(&mut self, data: Vec<u8>) -> io::Result<Vec<u8>> {
        self.input.write_u32(data.len() as u32).await?;
        self.input.write_all(&data).await?;
        drop(data);
        self.input.flush().await?;
        let size = self.output.read_u32().await? as usize;
        if size > MAX_FRAME {
            return Err(io::Error::other(
                "media worker response exceeded byte limit",
            ));
        }
        let mut result = vec![0; size];
        self.output.read_exact(&mut result).await?;
        Ok(result)
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Reply {
    Error {
        error: String,
    },
    Ok {
        replacements: Vec<Option<Value>>,
        visual_tokens: u64,
        rewritten: u64,
    },
}

fn error(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(json!({"error": {"message": message, "code": status.as_u16()}})),
    )
        .into_response()
}

fn worker_error(deepseek: bool, e: impl std::fmt::Display) -> Response {
    tracing::error!(error=%e, "inline media worker failure");
    error(
        StatusCode::SERVICE_UNAVAILABLE,
        if deepseek {
            "DS image processing unavailable"
        } else {
            "K31 media processing unavailable"
        },
    )
}

fn media_positions(body: &Value) -> Vec<(usize, usize)> {
    let mut positions = Vec::new();
    if let Some(messages) = body["messages"].as_array() {
        for (m, msg) in messages.iter().enumerate() {
            if let Some(parts) = msg["content"].as_array() {
                for (p, part) in parts.iter().enumerate() {
                    if matches!(part["type"].as_str(), Some("image_url" | "video_url")) {
                        positions.push((m, p));
                    }
                }
            }
        }
    }
    positions
}

// Traverse API content blocks only. Tool schemas and arbitrary user JSON are opaque.
fn ds_parts(block: &Value, path: &str, out: &mut Vec<String>) -> Result<(), Response> {
    if let Some(parts) = block.as_array() {
        for (i, part) in parts.iter().enumerate() {
            ds_parts(part, &format!("{path}/{i}"), out)?;
        }
        return Ok(());
    }
    let kind = block["type"].as_str().unwrap_or("");
    match kind {
        "video" | "input_video" | "video_url" => {
            return Err(error(
                StatusCode::BAD_REQUEST,
                "DeepSeek-V4.1 accepts images only",
            ));
        }
        "image_url" | "input_image" => {
            let field = &block["image_url"];
            if field
                .as_str()
                .or_else(|| field["url"].as_str())
                .is_some_and(|s| s.starts_with("data:"))
            {
                out.push(path.to_string());
            }
        }
        "image" if block["source"]["type"] == "base64" => out.push(path.to_string()),
        "tool_result" | "message" | "" if block.get("content").is_some() => {
            ds_parts(&block["content"], &format!("{path}/content"), out)?;
        }
        "function_call_output" if block["output"].is_array() => {
            ds_parts(&block["output"], &format!("{path}/output"), out)?;
        }
        _ => {}
    }
    Ok(())
}

fn worker_count(name: &str) -> Result<usize, String> {
    let count: usize = match std::env::var(name) {
        Ok(s) => s.parse().map_err(|_| format!("{name} must be 0..=16"))?,
        Err(std::env::VarError::NotPresent) => 0,
        Err(e) => return Err(e.to_string()),
    };
    if count > 16 {
        return Err(format!("{name} must be 0..=16"));
    }
    Ok(count)
}

impl MediaPool {
    // A shared Router may see another model name. Only opt the deployed DS checkpoint in.
    pub(super) fn ds_request(&self, body: &Value) -> bool {
        self.deepseek
            && match body["model"].as_str() {
                Some(name) => crate::protocols::spec::is_ds41_flash(name),
                None => self.default_ds_model && body["model"].is_null(),
            }
    }

    pub(super) fn chat_request(&self, body: &Value) -> bool {
        !self.deepseek || self.ds_request(body)
    }

    pub(super) fn add_headers(&self, response: &mut Response, (tokens, rewritten): (u64, u64)) {
        let (t, r) = if self.deepseek {
            ("x-ds-visual-tokens", "x-ds-rewritten-items")
        } else {
            ("x-k31-visual-tokens", "x-k31-rewritten-items")
        };
        response.headers_mut().insert(t, tokens.into());
        response.headers_mut().insert(r, rewritten.into());
    }

    pub(super) async fn from_env() -> Result<Option<Self>, String> {
        let kimi = worker_count("VLLM_ROUTER_K31_MEDIA_WORKERS")?;
        let ds = worker_count("VLLM_ROUTER_DS41_IMAGE_WORKERS")?;
        if kimi > 0 && ds > 0 {
            return Err("Choose one media profile for this model Router".into());
        }
        let count = kimi.max(ds);
        if count == 0 {
            return Ok(None);
        }
        let deepseek = ds > 0;
        let mut idle = Vec::with_capacity(count);
        for _ in 0..count {
            idle.push(Worker::start(deepseek).await.map_err(|e| {
                format!(
                    "{} media worker startup: {e}",
                    if deepseek { "DS" } else { "K31" }
                )
            })?);
        }
        Ok(Some(Self {
            deepseek,
            default_ds_model: std::env::var("VLLM_ROUTER_DS41_IMAGE_DEFAULT_MODEL").as_deref()
                == Ok("deepseek-v4.1-flash-1017"),
            idle: Mutex::new(idle),
            active: Semaphore::new(count),
            admitted: Semaphore::new(count * 3),
        }))
    }

    pub(super) async fn prepare(&self, body: &mut Value) -> Result<(u64, u64), Response> {
        let positions = if self.deepseek {
            let mut paths = Vec::new();
            for name in ["messages", "input", "previous_input_messages"] {
                if body[name].is_array() {
                    ds_parts(&body[name], &format!("/{name}"), &mut paths)?;
                }
            }
            paths
        } else {
            media_positions(body)
                .into_iter()
                .map(|(m, p)| format!("/messages/{m}/content/{p}"))
                .collect()
        };
        if positions.is_empty() {
            return Ok((0, 0));
        }
        let _admitted = self.admitted.try_acquire().map_err(|_| {
            error(
                StatusCode::TOO_MANY_REQUESTS,
                if self.deepseek {
                    "DS image queue full"
                } else {
                    "K31 media queue full"
                },
            )
        })?;
        let _active = self
            .active
            .acquire()
            .await
            .map_err(|e| worker_error(self.deepseek, e))?;
        let items: Vec<_> = positions
            .iter()
            .map(|path| body.pointer_mut(path).unwrap().take())
            .collect();
        let (items, data) = tokio::task::spawn_blocking(move || {
            serde_json::to_vec(&items).map(|data| (items, data))
        })
        .await
        .map_err(|e| worker_error(self.deepseek, e))?
        .map_err(|e| worker_error(self.deepseek, e))?;
        if data.len() > MAX_FRAME {
            return Err(error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "Inline media exceed the request byte limit",
            ));
        }
        let idle = self.idle.lock().pop();
        let mut worker = match idle {
            Some(worker) => worker,
            None => Worker::start(self.deepseek)
                .await
                .map_err(|e| worker_error(self.deepseek, e))?,
        };
        let result = timeout(Duration::from_secs(210), worker.exchange(data))
            .await
            .map_err(|_| {
                error(
                    StatusCode::GATEWAY_TIMEOUT,
                    if self.deepseek {
                        "DS image processing timed out"
                    } else {
                        "K31 media processing timed out"
                    },
                )
            })?
            .map_err(|e| worker_error(self.deepseek, e))?;
        let reply: Reply = tokio::task::spawn_blocking(move || serde_json::from_slice(&result))
            .await
            .map_err(|e| worker_error(self.deepseek, e))?
            .map_err(|e| worker_error(self.deepseek, e))?;
        self.idle.lock().push(worker);
        match reply {
            Reply::Error { error: message } => Err(error(StatusCode::BAD_REQUEST, &message)),
            Reply::Ok {
                replacements,
                visual_tokens,
                rewritten,
            } => {
                if replacements.len() != positions.len() {
                    return Err(worker_error(self.deepseek, "media part count mismatch"));
                }
                for ((path, original), replacement) in
                    positions.into_iter().zip(items).zip(replacements)
                {
                    *body.pointer_mut(&path).unwrap() = replacement.unwrap_or(original);
                }
                Ok((visual_tokens, rewritten))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identifies_only_media_in_message_order() {
        let body = json!({"messages":[{"content":"video_url"},{"content":[{"type":"text"},{"type":"image_url"}]},
            {"content":[{"type":"video_url"},{"type":"text"},{"type":"image_url"}]}]});
        assert_eq!(media_positions(&body), vec![(1, 1), (2, 0), (2, 2)]);
    }
    #[test]
    fn ds_image_paths_preserve_protocol_and_order() {
        let body = json!([
            {"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,A"}}]},
            {"type":"message","content":[{"type":"input_image","image_url":"data:image/png;base64,B"}]},
            {"content":[{"type":"tool_result","content":[{"type":"image","source":{"type":"base64"}}]}]},
            {"type":"function_call","arguments":{"type":"video"}}
        ]);
        let mut paths = Vec::new();
        ds_parts(&body, "", &mut paths).unwrap();
        assert_eq!(
            paths,
            vec!["/0/content/0", "/1/content/0", "/2/content/0/content/0"]
        );
        assert!(ds_parts(&json!([{"type":"input_video"}]), "", &mut paths).is_err());
        paths.clear();
        ds_parts(
            &json!([
                {"type":"input_image","image_url":"https://images.example/test.png"},
                {"type":"image","source":{"type":"url","url":"https://images.example/test.png"}},
                {"role":"user","content":"text"}
            ]),
            "",
            &mut paths,
        )
        .unwrap();
        assert!(paths.is_empty());
    }
    #[test]
    fn ds_model_gate_preserves_default_and_foreign_models() {
        let mut pool = MediaPool {
            deepseek: true,
            default_ds_model: false,
            idle: Mutex::new(Vec::new()),
            active: Semaphore::new(0),
            admitted: Semaphore::new(0),
        };
        let ds = json!({"model":"deepseek-v4.1-flash-1017"});
        let kimi = json!({"model":"kimi-k3.1-pro"});
        assert!(pool.ds_request(&ds));
        assert!(!pool.chat_request(&kimi));
        assert!(!pool.ds_request(&json!({})));
        pool.default_ds_model = true;
        assert!(pool.ds_request(&json!({})));
        assert!(!pool.ds_request(&kimi));
        pool.deepseek = false;
        assert!(pool.chat_request(&kimi));
        assert!(!pool.ds_request(&ds));
    }

}
