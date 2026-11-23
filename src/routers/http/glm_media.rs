//! GLM Flash media handling before worker selection and P/D request splitting.
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use image::{codecs::jpeg::JpegEncoder, ImageDecoder, ImageFormat, ImageReader, Limits};
use serde_json::{json, Value};
use std::{io::Cursor, sync::Arc};
use tokio::sync::Semaphore;

#[path = "glm_video.rs"]
mod video;

const MAX_SOURCE_BYTES: usize = 32 * 1024 * 1024;
const MAX_PIXELS: u64 = 16 * 1024 * 1024;
const DEFAULT_IMAGE_BUDGET: usize = 4 * 1024 * 1024;
const IMAGE_BUDGET_ENV: &str = "VLLM_ROUTER_GLM_INLINE_IMAGE_MAX_BYTES";
const CHAT: &str = "/v1/chat/completions";
const RESPONSES: &str = "/v1/responses";
const MESSAGES: &str = "/v1/messages";
const COUNT_TOKENS: &str = "/v1/messages/count_tokens";

#[derive(Debug)]
pub(super) struct GlmMedia {
    budget: usize,
    cpu_slots: Arc<Semaphore>,
}

impl GlmMedia {
    pub(super) fn from_env() -> Result<Self, String> {
        let budget = match std::env::var(IMAGE_BUDGET_ENV) {
            Ok(value) => value
                .parse::<usize>()
                .ok()
                .filter(|value| (64 * 1024..=MAX_SOURCE_BYTES).contains(value))
                .ok_or_else(|| {
                    format!("{IMAGE_BUDGET_ENV} must be between 65536 and {MAX_SOURCE_BYTES}")
                })?,
            Err(std::env::VarError::NotPresent) => DEFAULT_IMAGE_BUDGET,
            Err(_) => return Err(format!("Invalid {IMAGE_BUDGET_ENV}")),
        };
        Ok(Self {
            budget,
            cpu_slots: Arc::new(Semaphore::new(2)),
        })
    }

    pub(super) async fn prepare(&self, path: &str, body: Value) -> Result<Value, Response> {
        let Some(root) = media_root(path) else {
            return Ok(body);
        };
        let clip = video::locate(path, &body[root]).map_err(|message| {
            media_error(
                path,
                StatusCode::BAD_REQUEST,
                "invalid_inline_video",
                message,
            )
        })?;
        let has_clip = clip.is_some();
        let budget = self.budget;
        let resize_images = visit(&body[root], &mut |block| oversized(block, budget));
        if !has_clip && !resize_images {
            return Ok(body);
        }
        let permit = self.cpu_slots.clone().try_acquire_owned().map_err(|_| {
            media_error(
                path,
                StatusCode::SERVICE_UNAVAILABLE,
                if has_clip {
                    "video_processing_busy"
                } else {
                    "image_processing_busy"
                },
                "Inline media processing is busy; retry shortly.",
            )
        })?;
        // Move a single body into the CPU job. Keep the slot until that job exits,
        // including when the client drops the request.
        match tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let mut body = body;
            if let Some(clip) = clip {
                clip.prepare(&mut body[root])?;
            }
            if resize_images {
                normalize_body(body, root, budget)
                    .map_err(|message| (StatusCode::BAD_REQUEST, "invalid_inline_image", message))
            } else {
                Ok(body)
            }
        })
        .await
        {
            Ok(Ok(body)) => Ok(body),
            Ok(Err((status, code, message))) => Err(media_error(path, status, code, message)),
            Err(_) => Err(media_error(
                path,
                StatusCode::INTERNAL_SERVER_ERROR,
                "media_processing_failed",
                "Inline media processing failed.",
            )),
        }
    }
}

fn media_root(path: &str) -> Option<&'static str> {
    match path {
        CHAT | MESSAGES | COUNT_TOKENS => Some("messages"),
        RESPONSES => Some("input"),
        _ => None,
    }
}

fn media_error(path: &str, status: StatusCode, code: &str, message: &str) -> Response {
    let kind = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "api_error"
    };
    let mut body = json!({"error": {"type": kind, "message": message}});
    if matches!(path, MESSAGES | COUNT_TOKENS) {
        body["type"] = json!("error");
    } else {
        body["error"]["code"] = json!(code);
    }
    (status, Json(body)).into_response()
}

// Inspect only protocol content, never tool arguments, tool schemas or plain text.
fn is_container(node: &Value) -> bool {
    matches!(
        node["type"].as_str(),
        None | Some("message" | "tool_result")
    )
}

fn visit(node: &Value, predicate: &mut impl FnMut(&Value) -> bool) -> bool {
    if let Some(parts) = node.as_array() {
        return parts.iter().any(|part| visit(part, predicate));
    }
    node.is_object()
        && (predicate(node) || (is_container(node) && visit(&node["content"], predicate)))
}

fn image_url(block: &Value) -> Option<&str> {
    let field = &block["image_url"];
    field.as_str().or_else(|| field["url"].as_str())
}

fn inline(block: &Value) -> Option<(&str, &str)> {
    match block["type"].as_str()? {
        "image_url" | "input_image" => {
            let (header, data) = image_url(block)?.strip_prefix("data:")?.split_once(',')?;
            let mime = header.strip_suffix(";base64")?;
            Some((mime, data))
        }
        "image" if block["source"]["type"] == "base64" => Some((
            block["source"]["media_type"].as_str()?,
            block["source"]["data"].as_str()?,
        )),
        _ => None,
    }
}

fn oversized(block: &Value, budget: usize) -> bool {
    inline(block).is_some_and(|(_, data)| {
        let padding = if data.ends_with("==") {
            2
        } else {
            usize::from(data.ends_with('='))
        };
        (data.len() / 4 * 3).saturating_sub(padding) > budget
    })
}

fn normalize_body(mut body: Value, root: &str, budget: usize) -> Result<Value, &'static str> {
    fn rewrite(node: &mut Value, budget: usize) -> Result<(), &'static str> {
        if let Some(parts) = node.as_array_mut() {
            for part in parts {
                rewrite(part, budget)?;
            }
        } else if node.is_object() && oversized(node, budget) {
            let (mime, data) = inline(node).expect("oversized image has inline data");
            let (mime, bytes) = compress(mime, data, budget)?;
            let data = STANDARD.encode(bytes);
            if node["type"] == "image" {
                node["source"]["data"] = json!(data);
                node["source"]["media_type"] = json!(mime);
            } else {
                let url = json!(format!("data:{mime};base64,{data}"));
                if node["image_url"].is_object() {
                    node["image_url"]["url"] = url;
                } else {
                    node["image_url"] = url;
                }
            }
            // The client-supplied media identity only described the original bytes.
            if let Some(block) = node.as_object_mut() {
                block.remove("uuid");
            }
        } else if is_container(node) {
            if let Some(content) = node.get_mut("content") {
                rewrite(content, budget)?;
            }
        }
        Ok(())
    }

    if let Some(content) = body.get_mut(root) {
        rewrite(content, budget)?;
    }
    Ok(body)
}

fn compress(
    mime: &str,
    data: &str,
    budget: usize,
) -> Result<(&'static str, Vec<u8>), &'static str> {
    let format =
        match mime {
            "image/jpeg" => ImageFormat::Jpeg,
            "image/png" => ImageFormat::Png,
            _ => return Err(
                "Only over-budget inline JPEG and PNG images can be resized; send a smaller image.",
            ),
        };
    if data.len() > MAX_SOURCE_BYTES.div_ceil(3) * 4 {
        return Err("Inline image exceeds the source byte limit.");
    }
    let bytes = STANDARD
        .decode(data)
        .map_err(|_| "Invalid inline base64 image.")?;
    if bytes.len() > MAX_SOURCE_BYTES {
        return Err("Inline image exceeds the source byte limit.");
    }
    if image::guess_format(&bytes).ok() != Some(format) {
        return Err("Inline image does not match its declared format.");
    }
    let mut reader = ImageReader::with_format(Cursor::new(bytes), format);
    let mut limits = Limits::default();
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    limits.max_alloc = Some(128 * 1024 * 1024);
    reader.limits(limits);
    let mut decoder = reader
        .into_decoder()
        .map_err(|_| "Invalid inline image or image dimensions.")?;
    let (width, height) = decoder.dimensions();
    if u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err("Inline image exceeds the pixel limit.");
    }
    let orientation = decoder
        .orientation()
        .map_err(|_| "Invalid inline image orientation.")?;
    let mut image =
        image::DynamicImage::from_decoder(decoder).map_err(|_| "Invalid inline image pixels.")?;
    image.apply_orientation(orientation);
    let output_mime = if format == ImageFormat::Png {
        "image/png"
    } else {
        "image/jpeg"
    };
    for _ in 0..12 {
        let mut output = Vec::new();
        let result = if format == ImageFormat::Jpeg {
            JpegEncoder::new_with_quality(&mut output, 88).encode_image(&image)
        } else {
            image.write_to(&mut Cursor::new(&mut output), format)
        };
        result.map_err(|_| "Could not encode inline image.")?;
        if output.len() <= budget {
            return Ok((output_mime, output));
        }
        image = image.resize(
            (image.width() * 3 / 4).max(1),
            (image.height() * 3 / 4).max(1),
            image::imageops::FilterType::Triangle,
        );
    }
    Err("Inline image could not be resized to the configured byte budget.")
}
