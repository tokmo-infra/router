//! DeepSeek V4.1 Flash sampling policy, applied before P/D dispatch.
use serde_json::{json, Value};

use crate::protocols::spec::is_ds41_flash;

// Responses and Messages use the generic vLLM template. Its two thinking
// flags are ORed, and the "none" effort overrides both flags.
fn template_thinking(kwargs: &Value, effort: Option<&str>) -> bool {
    let effort = effort.or_else(|| kwargs["reasoning_effort"].as_str());
    if effort == Some("none") {
        return false;
    }
    let thinking = kwargs["thinking"].as_bool();
    let enabled = kwargs["enable_thinking"].as_bool();
    thinking.unwrap_or(false) || enabled.unwrap_or(effort.is_some() || thinking.is_none())
}

fn chat_thinking(body: &Value) -> bool {
    // The native Chat frontend resolves explicit thinking before effort,
    // then falls back to template flags and finally to enabled.
    if let Some(mode) = body["thinking"]["type"].as_str() {
        return mode == "enabled";
    }
    if let Some(effort) = body["reasoning_effort"].as_str() {
        return effort != "none";
    }
    let kwargs = &body["chat_template_kwargs"];
    kwargs["thinking"]
        .as_bool()
        .or_else(|| kwargs["enable_thinking"].as_bool())
        .unwrap_or(true)
        && kwargs["reasoning_effort"].as_str() != Some("none")
}

fn messages_thinking(body: &mut Value) -> bool {
    // The backend accepts thinking.disabled but its output_config schema
    // lacks the equivalent official "none" spelling. Keep all other fields.
    if body["output_config"]["effort"] == "none" {
        body["output_config"]
            .as_object_mut()
            .unwrap()
            .remove("effort");
        if body["thinking"].is_null() {
            body["thinking"] = json!({"type":"disabled"});
        }
    }
    body["thinking"]["type"] != "disabled"
        && template_thinking(
            &body["chat_template_kwargs"],
            body["output_config"]["effort"].as_str(),
        )
}

pub(super) fn normalize(body: &mut Value, path: &str) {
    if !body.is_object()
        || !matches!(
            path,
            "/v1/chat/completions" | "/v1/responses" | "/v1/messages"
        )
    {
        return;
    }
    let is_flash = match body.get("model") {
        None | Some(Value::Null) => std::env::var("VLLM_ROUTER_DS41_IMAGE_DEFAULT_MODEL")
            .is_ok_and(|name| is_ds41_flash(&name)),
        Some(model) => model.as_str().is_some_and(is_ds41_flash),
    };
    if !is_flash {
        return;
    }
    let thinking = match path {
        "/v1/chat/completions" => chat_thinking(body),
        "/v1/messages" => messages_thinking(body),
        _ => template_thinking(
            &body["chat_template_kwargs"],
            body["reasoning"]["effort"].as_str(),
        ),
    };
    // Leave invalid input intact for each protocol's normal 4xx validation.
    let temperature = &body["temperature"];
    if thinking
        && (temperature.is_null()
            || temperature
                .as_f64()
                .is_some_and(|n| (0.0..=2.0).contains(&n)))
    {
        body["temperature"] = json!(1.0);
    }
    tracing::debug!(target: "ds41_sampling", path, thinking, temperature = %body["temperature"], "sampling before P/D dispatch");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn applied(path: &str, fields: Value) -> Value {
        let mut body = fields;
        body["model"] = json!("deepseek-v4.1-flash-1017");
        normalize(&mut body, path);
        body
    }

    #[test]
    fn valid_temperature_and_default_mode() {
        for path in ["/v1/chat/completions", "/v1/responses", "/v1/messages"] {
            for t in [0.0, 0.1, 1.0, 2.0] {
                assert_eq!(applied(path, json!({"temperature":t}))["temperature"], 1.0);
            }
            assert_eq!(applied(path, json!({}))["temperature"], 1.0);
            for t in [json!(-0.1), json!(2.1), json!("invalid")] {
                assert_eq!(applied(path, json!({"temperature":t}))["temperature"], t);
            }
        }
    }

    #[test]
    fn off_modes_preserve_temperature_and_explicit_chat_switch_wins() {
        for (path, fields) in [
            ("/v1/chat/completions", json!({"reasoning_effort":"none"})),
            (
                "/v1/chat/completions",
                json!({"thinking":{"type":"disabled"},"reasoning_effort":"high"}),
            ),
            (
                "/v1/chat/completions",
                json!({"chat_template_kwargs":{"enable_thinking":false}}),
            ),
            ("/v1/responses", json!({"reasoning":{"effort":"none"}})),
            (
                "/v1/responses",
                json!({"chat_template_kwargs":{"enable_thinking":false},"reasoning":{"effort":"high"}}),
            ),
            ("/v1/messages", json!({"thinking":{"type":"disabled"}})),
            ("/v1/messages", json!({"output_config":{"effort":"none"}})),
        ] {
            for t in [0.0, 0.1, 2.0] {
                let mut body = fields.clone();
                body["temperature"] = json!(t);
                assert_eq!(applied(path, body)["temperature"], t);
            }
        }
        let body = applied(
            "/v1/chat/completions",
            json!({"temperature":0,"reasoning_effort":"none","thinking":{"type":"enabled"}}),
        );
        assert_eq!(body["temperature"], 1.0);
        let body = applied(
            "/v1/messages",
            json!({"temperature":0,"output_config":{"effort":"none","extra":"keep"},"thinking":{"type":"adaptive"}}),
        );
        assert_eq!(body["output_config"], json!({"extra":"keep"}));
        assert_eq!(body["temperature"], 1.0);
    }

    #[test]
    fn other_models_and_routes_are_untouched() {
        for (model, path) in [
            ("kimi-k3.1-pro", "/v1/messages"),
            ("deepseek-v4.1-flash", "/v1/completions"),
        ] {
            let original =
                json!({"model":model, "temperature":0.1, "output_config":{"effort":"none"}});
            let mut actual = original.clone();
            normalize(&mut actual, path);
            assert_eq!(actual, original);
        }
    }
}
