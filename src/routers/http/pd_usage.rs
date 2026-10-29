//! Forward measured P-side historical-cache usage through vLLM's existing
//! remote_prefill_cached_tokens protocol. This changes no KV data or routing.
use serde_json::{json, Value};

pub fn attach_mooncake_prefill_usage(
    prefill: &Value,
    params: &mut Value,
) -> Result<(), &'static str> {
    let prompt = prefill.pointer("/usage/prompt_tokens").and_then(Value::as_u64)
        .ok_or("missing or invalid measured P prompt_tokens")?;
    let cached = prefill.pointer("/usage/prompt_tokens_details/cached_tokens")
        .and_then(Value::as_u64).ok_or("missing or invalid measured P cached_tokens")?;
    if cached > prompt { return Err("P cached_tokens exceeds prompt_tokens"); }
    if let Some(remote) = prefill.pointer("/kv_transfer_params/remote_prefill_cached_tokens") {
        if remote.as_u64() != Some(cached) {
            return Err("P usage and transfer metadata disagree on cached_tokens");
        }
    }
    if !params.is_object() { return Err("Mooncake transfer params must be an object"); }
    params["remote_prefill_cached_tokens"] = json!(cached);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn response(cached: Value) -> Value {
        json!({"usage":{"prompt_tokens":80000,"prompt_tokens_details":{"cached_tokens":cached}}})
    }
    #[test] fn cold_zero_is_observed_not_fabricated() {
        let mut p=json!({"transfer_id":"abc","remote_engine_id":"P0"});
        attach_mooncake_prefill_usage(&response(json!(0)),&mut p).unwrap();
        assert_eq!(p,json!({"transfer_id":"abc","remote_engine_id":"P0","remote_prefill_cached_tokens":0}));
    }
    #[test] fn warm_uses_exact_measured_count() {
        let mut p=json!({});attach_mooncake_prefill_usage(&response(json!(78400)),&mut p).unwrap();
        assert_eq!(p["remote_prefill_cached_tokens"],78400);
    }
    #[test] fn missing_usage_fails_closed() {
        assert!(attach_mooncake_prefill_usage(&json!({}),&mut json!({})).is_err());
    }
    #[test] fn missing_cache_details_fails_closed() {
        assert!(attach_mooncake_prefill_usage(&json!({"usage":{"prompt_tokens":80000}}),&mut json!({})).is_err());
    }
    #[test] fn negative_rejected() {
        assert!(attach_mooncake_prefill_usage(&response(json!(-1)),&mut json!({})).is_err());
    }
    #[test] fn string_rejected() {
        assert!(attach_mooncake_prefill_usage(&response(json!("0")),&mut json!({})).is_err());
    }
    #[test] fn float_rejected() {
        assert!(attach_mooncake_prefill_usage(&response(json!(0.0)),&mut json!({})).is_err());
    }
    #[test] fn boolean_rejected() {
        assert!(attach_mooncake_prefill_usage(&response(json!(false)),&mut json!({})).is_err());
    }
    #[test] fn greater_than_input_rejected() {
        assert!(attach_mooncake_prefill_usage(&response(json!(80001)),&mut json!({})).is_err());
    }
    #[test] fn metadata_disagreement_rejected() {
        let mut r=response(json!(0));r["kv_transfer_params"]=json!({"remote_prefill_cached_tokens":79999});
        assert!(attach_mooncake_prefill_usage(&r,&mut json!({})).is_err());
    }
    #[test] fn matching_metadata_accepted() {
        let mut r=response(json!(78400));r["kv_transfer_params"]=json!({"remote_prefill_cached_tokens":78400});
        let mut p=json!({});attach_mooncake_prefill_usage(&r,&mut p).unwrap();assert_eq!(p["remote_prefill_cached_tokens"],78400);
    }
    #[test] fn invalid_params_rejected() {
        assert!(attach_mooncake_prefill_usage(&response(json!(0)),&mut Value::Null).is_err());
    }
}
