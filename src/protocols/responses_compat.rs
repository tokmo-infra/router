use serde_json::{map::Entry, Value};

type FieldError = (String, &'static str);

/// Accept Chat-style function definitions and named choices on the Responses endpoint.
fn unwrap_function(tool: &mut Value) -> Result<(), FieldError> {
    let Some(fields) = tool.as_object_mut() else {
        return Ok(());
    };
    if fields.get("type").and_then(Value::as_str) != Some("function") {
        return Ok(());
    }
    let Some(function) = fields.remove("function") else {
        return Ok(());
    };
    let Value::Object(nested) = function else {
        return Err(("function".into(), "Expected a function object."));
    };
    for (key, value) in nested {
        if key == "function" {
            return Err((key, "Unexpected second function wrapper."));
        }
        match fields.entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(value);
            }
            Entry::Occupied(entry) if entry.get() != &value => {
                return Err((
                    entry.key().clone(),
                    "Conflicting nested and top-level function fields.",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

pub(crate) fn normalize(body: &mut Value) -> Result<(), FieldError> {
    if let Some(tools) = body.get_mut("tools").and_then(Value::as_array_mut) {
        for (i, tool) in tools.iter_mut().enumerate() {
            unwrap_function(tool).map_err(|(key, msg)| (format!("tools[{i}].{key}"), msg))?;
        }
    }
    if let Some(choice) = body.get_mut("tool_choice") {
        unwrap_function(choice).map_err(|(key, msg)| (format!("tool_choice.{key}"), msg))?;
    }
    if let Some(input) = body.get_mut("input").and_then(Value::as_array_mut) {
        for item in input {
            // Some clients replay the assistant's content without the output message's
            // discriminator. Preserve every part, including annotations and refusals.
            if item.get("type").is_none()
                && item.get("role").and_then(Value::as_str) == Some("assistant")
                && item
                    .get("content")
                    .and_then(Value::as_array)
                    .is_some_and(|parts| {
                        !parts.is_empty()
                            && parts.iter().all(|p| {
                                matches!(
                                    p.get("type").and_then(Value::as_str),
                                    Some("output_text" | "refusal")
                                )
                            })
                    })
            {
                item["type"] = Value::from("message");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn nested_tools_and_named_choice_normalize_once() {
        let schema = json!({"type":"object","properties":{"function":{"type":"string"}}});
        let plain = json!({"type":"function","name":"flat","parameters":{}});
        let other = json!({"type":"custom","name":"raw","format":{"type":"text"}});
        let mut request = json!({
            "tools":[plain, {"type":"function","function":{"name":"call","description":"x","strict":false,"parameters":schema}}, other],
            "tool_choice":{"type":"function","function":{"name":"call"}}
        });
        let expected = json!({
            "tools":[plain, {"type":"function","name":"call","description":"x","strict":false,"parameters":schema}, other],
            "tool_choice":{"type":"function","name":"call"}
        });
        normalize(&mut request).unwrap();
        normalize(&mut request).unwrap();
        assert_eq!(request, expected);
    }

    #[test]
    fn conflicting_and_malformed_wrappers_identify_the_field() {
        let mut both = json!({"tools":[{"type":"function","name":"a","function":{"name":"a"}}]});
        normalize(&mut both).unwrap();
        assert_eq!(both["tools"][0], json!({"type":"function","name":"a"}));
        for (mut request, param) in [
            (
                json!({"tools":[{"type":"function","name":"a","function":{"name":"b"}}]}),
                "tools[0].name",
            ),
            (
                json!({"tool_choice":{"type":"function","function":null}}),
                "tool_choice.function",
            ),
        ] {
            assert_eq!(normalize(&mut request).unwrap_err().0, param);
        }
    }

    #[test]
    fn assistant_replay_preserves_content_and_unrelated_items() {
        let parts = json!([{"type":"output_text","text":"a","annotations":[{"type":"url_citation"}]},
                           {"type":"refusal","refusal":"b"}]);
        let mut request = json!({"input":[
            {"role":"assistant","content":parts},
            {"type":"message","role":"assistant","id":"msg_1","status":"completed","content":parts},
            {"role":"user","content":parts}, {"role":"assistant","content":"text"},
            {"role":"assistant","content":[{"type":"input_text","text":"in"}]},
            {"type":"function_call_output","call_id":"call_1","output":"result"}
        ]});
        let mut expected = request.clone();
        expected["input"][0]["type"] = json!("message");
        normalize(&mut request).unwrap();
        normalize(&mut request).unwrap();
        assert_eq!(request, expected);
    }
}
