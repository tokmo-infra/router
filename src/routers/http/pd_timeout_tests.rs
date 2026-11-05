//! Loopback-only CPU regressions; durations scale the old 600s failure to milliseconds.
use super::*;
use axum::{
    routing::{get, post},
    Json, Router,
};
use futures_util::{stream, StreamExt};
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};
use tokio::{net::TcpListener, task::JoinHandle};

struct Mock {
    url: String,
    task: JoinHandle<()>,
}
impl Drop for Mock {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn serve(app: Router) -> Mock {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Mock { url, task }
}
fn limits(total: u64, first: u64, idle: u64) -> PdTimeouts {
    PdTimeouts {
        connect: Duration::from_millis(100),
        total: Duration::from_millis(total),
        first_response: Duration::from_millis(first),
        idle: Duration::from_millis(idle),
    }
}
async fn send(
    budget: PdBudget,
    client: &reqwest::Client,
    url: &str,
) -> Result<super::super::pd_timeout::PdResponse, super::super::pd_timeout::PdTimeoutError> {
    budget
        .send(
            client.post(url),
            None,
            ClientRequestOptions {
                method: "POST",
                url,
                route: Some("/mock"),
                request_phase: Some("decode"),
            },
        )
        .await
}
fn chunks(count: usize, delay_ms: u64) -> Body {
    Body::from_stream(stream::unfold(0, move |i| async move {
        if i == count {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        Some((
            Ok::<_, std::io::Error>(Bytes::from_static(b"data: token\n\n")),
            i + 1,
        ))
    }))
}

#[tokio::test]
async fn legacy_overall_cuts_active_stream_but_explicit_budget_survives() {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let mock = serve(Router::new().route(
        "/",
        post(move || {
            c.fetch_add(1, Ordering::SeqCst);
            async { Response::new(chunks(10, 25)) }
        }),
    ))
    .await;
    let legacy = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_millis(80))
        .build()
        .unwrap();
    let response = legacy.post(&mock.url).send().await.unwrap();
    assert!(
        response.bytes().await.unwrap_err().is_timeout(),
        "active chunks do not reset overall timeout"
    );
    let bytes = send(limits(1500, 600, 150).start(), &legacy, &mock.url)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(bytes.len(), 10 * b"data: token\n\n".len());
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "one POST per explicit request"
    );
}

#[tokio::test]
async fn slow_prefill_headers_are_not_body_idle_and_have_a_finite_first_response_limit() {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let mock = serve(Router::new().route(
        "/",
        post(move || {
            c.fetch_add(1, Ordering::SeqCst);
            async {
                tokio::time::sleep(Duration::from_millis(180)).await;
                "ready"
            }
        }),
    ))
    .await;
    let client = limits(1500, 600, 50).client().unwrap();
    assert_eq!(
        send(limits(1500, 600, 50).start(), &client, &mock.url)
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap(),
        "ready"
    );
    let error = send(limits(1500, 50, 400).start(), &client, &mock.url)
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("first_response"), "{error}");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn response_headers_do_not_bypass_first_body_deadline() {
    let mock =
        serve(Router::new().route("/", post(|| async { Response::new(chunks(1, 180)) }))).await;
    let config = limits(1500, 60, 600);
    let error = send(config.start(), &config.client().unwrap(), &mock.url)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap_err();
    assert!(error.to_string().contains("first_response"), "{error}");
}

#[tokio::test]
async fn idle_after_first_chunk_errors_without_replay() {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let mock = serve(Router::new().route(
        "/",
        post(move || {
            c.fetch_add(1, Ordering::SeqCst);
            async { Response::new(chunks(3, 150)) }
        }),
    ))
    .await;
    let config = limits(1500, 600, 50);
    let mut body = send(config.start(), &config.client().unwrap(), &mock.url)
        .await
        .unwrap()
        .bytes_stream();
    assert!(body.next().await.unwrap().is_ok());
    assert!(body
        .next()
        .await
        .unwrap()
        .unwrap_err()
        .to_string()
        .contains("body_idle"));
    assert!(body.next().await.is_none());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn total_budget_is_shared_across_prefill_and_decode() {
    let mock =
        serve(Router::new().route("/", post(|| async { Response::new(chunks(4, 60)) }))).await;
    let config = limits(380, 350, 200);
    let client = config.client().unwrap();
    let budget = config.start();
    send(budget, &client, &mock.url)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let result = send(budget, &client, &mock.url)
        .await
        .unwrap()
        .bytes()
        .await;
    assert!(result.unwrap_err().to_string().contains("total"));
}

#[tokio::test]
async fn downstream_pause_is_backpressure_not_upstream_idle() {
    let mock =
        serve(Router::new().route("/", post(|| async { Response::new(chunks(6, 5)) }))).await;
    let config = limits(1500, 600, 40);
    let mut body = send(config.start(), &config.client().unwrap(), &mock.url)
        .await
        .unwrap()
        .bytes_stream();
    assert!(body.next().await.unwrap().is_ok());
    tokio::time::sleep(Duration::from_millis(180)).await;
    while let Some(chunk) = body.next().await {
        chunk.unwrap();
    }
}

#[tokio::test]
async fn connect_timeout_is_distinct_from_first_response() {
    // Local TCP accepts but never completes TLS; no external host or inference.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("https://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (_socket, _) = listener.accept().await.unwrap();
        std::future::pending::<()>().await;
    });
    let config = limits(1500, 600, 600);
    let error = send(config.start(), &config.client().unwrap(), &url)
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("connect timeout"), "{error}");
    server.abort();
}

async fn mock_router(p: &Mock, d: &Mock, config: PdTimeouts) -> VllmPDRouter {
    use crate::config::{RouterConfig, RoutingMode};
    let port = p.url.rsplit(':').next().unwrap().parse().unwrap();
    let prefill_urls = vec![(p.url.clone(), Some(port))];
    let decode_urls = vec![d.url.clone()];
    let router_config = RouterConfig {
        kv_connector: KvConnector::Mooncake,
        mode: RoutingMode::VllmPrefillDecode {
            prefill_urls: prefill_urls.clone(),
            decode_urls: decode_urls.clone(),
            prefill_policy: None,
            decode_policy: None,
            discovery_address: None,
        },
        worker_startup_timeout_secs: 2,
        worker_startup_check_interval_secs: 1,
        ..RouterConfig::default()
    };
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_millis(80))
        .build()
        .unwrap();
    let ctx = Arc::new(
        crate::server::AppContext::new(router_config, client, 1024, None, vec![]).unwrap(),
    );
    let mut router = VllmPDRouter::new(prefill_urls, decode_urls, None, &ctx)
        .await
        .unwrap();
    router.pd_timeouts = config;
    router.pd_client = config.client().unwrap();
    let mut workers = router.pd_router.worker_registry.get_prefill_workers();
    workers.extend(router.pd_router.worker_registry.get_decode_workers());
    router.token_balancer = Some(TokenBalancer::for_test_workers(&workers));
    router
}
fn endpoints(app: Router) -> Router {
    app.route("/health", get(|| async { "OK" }))
        .route(
            "/metrics",
            get(|| async { "vllm:num_requests_running{engine=\"0\"} 0\n" }),
        )
        .route(
            "/query",
            get(|| async { Json(json!({"0":{"engine_id":"mock-P"}})) }),
        )
}
async fn route_response(router: &VllmPDRouter) -> Result<Response, PDRouterError> {
    let request = json!({"model":"mock","messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":10});
    route_request(router, request, "/v1/chat/completions").await
}
async fn route_request(router: &VllmPDRouter, request: Value, path: &str) -> Result<Response, PDRouterError> {
    let buffered = is_buffered_logprobs_response(&request);
    let ps = router.pd_router.worker_registry.get_prefill_workers();
    let ds = router.pd_router.worker_registry.get_decode_workers();
    let (pi, di, reservation, pr, budget) = router
        .select_direct_pair(&ps, &ds, &request, None, None)
        .await
        .unwrap();
    router
        .process_vllm_two_stage_request(
            request,
            ps[pi].clone(),
            ds[di].clone(),
            path,
            None,
            pr,
            budget,
            RequestTrace::new(None, false),
        )
        .await
        .map(|response| finish_reserved_response(response, reservation, buffered))
}
fn assert_load(router: &VllmPDRouter, p: usize, d: usize) {
    assert_eq!(
        router
            .pd_router
            .worker_registry
            .get_prefill_workers()
            .iter()
            .map(|w| w.load())
            .sum::<usize>(),
        p
    );
    assert_eq!(
        router
            .pd_router
            .worker_registry
            .get_decode_workers()
            .iter()
            .map(|w| w.load())
            .sum::<usize>(),
        d
    );
    let b = router.token_balancer.as_ref().unwrap();
    assert_eq!(
        router
            .pd_router
            .worker_registry
            .get_prefill_workers()
            .iter()
            .map(|w| b.assigned_for_test(w.url()))
            .sum::<usize>(),
        p
    );
    assert_eq!(
        router
            .pd_router
            .worker_registry
            .get_decode_workers()
            .iter()
            .map(|w| b.assigned_for_test(w.url()))
            .sum::<usize>(),
        d
    );
}

#[tokio::test]
async fn force_length_cap_scope_and_unsupported_beam_search() {
    let p = serve(endpoints(Router::new())).await;
    let d = serve(endpoints(Router::new())).await;
    let mut router = mock_router(&p, &d, limits(1800, 600, 150)).await;
    let request = json!({"max_tokens":64,"ignore_eos":false,"min_tokens":0,
        "stop":"STOP","stop_token_ids":[2],"thinking_token_budget":128});
    router.prefill_force_length_cap = true;
    for path in ["/v1/responses", "/inference/v1/generate", "/generate", "/v1/chat/completions/batch"] {
        assert_eq!(router.prepare_direct_prefill_request(request.clone(), path).unwrap(),
            VllmPDRouter::prepare_prefill_request(request.clone(), path), "unsupported API {path}");
    }
    for connector in [KvConnector::Nixl, KvConnector::MoriIO] {
        router.kv_connector = connector;
        assert_eq!(router.prepare_direct_prefill_request(request.clone(), "/v1/chat/completions").unwrap(),
            VllmPDRouter::prepare_prefill_request(request.clone(), "/v1/chat/completions"));
    }
    router.kv_connector = KvConnector::Mooncake;
    router.use_discovery = true;
    assert_eq!(router.prepare_direct_prefill_request(request.clone(), "/v1/chat/completions").unwrap(),
        VllmPDRouter::prepare_prefill_request(request.clone(), "/v1/chat/completions"));
    router.use_discovery = false;
    let mut beam = request.clone();
    beam["use_beam_search"] = json!(true);
    for path in ["/v1/chat/completions", "/v1/completions"] {
        assert!(router.prepare_direct_prefill_request(beam.clone(), path).is_err());
    }
    assert!(route_request(&router, beam.clone(), "/v1/chat/completions").await.is_err());
    assert_load(&router, 0, 0);
    router.prefill_force_length_cap = false;
    assert_eq!(router.prepare_direct_prefill_request(beam.clone(), "/v1/chat/completions").unwrap(),
        VllmPDRouter::prepare_prefill_request(beam, "/v1/chat/completions"));
}

// Isolated subprocesses exercise the ACTUAL constructor getenv path without
// mutating the environment of concurrently running Rust tests.
#[test]
fn force_length_cap_environment_exact_one_and_default_off() {
    for value in [None, Some("0"), Some("1"), Some("true"), Some("01"), Some(" 1")] {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", "routers::http::vllm_pd_router::timeout_regressions::force_length_cap_env_two_stage_child", "--nocapture"])
            .env("VLLM_ROUTER_TEST_FORCE_CAP_CHILD", "1")
            .env("NO_PROXY", "*").env("no_proxy", "*");
        for key in ["OPENAI_API_KEY", "API_KEY_VALIDATION_URLS", "VLLM_ROUTER_PD_REUSE_PROMPT_IDS",
            "VLLM_ROUTER_PD_TOKEN_BALANCE_CONFIG", "VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP",
            "VLLM_ROUTER_PD_CONNECT_TIMEOUT_SECS", "VLLM_ROUTER_PD_FIRST_RESPONSE_TIMEOUT_SECS",
            "VLLM_ROUTER_PD_IDLE_TIMEOUT_SECS", "VLLM_ROUTER_PD_TOTAL_TIMEOUT_SECS"] {
            cmd.env_remove(key);
        }
        if let Some(value) = value { cmd.env("VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP", value); }
        let output = cmd.output().unwrap();
        assert!(output.status.success(), "isolated mode {value:?} failed: {} {}",
            String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"), "child test was not selected");
    }
}

#[tokio::test]
async fn force_length_cap_env_two_stage_child() {
    if std::env::var("VLLM_ROUTER_TEST_FORCE_CAP_CHILD").ok().as_deref() != Some("1") { return; }
    let enabled = std::env::var("VLLM_ROUTER_PD_PREFILL_FORCE_LENGTH_CAP").ok().as_deref() == Some("1");
    let ps = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
    let ds = Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
    let captured = ps.clone();
    let p_handler = move |Json(value): Json<Value>| {
        captured.lock().unwrap().push(value);
        async { Json(json!({"prompt_token_ids":[1,2,3],
            "usage":{"prompt_tokens":3,"prompt_tokens_details":{"cached_tokens":2}},
            "kv_transfer_params":{"remote_prefill_cached_tokens":2}})) }
    };
    let p = serve(endpoints(Router::new()
        .route("/v1/chat/completions", post(p_handler.clone()))
        .route("/v1/completions", post(p_handler)))).await;
    let captured = ds.clone();
    let d_handler = move |Json(value): Json<Value>| {
        let streaming = value["stream"].as_bool() == Some(true);
        captured.lock().unwrap().push(value);
        async move {
            if streaming { Response::new(chunks(3, 1)) }
            else { Json(json!({"choices":[],"usage":{"prompt_tokens":3}})).into_response() }
        }
    };
    let d = serve(endpoints(Router::new()
        .route("/v1/chat/completions", post(d_handler.clone()))
        .route("/v1/completions", post(d_handler)))).await;
    let router = mock_router(&p, &d, limits(1800, 600, 150)).await;
    assert_eq!(router.prefill_force_length_cap, enabled, "actual constructor must read opt-in env");
    let mut calls = 0;
    for path in ["/v1/chat/completions", "/v1/completions"] {
        for (i, stop) in [Value::Null, json!("STOP"), json!(["STOP", "END"])].into_iter().enumerate() {
            let mut original = json!({"model":"mock","max_tokens":4096,"min_tokens":0,
                "ignore_eos":false,"stop":stop,"stop_token_ids":[2,7],"temperature":0.25,
                "top_p":0.75,"seed":42,"thinking_token_budget":128,"include_stop_str_in_output":true,
                "vllm_xargs":{"max_thinking_tokens":256,"opaque":{"keep":[1,"two",null]}},
                "stream":i!=1,"stream_options":{"include_usage":true}});
            if i == 1 { original.as_object_mut().unwrap().remove("stream_options"); }
            if path == "/v1/chat/completions" {
                original["messages"] = json!([{"role":"user","content":"hello"}]);
                original["max_completion_tokens"] = json!(2048);
                original["reasoning_effort"] = json!("max");
                original["chat_template_kwargs"] = json!({"thinking":true,"opaque":"preserve"});
            } else { original["prompt"] = json!("hello"); }
            let response = route_request(&router, original.clone(), path).await.unwrap();
            assert_load(&router, 0, 1);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
            assert!(!bytes.is_empty());
            assert_load(&router, 0, 0);
            let mut sent_p = ps.lock().unwrap()[calls].clone();
            let mut sent_d = ds.lock().unwrap()[calls].clone();
            calls += 1;
            let p_kv = sent_p.as_object_mut().unwrap().remove("kv_transfer_params").unwrap();
            let d_kv = sent_d.as_object_mut().unwrap().remove("kv_transfer_params").unwrap();
            let mut expected_p = VllmPDRouter::prepare_prefill_request(original.clone(), path);
            if enabled {
                expected_p["ignore_eos"] = json!(true);
                expected_p["min_tokens"] = json!(1);
                expected_p["stop"] = json!([]);
                expected_p["stop_token_ids"] = json!([]);
            }
            assert_eq!(sent_p, expected_p, "only the audited P fields may change");
            assert_eq!(sent_d, original, "D sampling/thinking/body must stay JSON-value equivalent");
            assert_eq!(p_kv["do_remote_decode"], true);
            assert_eq!(p_kv["do_remote_prefill"], false);
            let mut expected_kv = json!({"do_remote_decode":false,"do_remote_prefill":true,
                "transfer_id":p_kv["transfer_id"], "remote_bootstrap_addr":p.url,
                "remote_engine_id":"mock-P","remote_prefill_cached_tokens":2});
            if path == "/v1/chat/completions" { expected_kv["prompt_token_ids"] = json!([1,2,3]); }
            assert_eq!(d_kv, expected_kv, "existing Mooncake metadata/usage/IDs must survive");
        }
    }
    assert_eq!(ps.lock().unwrap().len(), calls, "no P replay");
    assert_eq!(ds.lock().unwrap().len(), calls, "no D replay");
    let response = route_request(&router, json!({"model":"mock","messages":[{"role":"user","content":"drop"}],
        "stream":true,"max_tokens":4,"stop":"STOP","stop_token_ids":[2]}), "/v1/chat/completions").await.unwrap();
    assert_load(&router, 0, 1);
    drop(response);
    assert_load(&router, 0, 0);
    assert_eq!(ps.lock().unwrap().len(), calls + 1, "no P replay on body drop");
    assert_eq!(ds.lock().unwrap().len(), calls + 1, "no D replay on body drop");
}

#[tokio::test]
async fn real_pd_path_slow_p_stream_prompt_ids_usage_and_body_drop() {
    let p_calls = Arc::new(AtomicUsize::new(0));
    let pc = p_calls.clone();
    let d_calls = Arc::new(AtomicUsize::new(0));
    let dc = d_calls.clone();
    let p = serve(endpoints(Router::new().route("/v1/chat/completions", post(move |Json(value): Json<Value>| {
        pc.fetch_add(1, Ordering::SeqCst);
        async move {
            assert_eq!(value["return_token_ids"], true); assert_eq!(value["max_tokens"], 1); assert_eq!(value["stream"], false);
            tokio::time::sleep(Duration::from_millis(180)).await;
            Json(json!({"prompt_token_ids":[1,2,3],"usage":{"prompt_tokens":3,"prompt_tokens_details":{"cached_tokens":2}}}))
        }
    })))).await;
    let d = serve(endpoints(Router::new().route(
        "/v1/chat/completions",
        post(move |Json(value): Json<Value>| {
            dc.fetch_add(1, Ordering::SeqCst);
            async move {
                assert_eq!(
                    value["kv_transfer_params"]["prompt_token_ids"],
                    json!([1, 2, 3])
                );
                assert_eq!(
                    value["kv_transfer_params"]["remote_prefill_cached_tokens"],
                    2
                );
                assert_eq!(value["kv_transfer_params"]["remote_engine_id"], "mock-P");
                Response::new(chunks(10, 25))
            }
        }),
    )))
    .await;
    let mut router = mock_router(&p, &d, limits(1800, 600, 150)).await;
    let response = route_response(&router).await.unwrap();
    assert_load(&router, 0, 1);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(bytes.len(), 10 * b"data: token\n\n".len());
    assert_load(&router, 0, 0);
    let response = route_response(&router).await.unwrap();
    assert_load(&router, 0, 1);
    drop(response);
    assert_load(&router, 0, 0);
    for config in [limits(1800, 600, 5), limits(300, 280, 150)] {
        router.pd_timeouts = config;
        router.pd_client = config.client().unwrap();
        let response = route_response(&router).await.unwrap();
        assert!(axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .is_err());
        assert_load(&router, 0, 0); // Both body-idle and total expiry release D/token demand.
    }
    router.pd_timeouts = limits(1800, 600, 150);
    router.pd_client = router.pd_timeouts.client().unwrap();
    let responses: Vec<_> = futures::future::join_all((0..32).map(|_| route_response(&router)))
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_load(&router, 0, 32);
    drop(responses);
    assert_load(&router, 0, 0);
    assert_eq!(p_calls.load(Ordering::SeqCst), 36);
    assert_eq!(d_calls.load(Ordering::SeqCst), 36);
}

#[tokio::test]
async fn real_pd_timeout_cancellation_and_missing_usage_release_admission() {
    let d_calls = Arc::new(AtomicUsize::new(0));
    let dc = d_calls.clone();
    let mode = Arc::new(AtomicUsize::new(0));
    let m = mode.clone();
    let p = serve(endpoints(Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let m = m.clone();
            async move {
                tokio::time::sleep(Duration::from_millis(180)).await;
                Json(if m.load(Ordering::SeqCst) == 0 {
                    json!({"prompt_token_ids":[1,2,3]})
                } else {
                    json!({"usage":{"prompt_tokens":3,"prompt_tokens_details":{"cached_tokens":0}}})
                })
            }
        }),
    )))
    .await;
    let d = serve(endpoints(Router::new().route(
        "/v1/chat/completions",
        post(move || {
            dc.fetch_add(1, Ordering::SeqCst);
            async { "unexpected" }
        }),
    )))
    .await;
    let mut router = mock_router(&p, &d, limits(1500, 50, 600)).await;
    assert!(route_response(&router).await.is_err());
    assert_load(&router, 0, 0);
    router.pd_timeouts = limits(1500, 600, 600);
    router.pd_client = router.pd_timeouts.client().unwrap();
    assert!(
        route_response(&router).await.is_err(),
        "missing measured usage must still fail closed"
    );
    assert_load(&router, 0, 0);
    mode.store(1, Ordering::SeqCst);
    assert!(
        route_response(&router).await.is_err(),
        "missing prompt IDs must still fail closed"
    );
    assert_load(&router, 0, 0);
    let router = Arc::new(router);
    let r = router.clone();
    let task = tokio::spawn(async move { route_response(&r).await });
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert_load(&router, 1, 1);
    task.abort();
    let _ = task.await;
    assert_load(&router, 0, 0);
    assert_eq!(d_calls.load(Ordering::SeqCst), 0);
}
