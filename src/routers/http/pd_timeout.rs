//! Bounded timeouts for direct, sequential HTTP P/D dispatch. No retry or pump.
use crate::otel_http::{self, ClientRequestOptions};
use axum::http::HeaderMap;
use bytes::{Bytes, BytesMut};
use futures_util::{Stream, StreamExt};
use std::{
    future::Future,
    ops::Deref,
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};
use tokio::time::{Instant, Sleep};

#[derive(Clone, Copy, Debug)]
pub(super) struct PdTimeouts {
    pub connect: Duration,
    pub first_response: Duration,
    pub idle: Duration,
    pub total: Duration,
}

impl PdTimeouts {
    pub fn from_env(legacy_total_secs: u64) -> Result<Self, String> {
        Self::parse(legacy_total_secs, |name| std::env::var(name).ok())
    }

    fn parse(legacy: u64, get: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let seconds = |name: &str, default| -> Result<u64, String> {
            let n = match get(name) {
                Some(value) => value
                    .parse::<u64>()
                    .map_err(|_| format!("{name} must be positive integer seconds"))?,
                None => default,
            };
            if n == 0 || n > 86400 {
                return Err(format!("{name} must be in 1..=86400 seconds"));
            }
            Ok(n)
        };
        let total = seconds("VLLM_ROUTER_PD_TOTAL_TIMEOUT_SECS", legacy)?;
        let connect = seconds("VLLM_ROUTER_PD_CONNECT_TIMEOUT_SECS", 10.min(total))?;
        let first = seconds("VLLM_ROUTER_PD_FIRST_RESPONSE_TIMEOUT_SECS", total)?;
        let idle = seconds("VLLM_ROUTER_PD_IDLE_TIMEOUT_SECS", total)?;
        if [connect, first, idle].into_iter().any(|n| n > total) {
            return Err("PD connect/first-response/idle timeouts must not exceed PD total".into());
        }
        Ok(Self {
            connect: Duration::from_secs(connect),
            first_response: Duration::from_secs(first),
            idle: Duration::from_secs(idle),
            total: Duration::from_secs(total),
        })
    }

    pub fn client(self) -> Result<reqwest::Client, String> {
        let max_idle = Self::pool_limit(|name| std::env::var(name).ok())?;
        self.client_with_pool_limit(max_idle)
    }

    fn pool_limit(get: impl Fn(&str) -> Option<String>) -> Result<usize, String> {
        match get("VLLM_ROUTER_PD_DISABLE_HTTP_POOL").as_deref() {
            None | Some("0") => Ok(500),
            Some("1") => Ok(0),
            _ => Err("VLLM_ROUTER_PD_DISABLE_HTTP_POOL must be 0 or 1".into()),
        }
    }

    fn client_with_pool_limit(self, max_idle: usize) -> Result<reqwest::Client, String> {
        // Opt-in isolation of HTTP/1 busy-connection reuse during long SSE.
        // Zero disables this PD client's pool; this is not Connection: close
        // on shared clients and does not change tokenizer/KV caches or retries.
        reqwest::Client::builder()
            .connect_timeout(self.connect)
            .timeout(self.total)
            .retry(reqwest::retry::never())
            .pool_idle_timeout(Duration::from_secs(50))
            .pool_max_idle_per_host(max_idle)
            .tcp_nodelay(true)
            .tcp_keepalive(Duration::from_secs(30))
            .build()
            .map_err(|e| e.to_string())
    }

    pub fn start(self) -> PdBudget {
        PdBudget {
            config: self,
            deadline: Instant::now() + self.total,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum PdTimeoutError {
    #[error("PD {0} timeout")]
    Timeout(&'static str),
    #[error(transparent)]
    Http(#[from] reqwest::Error),
}
fn expired(phase: &'static str) -> PdTimeoutError {
    metrics::counter!("vllm_router_pd_timeout_total", "phase" => phase).increment(1);
    PdTimeoutError::Timeout(phase)
}

// Fixed-cardinality labels only: no prompt, URL, request ID, or credentials.
struct PhaseTimer {
    stage: &'static str,
    phase: &'static str,
    start: Instant,
    outcome: &'static str,
}
impl PhaseTimer {
    fn new(stage: &'static str, phase: &'static str, start: Instant) -> Self {
        metrics::gauge!("vllm_router_pd_stage_waiting", "stage" => stage, "phase" => phase)
            .increment(1.0);
        Self {
            stage,
            phase,
            start,
            outcome: "cancelled",
        }
    }
}
impl Drop for PhaseTimer {
    fn drop(&mut self) {
        metrics::gauge!("vllm_router_pd_stage_waiting", "stage" => self.stage, "phase" => self.phase).decrement(1.0);
        metrics::histogram!("vllm_router_pd_stage_wait_seconds", "stage" => self.stage,
            "phase" => self.phase, "outcome" => self.outcome)
        .record(self.start.elapsed().as_secs_f64());
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PdBudget {
    config: PdTimeouts,
    deadline: Instant,
}
impl PdBudget {
    pub async fn until_total<T>(
        self,
        future: impl Future<Output = T>,
    ) -> Result<T, PdTimeoutError> {
        tokio::time::timeout_at(self.deadline, future)
            .await
            .map_err(|_| expired("total"))
    }

    pub async fn send(
        self,
        builder: reqwest::RequestBuilder,
        headers: Option<&HeaderMap>,
        options: ClientRequestOptions<'_>,
    ) -> Result<PdResponse, PdTimeoutError> {
        let now = Instant::now();
        let stage = match options.request_phase {
            Some("prefill") => "prefill",
            Some("decode") => "decode",
            _ => "other",
        };
        let mut timer = PhaseTimer::new(stage, "response_headers", now);
        if now >= self.deadline {
            return Err(expired("total"));
        }
        let first_deadline = (now + self.config.first_response).min(self.deadline);
        // Per-request override is essential: an inherited 600s client timeout
        // otherwise continues ticking throughout the streaming response body.
        let send =
            otel_http::send_client_request(builder.timeout(self.deadline - now), headers, options);
        let result = tokio::time::timeout_at(first_deadline, send).await;
        timer.outcome = match &result {
            Ok(Ok(_)) => "ok",
            Ok(Err(_)) => "transport_error",
            Err(_) => "timeout",
        };
        let response = result
            .map_err(|_| {
                expired(if Instant::now() >= self.deadline {
                    "total"
                } else {
                    "first_response"
                })
            })?
            .map_err(|e| {
                if e.is_connect() && e.is_timeout() {
                    expired("connect")
                } else {
                    e.into()
                }
            })?;
        Ok(PdResponse {
            response,
            budget: self,
            first_deadline,
            stage,
            stage_started: now,
        })
    }
}

pub(super) struct PdResponse {
    response: reqwest::Response,
    budget: PdBudget,
    first_deadline: Instant,
    stage: &'static str,
    stage_started: Instant,
}
impl Deref for PdResponse {
    type Target = reqwest::Response;
    fn deref(&self) -> &Self::Target {
        &self.response
    }
}
impl PdResponse {
    pub fn bytes_stream(self) -> PdBodyStream {
        PdBodyStream {
            inner: Some(Box::pin(self.response.bytes_stream())),
            total: Box::pin(tokio::time::sleep_until(self.budget.deadline)),
            first: Some(Box::pin(tokio::time::sleep_until(self.first_deadline))),
            idle: Box::pin(tokio::time::sleep(self.budget.config.idle)),
            idle_active: false,
            idle_duration: self.budget.config.idle,
            stage: self.stage,
            request_started: self.budget.deadline - self.budget.config.total,
            first_observation: Some(PhaseTimer::new(
                self.stage,
                "first_body",
                self.stage_started,
            )),
        }
    }
    pub async fn bytes(self) -> Result<Bytes, PdTimeoutError> {
        let mut stream = self.bytes_stream();
        let mut bytes = BytesMut::new();
        while let Some(chunk) = stream.next().await {
            bytes.extend_from_slice(&chunk?);
        }
        Ok(bytes.freeze())
    }
}

type Upstream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;
pub(super) struct PdBodyStream {
    inner: Option<Upstream>,
    total: Pin<Box<Sleep>>,
    first: Option<Pin<Box<Sleep>>>,
    idle: Pin<Box<Sleep>>,
    idle_active: bool,
    idle_duration: Duration,
    stage: &'static str,
    request_started: Instant,
    first_observation: Option<PhaseTimer>,
}
impl PdBodyStream {
    fn fail(&mut self, phase: &'static str) -> Poll<Option<Result<Bytes, PdTimeoutError>>> {
        self.inner.take(); // Cancel upstream immediately; no background drain.
        if let Some(mut observation) = self.first_observation.take() {
            observation.outcome = "timeout";
        }
        tracing::warn!(
            stage = self.stage,
            phase,
            elapsed_secs = self.request_started.elapsed().as_secs_f64(),
            "PD upstream deadline expired"
        );
        Poll::Ready(Some(Err(expired(phase))))
    }
}
impl Stream for PdBodyStream {
    type Item = Result<Bytes, PdTimeoutError>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.inner.is_none() {
            return Poll::Ready(None);
        }
        if self.total.as_mut().poll(cx).is_ready() {
            return self.fail("total");
        }
        if let Some(first) = &mut self.first {
            if first.as_mut().poll(cx).is_ready() {
                return self.fail("first_response");
            }
        }
        match self.inner.as_mut().unwrap().as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                if !bytes.is_empty() && self.first.take().is_some() {
                    if let Some(mut observation) = self.first_observation.take() {
                        observation.outcome = "ok";
                    }
                    if self.stage == "decode" {
                        // First body byte is NOT necessarily the first generated token (SSE role/heartbeat).
                        metrics::histogram!("vllm_router_pd_first_decode_body_seconds")
                            .record(self.request_started.elapsed().as_secs_f64());
                    }
                }
                if !bytes.is_empty() {
                    self.idle_active = false;
                }
                Poll::Ready(Some(Ok(bytes)))
            }
            Poll::Ready(Some(Err(e))) => {
                self.inner.take();
                if let Some(mut observation) = self.first_observation.take() {
                    observation.outcome = "transport_error";
                }
                Poll::Ready(Some(Err(e.into())))
            }
            Poll::Ready(None) => {
                self.inner.take();
                if let Some(mut observation) = self.first_observation.take() {
                    observation.outcome = "empty_body";
                }
                Poll::Ready(None)
            }
            Poll::Pending => {
                // Only time a pending upstream read AFTER first data. Downstream
                // pauses between chunks are backpressure, not upstream idleness.
                if self.first.is_none() {
                    if !self.idle_active {
                        let deadline = Instant::now() + self.idle_duration;
                        self.idle.as_mut().reset(deadline);
                        self.idle_active = true;
                    }
                    if self.idle.as_mut().poll(cx).is_ready() {
                        return self.fail("body_idle");
                    }
                }
                Poll::Pending
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabling_pd_pool_is_explicit_and_defaults_are_unchanged() {
        assert_eq!(PdTimeouts::pool_limit(|_| None).unwrap(), 500);
        assert_eq!(PdTimeouts::pool_limit(|_| Some("0".into())).unwrap(), 500);
        assert_eq!(PdTimeouts::pool_limit(|_| Some("1".into())).unwrap(), 0);
        assert!(PdTimeouts::pool_limit(|_| Some("true".into())).is_err());
    }

    #[tokio::test]
    async fn disabled_pd_pool_does_not_wait_for_another_response_body() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::{TcpListener, TcpStream};
        use tokio::sync::oneshot;

        async fn read_head(socket: &mut TcpStream) {
            let mut head = Vec::new();
            while !head.ends_with(b"\r\n\r\n") {
                head.push(socket.read_u8().await.unwrap());
                assert!(head.len() < 16_384);
            }
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let (release, released) = oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            let (mut a, _) = listener.accept().await.unwrap();
            read_head(&mut a).await;
            a.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\n")
                .await
                .unwrap();
            let (mut b, _) = listener.accept().await.unwrap();
            read_head(&mut b).await;
            b.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .await
                .unwrap();
            released.await.unwrap();
            a.write_all(b"done").await.unwrap();
        });
        let client = PdTimeouts::parse(30, |_| None)
            .unwrap()
            .client_with_pool_limit(0)
            .unwrap();
        let first = client.get(&url).send().await.unwrap();
        let second = tokio::time::timeout(Duration::from_secs(3), client.get(&url).send())
            .await
            .expect("second request waited for the first response body")
            .unwrap();
        assert_eq!(second.status(), reqwest::StatusCode::OK);
        release.send(()).unwrap();
        assert_eq!(first.text().await.unwrap(), "done");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn disabled_pd_pool_post_sse_uses_new_connections_after_eof_and_cancel() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::{TcpListener, TcpStream};
        use tokio::sync::oneshot;

        const FIRST: &[u8] = b"data: {\"delta\":{\"content\":\"held\"}}\n\n";
        const COMPLETE: &[u8] = b"data: {\"delta\":{\"content\":\"raw\"}}\n\ndata: [DONE]\n\n";
        async fn read_post(socket: &mut TcpStream, id: &str) {
            let mut head = Vec::new();
            while !head.ends_with(b"\r\n\r\n") {
                head.push(socket.read_u8().await.unwrap());
                assert!(head.len() < 16_384);
            }
            let head = String::from_utf8(head).unwrap();
            assert!(head.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"));
            let headers = head.to_ascii_lowercase();
            assert!(!headers.lines().any(|line| line == "connection: close"));
            assert!(headers
                .lines()
                .any(|line| line == "content-type: application/json"));
            let length: usize = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length: "))
                .unwrap()
                .parse()
                .unwrap();
            assert!(length < 1024);
            let mut body = vec![0; length];
            socket.read_exact(&mut body).await.unwrap();
            assert_eq!(
                serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
                serde_json::json!({"id": id, "stream": true})
            );
        }
        async fn sse(socket: &mut TcpStream, body: &[u8], eof: bool) {
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
                .await
                .unwrap();
            socket
                .write_all(format!("{:x}\r\n", body.len()).as_bytes())
                .await
                .unwrap();
            socket.write_all(body).await.unwrap();
            socket.write_all(b"\r\n").await.unwrap();
            if eof {
                socket.write_all(b"0\r\n\r\n").await.unwrap();
            }
        }

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!(
            "http://{}/v1/chat/completions",
            listener.local_addr().unwrap()
        );
        let (cancelled, cancellation_seen) = oneshot::channel();
        let mut server = tokio::spawn(async move {
            let (mut a, peer_a) = listener.accept().await.unwrap();
            read_post(&mut a, "A").await;
            sse(&mut a, FIRST, false).await;
            let mut peers = vec![peer_a];
            let mut posts = vec!["A"];
            // Keep completed B/C sockets open: a pooling client must not pass
            // merely because this fixture closed the old connections for it.
            let mut completed = Vec::new();
            for id in ["B", "C"] {
                let (mut socket, peer) = listener.accept().await.unwrap();
                read_post(&mut socket, id).await;
                peers.push(peer);
                posts.push(id);
                sse(&mut socket, COMPLETE, true).await;
                completed.push(socket);
            }
            let mut byte = [0];
            assert_eq!(a.read(&mut byte).await.unwrap(), 0, "A must be cancelled");
            cancelled.send(()).unwrap();
            let (mut d, peer_d) = listener.accept().await.unwrap();
            read_post(&mut d, "D").await;
            peers.push(peer_d);
            posts.push("D");
            sse(&mut d, COMPLETE, true).await;
            (posts, peers)
        });
        let outcome = tokio::time::timeout(Duration::from_secs(8), async {
            let client = PdTimeouts::parse(30, |_| None)
                .unwrap()
                .client_with_pool_limit(0)
                .unwrap();
            let post = |id: &str| {
                client
                    .post(&url)
                    .json(&serde_json::json!({"id": id, "stream": true}))
            };
            let response_a = post("A").send().await.unwrap();
            assert_eq!(response_a.status(), reqwest::StatusCode::OK);
            let mut a = response_a.bytes_stream();
            let mut prefix = Vec::new();
            while prefix.len() < FIRST.len() {
                prefix.extend_from_slice(&a.next().await.unwrap().unwrap());
            }
            assert_eq!(prefix, FIRST);
            for id in ["B", "C"] {
                let response = post(id).send().await.unwrap();
                assert_eq!(response.status(), reqwest::StatusCode::OK);
                assert_eq!(response.bytes().await.unwrap().as_ref(), COMPLETE);
            }
            // A has never reached EOF; drop is cancellation, not a body drain.
            drop(a);
            cancellation_seen.await.unwrap();
            let response_d = post("D").send().await.unwrap();
            assert_eq!(response_d.status(), reqwest::StatusCode::OK);
            assert_eq!(response_d.bytes().await.unwrap().as_ref(), COMPLETE);
            let (posts, peers) = (&mut server).await.unwrap();
            assert_eq!(
                posts,
                ["A", "B", "C", "D"],
                "one wire POST per logical request"
            );
            assert_eq!(
                peers.iter().collect::<std::collections::HashSet<_>>().len(),
                4
            );
        })
        .await;
        // Abort the owned listener on deadline; no detached fixture can hang CI.
        if outcome.is_err() {
            server.abort();
            let _ = server.await;
        }
        outcome.expect("pool0 POST/SSE fixture exceeded its total 8s deadline");
    }

    #[test]
    fn stage_observations_finish_and_cancel_without_leaking_waiters() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let mut completed = PhaseTimer::new("prefill", "response_headers", Instant::now());
            completed.outcome = "ok";
            drop(completed);
            drop(PhaseTimer::new("decode", "first_body", Instant::now()));
        });
        let rendered = handle.render();
        let waiters: Vec<_> = rendered
            .lines()
            .filter(|line| line.starts_with("vllm_router_pd_stage_waiting{"))
            .collect();
        assert_eq!(waiters.len(), 2, "{rendered}");
        assert!(
            waiters.iter().all(|line| line.ends_with(" 0")),
            "{rendered}"
        );
        assert!(rendered.contains("outcome=\"ok\""));
        assert!(rendered.contains("outcome=\"cancelled\""));
    }
    #[test]
    fn finite_config_defaults_and_validation() {
        let c = PdTimeouts::parse(600, |_| None).unwrap();
        assert_eq!(c.total, Duration::from_secs(600));
        assert_eq!(c.first_response, c.total);
        for value in ["0", "-1", "NaN", "86401"] {
            assert!(
                PdTimeouts::parse(600, |key| (key == "VLLM_ROUTER_PD_TOTAL_TIMEOUT_SECS")
                    .then(|| value.into()))
                .is_err()
            );
        }
        assert!(
            PdTimeouts::parse(600, |key| (key == "VLLM_ROUTER_PD_IDLE_TIMEOUT_SECS")
                .then(|| "601".into()))
            .is_err()
        );
    }
    #[tokio::test]
    async fn stream_is_lazy_and_drop_cancels_its_source() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        struct Probe(Arc<AtomicUsize>, Arc<AtomicUsize>);
        impl Stream for Probe {
            type Item = Result<Bytes, reqwest::Error>;
            fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Poll::Ready(Some(Ok(Bytes::from_static(b"one"))))
            }
        }
        impl Drop for Probe {
            fn drop(&mut self) {
                self.1.fetch_add(1, Ordering::SeqCst);
            }
        }
        let polls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let now = Instant::now();
        let mut body = PdBodyStream {
            inner: Some(Box::pin(Probe(polls.clone(), drops.clone()))),
            total: Box::pin(tokio::time::sleep_until(now + Duration::from_secs(2))),
            first: None,
            idle: Box::pin(tokio::time::sleep(Duration::from_millis(10))),
            idle_active: false,
            idle_duration: Duration::from_millis(10),
            stage: "decode",
            request_started: now,
            first_observation: None,
        };
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        body.next().await.unwrap().unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(polls.load(Ordering::SeqCst), 1, "no background reader/pump");
        body.next().await.unwrap().unwrap();
        drop(body);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
