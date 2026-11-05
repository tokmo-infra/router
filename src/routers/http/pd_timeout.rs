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
        reqwest::Client::builder()
            .connect_timeout(self.connect)
            .timeout(self.total)
            .retry(reqwest::retry::never())
            .pool_idle_timeout(Duration::from_secs(50))
            .pool_max_idle_per_host(500)
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
