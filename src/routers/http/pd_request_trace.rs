//! Opt-in, observational direct-PD tracing. Never parses SSE or changes deadlines.
//! Set VLLM_ROUTER_PD_REQUEST_TRACE=1 for diagnostic runs only: per-stage INFO
//! adds load. Compare performance only with identical trace settings. Direct
//! sequential P/D is phase traced; concurrent paths emit a bridge/scope marker.
use axum::{body::Body, http::HeaderMap, response::Response};
use bytes::{Bytes, BytesMut};
use futures_util::{Stream, StreamExt};
use http_body::{Body as HttpBody, Frame, SizeHint};
use parking_lot::Mutex;
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};

const ENV: &str = "VLLM_ROUTER_PD_REQUEST_TRACE";

#[derive(Clone, Default)]
struct Events(Option<Arc<Mutex<State>>>);
struct State {
    started: Instant,
    incoming: Option<String>,
    pd_id: Option<String>,
    transfer_id: Option<String>,
    prefill: Option<String>,
    decode: Option<String>,
    p_rank: Option<usize>,
    d_rank: Option<usize>,
    last_stage: &'static str,
}
impl Events {
    fn event(&self, stage: &'static str, outcome: &'static str) {
        if let Some(state) = &self.0 {
            let mut s = state.lock();
            let previous_stage = s.last_stage;
            s.last_stage = stage;
            tracing::info!(target: "vllm_router_rs::pd_request_trace",
                stage, outcome, previous_stage,
                body_semantics = "raw_bytes_not_effective_token",
                elapsed_origin = "route_entry",
                elapsed_seconds = s.started.elapsed().as_secs_f64(),
                incoming_request_id = ?s.incoming, pd_request_id = ?s.pd_id,
                transfer_id = ?s.transfer_id,
                prefill_worker = ?s.prefill, prefill_rank = ?s.p_rank,
                decode_worker = ?s.decode, decode_rank = ?s.d_rank,
                "pd_request_trace");
        }
    }
}

/// Unique lifecycle owner. Cloned event observers do not affect request lifetime.
/// A dropped owner only logs: it does not abort, drain, release, or mutate routing state.
pub(super) struct RequestTrace(Events);
impl RequestTrace {
    pub(super) fn new(headers: Option<&HeaderMap>, direct: bool) -> Self {
        Self::with_enabled(headers, direct && std::env::var(ENV).as_deref() == Ok("1"))
    }
    fn with_enabled(headers: Option<&HeaderMap>, enabled: bool) -> Self {
        let events = if enabled {
            Events(Some(Arc::new(Mutex::new(State {
                started: Instant::now(),
                incoming: headers
                    .and_then(|h| h.get("x-request-id"))
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_owned),
                pd_id: None,
                transfer_id: None,
                prefill: None,
                decode: None,
                p_rank: None,
                d_rank: None,
                last_stage: "route_entry",
            }))))
        } else {
            Events::default()
        };
        let trace = Self(events);
        trace.stage("route_entry");
        trace
    }
    pub(super) fn enabled(&self) -> bool {
        self.0 .0.is_some()
    }
    pub(super) fn stage(&self, stage: &'static str) {
        self.0.event(stage, "observed");
    }
    pub(super) fn selected(&self, p: &str, pr: Option<usize>, d: &str, dr: Option<usize>) {
        if let Some(state) = &self.0 .0 {
            let mut s = state.lock();
            // Origins exclude URL credentials, paths, queries, and fragments.
            let origin = |value| {
                url::Url::parse(value)
                    .map(|u| u.origin().ascii_serialization())
                    .unwrap_or_else(|_| "<invalid-worker-url>".into())
            };
            s.prefill = Some(origin(p));
            s.decode = Some(origin(d));
            s.p_rank = pr;
            s.d_rank = dr;
        }
        self.stage("selected");
    }
    pub(super) fn bridge(&self, pd_id: &str, transfer_id: Option<&str>) {
        if let Some(state) = &self.0 .0 {
            let mut s = state.lock();
            s.pd_id = Some(pd_id.to_owned());
            s.transfer_id = transfer_id.map(str::to_owned);
        }
        self.stage("id_bridge");
    }
    pub(super) fn finish(&mut self, outcome: &'static str) {
        self.0.event("end", outcome);
        self.0 .0.take();
    }
    pub(super) fn observe_stream<S>(&self, inner: S) -> ObservedStream<S> {
        ObservedStream {
            inner,
            events: self.0.clone(),
            first: true,
        }
    }
    // Only the pre-existing NON-streaming logprobs merge branch calls this.
    // Same collection as PdResponse::bytes; never used to buffer streaming SSE.
    pub(super) async fn collect_body<S, E>(&self, inner: S) -> Result<Bytes, E>
    where
        S: Stream<Item = Result<Bytes, E>> + Unpin,
    {
        let mut stream = self.observe_stream(inner);
        let mut bytes = BytesMut::new();
        while let Some(chunk) = stream.next().await {
            bytes.extend_from_slice(&chunk?);
        }
        Ok(bytes.freeze())
    }
    pub(super) fn response(mut self, response: Response) -> Response {
        if self.0 .0.is_none() {
            return response;
        }
        if response.body().is_end_stream() {
            self.finish("body_eof");
            return response;
        }
        let (parts, inner) = response.into_parts();
        Response::from_parts(parts, Body::new(TracedBody { inner, trace: self }))
    }
}
impl Drop for RequestTrace {
    fn drop(&mut self) {
        // Before dispatch, an early route return is indistinguishable from cancellation.
        self.0.event("drop", "cancel_or_early_return");
    }
}

pub(super) struct ObservedStream<S> {
    inner: S,
    events: Events,
    first: bool,
}
impl<S, E> Stream for ObservedStream<S>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
{
    type Item = Result<Bytes, E>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let result = Pin::new(&mut self.inner).poll_next(cx);
        if let Poll::Ready(Some(Ok(bytes))) = &result {
            if self.first && !bytes.is_empty() {
                self.first = false;
                // Raw upstream bytes: role/heartbeat counts; NOT first effective token.
                self.events.event("d_first_nonempty_body", "observed");
            }
        }
        result
    }
}
struct TracedBody {
    inner: Body,
    trace: RequestTrace,
}
impl HttpBody for TracedBody {
    type Data = Bytes;
    type Error = axum::Error;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        let result = Pin::new(&mut self.inner).poll_frame(cx);
        match &result {
            Poll::Ready(None) => self.trace.finish("body_eof"),
            Poll::Ready(Some(Err(_))) => self.trace.finish("body_error"),
            _ => {}
        }
        if matches!(&result, Poll::Ready(Some(Ok(_)))) && self.inner.is_end_stream() {
            self.trace.finish("body_eof");
        }
        result
    }
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{stream, task::noop_waker_ref};
    use http_body_util::BodyExt;
    use std::{
        convert::Infallible,
        io::{self, Write},
        sync::atomic::{AtomicUsize, Ordering},
    };
    #[derive(Clone, Default)]
    struct Logs(Arc<std::sync::Mutex<Vec<u8>>>);
    impl Write for Logs {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Logs {
        type Writer = Self;
        fn make_writer(&'a self) -> Self {
            self.clone()
        }
    }
    fn capture() -> (Logs, tracing::subscriber::DefaultGuard) {
        let logs = Logs::default();
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_writer(logs.clone())
            .finish();
        (logs, tracing::subscriber::set_default(subscriber))
    }
    fn text(logs: &Logs) -> String {
        String::from_utf8(logs.0.lock().unwrap().clone()).unwrap()
    }
    #[test]
    fn disabled_is_silent_and_ids_bridge_without_secrets() {
        let (logs, _guard) = capture();
        let disabled = RequestTrace::with_enabled(None, false);
        disabled.stage("d_json_ready");
        drop(disabled);
        assert!(!RequestTrace::new(None, false).enabled());
        assert!(text(&logs).is_empty());
        let mut headers = HeaderMap::new();
        headers.insert("X-Request-ID", "run:trial:turn:a1".parse().unwrap());
        headers.insert("authorization", "Bearer SECRET".parse().unwrap());
        let mut trace = RequestTrace::with_enabled(Some(&headers), true);
        trace.selected(
            "http://user:PASS@p:8000/private?key=SECRET",
            Some(1),
            "http://d:8000",
            Some(7),
        );
        trace.bridge("___prefill_addr_p___decode_addr_d_uuid", Some("xfer-uuid"));
        trace.stage("p_dispatch");
        trace.stage("p_headers");
        trace.stage("p_body_done");
        trace.stage("d_dispatch");
        trace.stage("d_json_ready");
        trace.stage("d_headers");
        trace.finish("error");
        drop(trace);
        let out = text(&logs);
        for expected in [
            "run:trial:turn:a1",
            "___prefill_addr_p___decode_addr_d_uuid",
            "xfer-uuid",
            "http://p:8000",
            "p_body_done",
            "d_headers",
            "elapsed_seconds",
            "prefill_rank=Some(1)",
            "decode_rank=Some(7)",
        ] {
            assert!(out.contains(expected), "missing {expected}: {out}");
        }
        let mut last = 0.0;
        for line in out.lines() {
            let elapsed: f64 = line
                .split("elapsed_seconds=")
                .nth(1)
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap()
                .parse()
                .unwrap();
            assert!(elapsed >= last);
            last = elapsed;
        }
        let mut last = 0;
        for stage in [
            "route_entry",
            "selected",
            "id_bridge",
            "p_dispatch",
            "p_headers",
            "p_body_done",
            "d_dispatch",
            "d_json_ready",
            "d_headers",
            "end",
        ] {
            let needle = format!("stage={stage:?}");
            let index = out[last..].find(&needle).unwrap() + last;
            last = index + needle.len();
        }
        for secret in [
            "SECRET",
            "PASS",
            "private",
            "authorization",
            "cancel_or_early_return",
        ] {
            assert!(!out.contains(secret), "leaked {secret}: {out}");
        }
    }
    #[tokio::test]
    async fn bytes_unchanged_first_nonempty_once_and_eof_not_cancel() {
        let (logs, _guard) = capture();
        let trace = RequestTrace::with_enabled(None, true);
        let chunks = vec![
            Ok::<_, Infallible>(Bytes::new()),
            Ok(Bytes::from_static(b": heartbeat\n\n")),
            Ok(Bytes::from_static(b"data: token\n\n")),
        ];
        let body = Body::from_stream(trace.observe_stream(stream::iter(chunks)));
        let response = trace.response(Response::new(body));
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(bytes, ": heartbeat\n\ndata: token\n\n");
        let out = text(&logs);
        assert_eq!(
            out.matches(r#" stage="d_first_nonempty_body""#).count(),
            1,
            "{out}"
        );
        assert!(out.contains("body_eof"));
        assert!(!out.contains("cancel_or_early_return"));
    }
    #[tokio::test]
    async fn trailers_and_empty_body_are_preserved_without_false_cancel() {
        let (logs, _guard) = capture();
        let trace = RequestTrace::with_enabled(None, true);
        let mut trailers = HeaderMap::new();
        trailers.insert("x-fixture-trailer", "retained".parse().unwrap());
        let inner = http_body_util::StreamBody::new(stream::iter([
            Ok::<_, Infallible>(Frame::data(Bytes::from_static(b"payload"))),
            Ok(Frame::trailers(trailers)),
        ]));
        let collected = trace
            .response(Response::new(Body::new(inner)))
            .into_body()
            .collect()
            .await
            .unwrap();
        assert_eq!(
            collected.trailers().unwrap()["x-fixture-trailer"],
            "retained"
        );
        assert_eq!(collected.to_bytes(), "payload");
        let trace = RequestTrace::with_enabled(None, true);
        drop(trace.response(Response::new(Body::empty())));
        assert!(!text(&logs).contains("cancel_or_early_return"));
    }
    struct LastFrame {
        frame: Option<Frame<Bytes>>,
        polls: Arc<AtomicUsize>,
    }
    impl HttpBody for LastFrame {
        type Data = Bytes;
        type Error = Infallible;
        fn poll_frame(
            mut self: Pin<&mut Self>,
            _: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
            self.polls.fetch_add(1, Ordering::SeqCst);
            Poll::Ready(self.frame.take().map(Ok))
        }
        fn is_end_stream(&self) -> bool {
            self.frame.is_none()
        }
    }
    #[test]
    fn last_data_or_trailer_then_drop_is_eof_without_extra_poll() {
        let (logs, _guard) = capture();
        let mut trailers = HeaderMap::new();
        trailers.insert("x-last", "retained".parse().unwrap());
        for frame in [
            Frame::data(Bytes::from_static(b"last")),
            Frame::trailers(trailers),
        ] {
            let polls = Arc::new(AtomicUsize::new(0));
            let trace = RequestTrace::with_enabled(None, true);
            let inner = Body::new(LastFrame {
                frame: Some(frame),
                polls: polls.clone(),
            });
            let mut body = trace.response(Response::new(inner)).into_body();
            let frame =
                match Pin::new(&mut body).poll_frame(&mut Context::from_waker(noop_waker_ref())) {
                    Poll::Ready(Some(Ok(frame))) => frame,
                    _ => panic!("expected one unchanged frame"),
                };
            if frame.is_data() {
                assert_eq!(frame.into_data().unwrap(), "last");
            } else {
                assert_eq!(frame.into_trailers().unwrap()["x-last"], "retained");
            }
            assert!(body.is_end_stream());
            drop(body); // Consumer does NOT poll(None).
            assert_eq!(polls.load(Ordering::SeqCst), 1);
        }
        let out = text(&logs);
        assert_eq!(out.matches("body_eof").count(), 2);
        assert!(!out.contains("cancel_or_early_return"));
    }
    struct Pending(Arc<AtomicUsize>);
    impl Stream for Pending {
        type Item = Result<Bytes, io::Error>;
        fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Poll::Pending
        }
    }
    #[test]
    fn pending_is_single_poll_and_drop_logs_cancel_without_draining() {
        let (logs, _guard) = capture();
        let polls = Arc::new(AtomicUsize::new(0));
        let trace = RequestTrace::with_enabled(None, true);
        let body = Body::from_stream(trace.observe_stream(Pending(polls.clone())));
        let mut body = trace.response(Response::new(body)).into_body();
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        assert!(Pin::new(&mut body)
            .poll_frame(&mut Context::from_waker(noop_waker_ref()))
            .is_pending());
        assert_eq!(polls.load(Ordering::SeqCst), 1);
        drop(body);
        assert_eq!(polls.load(Ordering::SeqCst), 1);
        assert!(text(&logs).contains("cancel_or_early_return"));
        assert!(!text(&logs).contains("d_first_nonempty_body"));
    }
    #[tokio::test]
    async fn buffered_collection_and_errors_preserve_original_bytes_and_error() {
        let (logs, _guard) = capture();
        let trace = RequestTrace::with_enabled(None, true);
        let bytes = trace
            .collect_body(stream::iter([Ok::<_, io::Error>(Bytes::from_static(
                b"json",
            ))]))
            .await
            .unwrap();
        assert_eq!(bytes, "json");
        let body = Body::from_stream(trace.observe_stream(stream::iter([Err::<Bytes, _>(
            io::Error::other("sentinel"),
        )])));
        assert!(trace
            .response(Response::new(body))
            .into_body()
            .collect()
            .await
            .unwrap_err()
            .to_string()
            .contains("sentinel"));
        assert!(text(&logs).contains("body_error"));
    }
}
