//! Opt-in PD routing by estimated prefill work and capacity-normalized decode KV.
//! Token counts are estimates on serialized model content, not native cache-hit metrics.
use crate::core::Worker;
use lru::LruCache;
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::{HashMap, HashSet}, num::NonZeroUsize, sync::Arc, time::{Duration, Instant}};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Deserialize)]
pub(super) struct Capacity {
    pub kv_tokens: f64,
    pub max_seqs: f64,
}
#[derive(Debug, Deserialize)]
struct Config {
    tokenizer_path: String,
    workers: HashMap<String, Capacity>,
    #[serde(default = "default_headroom")]
    kv_fraction: f64,
    #[serde(default)]
    prefill_kv_guard: bool,
}
fn default_headroom() -> f64 { 0.85 }
const TOKENIZATION_CONCURRENCY_ENV: &str = "VLLM_ROUTER_PD_TOKENIZATION_CONCURRENCY";
const BPE_CACHE_CAPACITY_ENV: &str = "VLLM_ROUTER_PD_BPE_CACHE_CAPACITY";
const MAX_TOKENIZATION_CONCURRENCY: usize = 32;
const MAX_BPE_CACHE_CAPACITY: usize = 16_384;

fn bounded_setting(name: &str, value: Option<&str>, default: usize, min: usize, max: usize) -> Result<usize, String> {
    match value {
        None => Ok(default),
        Some(raw) => {
            let parsed = raw.parse::<usize>().map_err(|_| format!("{name} must be an integer in {min}..={max}"))?;
            if !(min..=max).contains(&parsed) { return Err(format!("{name} must be in {min}..={max}")); }
            Ok(parsed)
        }
    }
}
fn env_setting(name: &str) -> Result<Option<String>, String> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(format!("invalid {name}: {e}")),
    }
}
fn tokenization_concurrency(cpus: usize, override_value: Option<&str>) -> Result<usize, String> {
    let max = cpus.max(1).min(MAX_TOKENIZATION_CONCURRENCY);
    let default = if cpus > 4 { (cpus - 4).min(max) } else { max };
    bounded_setting(TOKENIZATION_CONCURRENCY_ENV, override_value, default, 1, max)
}
fn bpe_cache_capacity(override_value: Option<&str>) -> Result<usize, String> {
    // Leave the proven configuration unchanged until workload-specific CPU tests justify enabling it.
    bounded_setting(BPE_CACHE_CAPACITY_ENV, override_value, 0, 0, MAX_BPE_CACHE_CAPACITY)
}
fn set_bpe_cache_capacity(tokenizer: &mut Tokenizer, capacity: usize) {
    // Tokenizers 0.22.2 deserializes a BPE with a shared, bounded cache. Entries
    // are word-level (<256 bytes), not copies of complete prompt prefixes.
    if let tokenizers::models::ModelWrapper::BPE(mut model) = tokenizer.get_model().clone() {
        model.resize_cache(capacity);
        tokenizer.with_model(model);
    }
}
#[derive(Debug, Default)]
struct Demand { requests: usize, kv: f64, work: f64 }
#[derive(Debug, Clone, Copy)]
struct LiveKv { usage: f64, observed: Instant }
#[derive(Debug)]
struct State {
    live_kv: HashMap<String, LiveKv>,
    demand: HashMap<String, Demand>,
    // Completed prefill page-prefix observations, with bounded memory and TTL.
    // Shared across P workers because their Mooncake Store is shared.
    prefixes: LruCache<[u8; 32], Instant>,
}
impl State {
    fn expire_live_kv(&mut self, now: Instant) {
        self.live_kv.retain(|_, row| now.saturating_duration_since(row.observed) < Duration::from_secs(5));
    }
}
#[derive(Debug)]
pub(super) struct TokenBalancer {
    config: Config,
    tokenizer: Tokenizer,
    pub(super) tokenization_slots: Arc<tokio::sync::Semaphore>,
    state: Mutex<State>,
}
#[derive(Debug)]
pub(super) struct Estimate {
    tokens: usize,
    output: usize,
    pages: Vec<[u8; 32]>,
}
// Fixed phase labels only; dropping also observes cancelled/error paths.
struct EstimationTimer { phase: &'static str, start: Instant }
impl EstimationTimer {
    fn start(phase: &'static str) -> Self { Self { phase, start: Instant::now() } }
}
impl Drop for EstimationTimer {
    fn drop(&mut self) {
        metrics::histogram!("vllm_router_pd_estimation_seconds", "phase" => self.phase)
            .record(self.start.elapsed().as_secs_f64());
    }
}

// Gauges represent waiters and all permit holders, including blocking-pool queue
// time. RAII covers errors, panics and cancellation; the active guard moves with
// the owned semaphore permit into the blocking closure.
struct OccupancyGuard(&'static str);
impl OccupancyGuard {
    fn new(name: &'static str) -> Self {
        metrics::gauge!(name).increment(1.0);
        Self(name)
    }
}
impl Drop for OccupancyGuard {
    fn drop(&mut self) { metrics::gauge!(self.0).decrement(1.0); }
}
async fn run_tokenization<T: Send + 'static, F: FnOnce() -> Result<T, String> + Send + 'static>(
    slots: Arc<tokio::sync::Semaphore>, make_work: impl FnOnce() -> F,
) -> Result<T, String> {
    let wait = EstimationTimer::start("slot_wait");
    let waiting = OccupancyGuard::new("vllm_router_pd_tokenization_waiting");
    let slot = slots.acquire_owned().await.map_err(|e| e.to_string())?;
    let active = OccupancyGuard::new("vllm_router_pd_tokenization_active");
    drop(waiting);
    drop(wait);
    let work = make_work();
    let blocking_wait = EstimationTimer::start("blocking_wait");
    tokio::task::spawn_blocking(move || {
        let _slot = slot;
        let _active = active;
        drop(blocking_wait);
        let _compute = EstimationTimer::start("compute");
        work()
    }).await.map_err(|e| e.to_string())?
}
#[derive(Debug)]
pub(super) struct Reservation {
    owner: Arc<TokenBalancer>,
    url: String,
    kv: f64,
    work: f64,
    pages: Vec<[u8;32]>,
    tokens: usize,
    started: Instant,
    stage: &'static str,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        let mut s = self.owner.state.lock();
        if let Some(d) = s.demand.get_mut(&self.url) {
            d.requests = d.requests.saturating_sub(1);
            d.kv = (d.kv - self.kv).max(0.0);
            d.work = (d.work - self.work).max(0.0);
            metrics::gauge!("vllm_router_pd_assigned_requests", "worker" => self.url.clone(), "stage" => self.stage).set(d.requests as f64);
            metrics::gauge!("vllm_router_pd_assigned_kv_tokens_estimate", "worker" => self.url.clone(), "stage" => self.stage).set(d.kv);
        }
    }
}
impl Reservation {
    pub(super) fn complete_prefill(self, response: &Value) {
        // Only successful P responses seed prefix observations; errors/cancellation do not.
        if response.pointer("/usage/prompt_tokens").and_then(Value::as_u64).is_some() {
            let now = Instant::now();
            let mut s = self.owner.state.lock();
            for page in &self.pages { s.prefixes.put(*page, now); }
            metrics::histogram!("vllm_router_pd_prefill_elapsed_seconds", "worker" => self.url.clone()).record(self.started.elapsed().as_secs_f64());
            if let Some(n) = response.pointer("/usage/prompt_tokens").and_then(Value::as_u64) {
                metrics::histogram!("vllm_router_pd_prompt_estimate_ratio").record(n as f64 / self.tokens.max(1) as f64);
            }
        }
        // Drop releases exactly this P allocation; D remains reserved through stream EOF.
    }
}
fn decode_score(d: &Demand, cap: &Capacity, tokens: f64, fraction: f64) -> f64 {
    let kv = (d.kv + tokens) / (cap.kv_tokens * fraction);
    let requests = (d.requests + 1) as f64 / cap.max_seqs;
    // Short requests favor smaller batches; long ones favor KV headroom.
    let length_weight = (tokens / 131072.0).clamp(0.25, 3.0);
    requests + length_weight * kv + 100.0 * (kv - 1.0).max(0.0)
}
fn prefill_score(d: &Demand, cap: &Capacity, work: f64, kv: f64, fraction: f64, live: Option<f64>) -> f64 {
    let assigned = (d.kv + kv) / (cap.kv_tokens * fraction);
    let live_excess = live.map(|u| ((u - fraction) / (1.0 - fraction).max(0.01)).max(0.0)).unwrap_or(0.0);
    let compute = (d.work + work) / cap.max_seqs;
    compute * (1.0 + 2.0 * assigned.powi(2))
        + 100_000.0 * (assigned - 1.0).max(0.0).powi(2)
        + 100_000.0 * live_excess.powi(2)
        + d.requests as f64 * 256.0
}
fn live_kv_rows(base: &str, body: &str, now: Instant) -> Vec<(String, LiveKv)> {
    body.lines().filter(|l|l.starts_with("vllm:kv_cache_usage_perc{"))
        .filter_map(|l| {
            let rank = l.split("engine=\"").nth(1)?.split('"').next()?.parse::<usize>().ok()?;
            let usage = l.split_whitespace().nth(1)?.parse::<f64>().ok()?;
            if !usage.is_finite() || !(0.0..=1.0).contains(&usage) { return None; }
            Some((format!("{}@{}",base,rank),LiveKv{usage,observed:now}))
        }).collect()
}
fn prefill_work(q: f64, c: f64) -> f64 {
    // Linear token work + attention q*(c+q/2), normalized to 128K.
    // Deliberately a heuristic, not an offline-fitted nanosecond cost model.
    q + q * (c + q * 0.5) / 131072.0
}
impl TokenBalancer {
    pub(super) fn from_env() -> Result<Option<Arc<Self>>, String> {
        let Some(path) = std::env::var_os("VLLM_ROUTER_PD_TOKEN_BALANCE_CONFIG") else { return Ok(None) };
        let config: Config = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        if !config.kv_fraction.is_finite() || config.kv_fraction <= 0.0 || config.kv_fraction > 1.0 || config.workers.is_empty()
            || config.workers.values().any(|c| !c.kv_tokens.is_finite() || c.kv_tokens <= 0.0 || !c.max_seqs.is_finite() || c.max_seqs <= 0.0) {
            return Err("invalid PD token-balance capacities".into());
        }
        // available_parallelism observes platform affinity and, on Linux, cgroup
        // CPU limits where supported. Fall back conservatively if detection fails.
        let cpus = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
        let concurrency = tokenization_concurrency(cpus, env_setting(TOKENIZATION_CONCURRENCY_ENV)?.as_deref())?;
        let cache_capacity = bpe_cache_capacity(env_setting(BPE_CACHE_CAPACITY_ENV)?.as_deref())?;
        let mut tokenizer = Tokenizer::from_file(&config.tokenizer_path).map_err(|e| e.to_string())?;
        set_bpe_cache_capacity(&mut tokenizer, cache_capacity);
        tracing::info!(workers=config.workers.len(), kv_fraction=config.kv_fraction, effective_cpus=cpus,
            tokenization_concurrency=concurrency, bpe_cache_capacity=cache_capacity,
            "PD token balance enabled (heuristic cost; estimated token demand)");
        metrics::gauge!("vllm_router_pd_tokenization_capacity").set(concurrency as f64);
        metrics::gauge!("vllm_router_pd_tokenization_active").set(0.0);
        metrics::gauge!("vllm_router_pd_tokenization_waiting").set(0.0);
        let enabled=config.prefill_kv_guard;
        let result=Arc::new(Self { config, tokenizer, tokenization_slots: Arc::new(tokio::sync::Semaphore::new(concurrency)), state: Mutex::new(State {
            demand: HashMap::new(), live_kv:HashMap::new(), prefixes: LruCache::new(NonZeroUsize::new(262144).unwrap()),
        }) });
        if enabled { Self::start_kv_watch(&result); }
        Ok(Some(result))
    }
    fn start_kv_watch(owner: &Arc<Self>) {
        let weak=Arc::downgrade(owner);
        tokio::spawn(async move {
            let client=match reqwest::Client::builder().timeout(Duration::from_secs(2)).no_proxy().build() {
                Ok(v)=>v,Err(e)=>{tracing::warn!(error=%e,"PD KV metric watcher unavailable");return;}
            };
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
                let Some(owner)=weak.upgrade() else {return};
                let bases:Vec<_>=owner.config.workers.keys().cloned().collect();
                let results=futures::future::join_all(bases.into_iter().map(|base| {
                    let client=client.clone();
                    async move {
                        let response=client.get(format!("{}/metrics",base)).send().await.ok()?;
                        if !response.status().is_success(){return None;}
                        let body=response.text().await.ok()?;
                        Some(live_kv_rows(&base,&body,Instant::now()))
                    }
                })).await;
                let mut state=owner.state.lock();
                state.expire_live_kv(Instant::now());
                for rows in results.into_iter().flatten() {
                    for (url,row) in rows {
                        metrics::gauge!("vllm_router_pd_live_kv_usage", "worker"=>url.clone()).set(row.usage);
                        state.live_kv.insert(url,row);
                    }
                }
            }
        });
    }
    pub(super) async fn estimate_async(self: &Arc<Self>, request: &Value) -> Result<Estimate, String> {
        let _total = EstimationTimer::start("total");
        run_tokenization(self.tokenization_slots.clone(), || {
            let owner = self.clone();
            let value = request.clone();
            move || owner.estimate(&value)
        }).await
    }
    pub(super) fn estimate(&self, request: &Value) -> Result<Estimate, String> {
        // Keep tools ahead of messages, independent of temperature/stream/max_tokens.
        // No inference is sent, and no benchmark reference token counts are consumed.
        let content = serde_json::to_string(&serde_json::json!({
            "tools": request.get("tools"), "messages": request.get("messages"),
            "prompt": request.get("prompt"), "cache_salt": request.get("cache_salt"),
            "model": request.get("model"), "chat_template_kwargs": request.get("chat_template_kwargs")
        })).map_err(|e| e.to_string())?;
        let encoded = self.tokenizer.encode(content, false).map_err(|e| e.to_string())?;
        let ids = encoded.get_ids();
        let mut hash = Sha256::new();
        hash.update(serde_json::to_vec(&serde_json::json!([request.get("model"),request.get("cache_salt"),request.get("chat_template_kwargs")])).map_err(|e|e.to_string())?);
        let mut pages = Vec::with_capacity(ids.len() / 256);
        for page in ids.chunks_exact(256) {
            for id in page { hash.update(id.to_le_bytes()); }
            pages.push(hash.clone().finalize().into());
        }
        Ok(Estimate { tokens: ids.len().max(1), pages,
            output: request.get("max_completion_tokens").or_else(||request.get("max_tokens")).and_then(Value::as_u64).unwrap_or(4096) as usize })
    }
    pub(super) fn select(self: &Arc<Self>, ps: &[Arc<dyn Worker>], ds: &[Arc<dyn Worker>], e: Estimate)
        -> Option<(usize, usize, Reservation, Reservation)> {
        let mut s = self.state.lock();
        let now = Instant::now();
        s.expire_live_kv(now);
        let matched = e.pages.iter().take_while(|h| s.prefixes.peek(*h).is_some_and(|t| now.duration_since(*t) < Duration::from_secs(60))).count();
        let c = (matched.saturating_sub(1) * 256).min(e.tokens.saturating_sub(1)) as f64;
        let q = (e.tokens as f64 - c).max(1.0);
        let work = prefill_work(q, c);
        let kv = (e.tokens + e.output) as f64 * 1.10; // template/estimate margin
        let pi = ps.iter().enumerate().filter(|(_,w)|w.is_available())
            .filter_map(|(i,w)| {
                let cap = self.config.workers.get(w.base_url())?;
                let live=s.live_kv.get(w.url()).filter(|v|now.duration_since(v.observed)<Duration::from_secs(5)).map(|v|v.usage);
                let d = s.demand.entry(w.url().into()).or_default();
                let score=if self.config.prefill_kv_guard {
                    prefill_score(d,cap,work,kv,self.config.kv_fraction,live)
                } else {(d.work + work) / cap.max_seqs + d.requests as f64 * 256.0};
                Some((i,score))
            }).min_by(|a,b|a.1.total_cmp(&b.1))?.0;
        let di = ds.iter().enumerate().filter(|(_,w)|w.is_available())
            .filter_map(|(i,w)| {
                let cap = self.config.workers.get(w.base_url())?;
                let d = s.demand.entry(w.url().into()).or_default();
                Some((i, decode_score(d, cap, kv, self.config.kv_fraction)))
            }).min_by(|a,b|a.1.total_cmp(&b.1))?.0;
        let mut reserve = |w: &Arc<dyn Worker>, stage, kv, work, pages| {
            let d = s.demand.entry(w.url().into()).or_default();
            d.requests += 1; d.kv += kv; d.work += work;
            metrics::counter!("vllm_router_pd_token_balance_selected_total", "worker" => w.url().to_string(), "stage" => stage).increment(1);
            metrics::gauge!("vllm_router_pd_assigned_requests", "worker" => w.url().to_string(), "stage" => stage).set(d.requests as f64);
            metrics::gauge!("vllm_router_pd_assigned_kv_tokens_estimate", "worker" => w.url().to_string(), "stage" => stage).set(d.kv);
            Reservation { owner: self.clone(), url: w.url().into(), kv, work, pages, tokens:e.tokens, started:now, stage }
        };
        let p = reserve(&ps[pi], "prefill", if self.config.prefill_kv_guard { kv } else { 0.0 }, work, e.pages);
        let d = reserve(&ds[di], "decode", kv, 0.0, Vec::new());
        Some((pi, di, p, d))
    }
    pub(super) fn validate_workers(&self, workers: &[Arc<dyn Worker>]) -> Result<(), String> {
        let missing: HashSet<_> = workers.iter().filter(|w| !self.config.workers.contains_key(w.base_url())).map(|w|w.base_url()).collect();
        if missing.is_empty() { Ok(()) } else { Err(format!("missing PD capacity config: {:?}", missing)) }
    }
    #[cfg(test)]
    pub(super) fn for_test_workers(workers: &[Arc<dyn Worker>]) -> Arc<Self> {
        Arc::new(Self { config: Config { tokenizer_path: String::new(), kv_fraction: 0.85,
            prefill_kv_guard: false, workers: workers.iter().map(|w| (w.base_url().to_string(),
                Capacity { kv_tokens: 2e6, max_seqs: 512.0 })).collect() },
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            tokenization_slots: Arc::new(tokio::sync::Semaphore::new(4)),
            state: Mutex::new(State { demand: HashMap::new(), live_kv: HashMap::new(),
                prefixes: LruCache::new(NonZeroUsize::new(20).unwrap()) }) })
    }
    #[cfg(test)]
    pub(super) fn assigned_for_test(&self, url: &str) -> usize {
        self.state.lock().demand.get(url).map_or(0, |d| d.requests)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn expired_kv_rows_are_reclaimed_not_just_ignored() {
        let now = Instant::now();
        let mut state = State { live_kv: HashMap::new(), demand: HashMap::new(),
            prefixes: LruCache::new(NonZeroUsize::new(20).unwrap()) };
        state.live_kv.insert("old".into(), LiveKv { usage: 0.5, observed: now - Duration::from_secs(6) });
        state.live_kv.insert("boundary".into(), LiveKv { usage: 0.5, observed: now - Duration::from_secs(5) });
        state.live_kv.insert("current".into(), LiveKv { usage: 0.5, observed: now });
        state.expire_live_kv(now);
        assert_eq!(state.live_kv.len(), 1);
        assert!(state.live_kv.contains_key("current"));
    }
    fn fixture() -> (Arc<TokenBalancer>, Vec<Arc<dyn Worker>>, Vec<Arc<dyn Worker>>) {
        use crate::core::{BasicWorker,WorkerType};
        let ps:Vec<Arc<dyn Worker>>=vec![Arc::new(BasicWorker::new("http://p:1".into(),WorkerType::Regular))];
        let ds:Vec<Arc<dyn Worker>>=vec![Arc::new(BasicWorker::new("http://d:1".into(),WorkerType::Regular))];
        let workers=HashMap::from([("http://p:1".into(),Capacity{kv_tokens:2e6,max_seqs:24.0}),
            ("http://d:1".into(),Capacity{kv_tokens:2e6,max_seqs:24.0})]);
        let b=Arc::new(TokenBalancer{config:Config{tokenizer_path:String::new(),workers,kv_fraction:0.85,prefill_kv_guard:false},
            tokenizer:Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            tokenization_slots:Arc::new(tokio::sync::Semaphore::new(4)),
            state:Mutex::new(State{demand:HashMap::new(),live_kv:HashMap::new(),prefixes:LruCache::new(NonZeroUsize::new(20).unwrap())})});
        (b,ps,ds)
    }
    #[test] fn prefill_and_decode_have_independent_lifetimes() {
        let (b,ps,ds)=fixture();
        let (_,_,p,d)=b.select(&ps,&ds,Estimate{tokens:1000,output:400,pages:vec![[1;32]]}).unwrap();
        assert_eq!(b.state.lock().demand["http://d:1"].requests,1);
        p.complete_prefill(&serde_json::json!({"usage":{"prompt_tokens":1001}}));
        assert_eq!(b.state.lock().demand["http://p:1"].requests,0);
        assert_eq!(b.state.lock().demand["http://d:1"].requests,1);
        assert_eq!(b.state.lock().prefixes.len(),1);
        drop(d);
        assert_eq!(b.state.lock().demand["http://d:1"].kv,0.0);
        assert_eq!(b.state.lock().demand["http://d:1"].requests,0);
    }
    #[tokio::test] async fn async_estimation_preserves_tokens_and_releases_slot() {
        let (b,_,_) = fixture();
        let capacity = b.tokenization_slots.available_permits();
        let request = serde_json::json!({"messages":[{"role":"user","content":"hello"}],"max_tokens":12});
        let expected = b.estimate(&request).unwrap();
        let actual = b.estimate_async(&request).await.unwrap();
        assert_eq!(actual.tokens, expected.tokens);
        assert_eq!(actual.output, expected.output);
        assert_eq!(actual.pages, expected.pages);
        assert_eq!(b.tokenization_slots.available_permits(), capacity);
    }
    #[tokio::test] async fn cancelled_slot_wait_does_not_consume_a_permit() {
        let (b,_,_) = fixture();
        let capacity = b.tokenization_slots.available_permits();
        let all = b.tokenization_slots.clone().acquire_many_owned(capacity as u32).await.unwrap();
        let owner = b.clone();
        let task = tokio::spawn(async move { owner.estimate_async(&serde_json::json!({})).await });
        tokio::task::yield_now().await;
        assert!(!task.is_finished());
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        drop(all);
        assert_eq!(b.tokenization_slots.available_permits(), capacity);
    }
    #[test] fn tokenization_configuration_is_bounded() {
        assert_eq!(tokenization_concurrency(16, None).unwrap(), 12);
        assert_eq!(tokenization_concurrency(1, None).unwrap(), 1);
        assert_eq!(tokenization_concurrency(4, None).unwrap(), 4);
        assert_eq!(tokenization_concurrency(128, None).unwrap(), MAX_TOKENIZATION_CONCURRENCY);
        assert_eq!(tokenization_concurrency(16, Some("16")).unwrap(), 16);
        for invalid in ["0", "17", "-1", "abc", "", "99999999999999999999999999999999"] {
            assert!(tokenization_concurrency(16, Some(invalid)).is_err(), "{invalid}");
        }
        assert_eq!(bpe_cache_capacity(None).unwrap(), 0);
        assert_eq!(bpe_cache_capacity(Some("1024")).unwrap(), 1024);
        assert!(bpe_cache_capacity(Some("16385")).is_err());
        assert!(bpe_cache_capacity(Some("-1")).is_err());
    }
    #[test]
    #[ignore = "CPU-only workload benchmark; requires explicit tokenizer and SWE request corpus"]
    fn tokenization_workload_abba() {
        let path = std::env::var("ROUTER_BENCH_TOKENIZER").unwrap();
        let corpus = std::env::var("ROUTER_BENCH_REQUESTS").unwrap();
        let requests: Vec<Value> = serde_json::from_slice(&std::fs::read(corpus).unwrap()).unwrap();
        assert!((12..=128).contains(&requests.len()));
        let make = |slots, cache| {
            let (mut b, _, _) = fixture();
            let inner = Arc::get_mut(&mut b).unwrap();
            inner.tokenizer = Tokenizer::from_file(&path).unwrap();
            set_bpe_cache_capacity(&mut inner.tokenizer, cache);
            inner.tokenization_slots = Arc::new(tokio::sync::Semaphore::new(slots));
            b
        };
        let reference = make(4, 0);
        let expected: Vec<_> = requests.iter().map(|r| {
            let e = reference.estimate(r).unwrap(); (e.tokens, e.pages, e.output)
        }).collect();
        let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(2)
            .max_blocking_threads(16).enable_all().build().unwrap();
        // ABC-CBA repeats each admission setting with unchanged requests and CPU affinity.
        for (slots, cache) in [(4,0),(12,0),(12,4096),(12,4096),(12,0),(4,0)] {
            let b = make(slots, cache);
            for request in requests.iter().take(2) { b.estimate(request).unwrap(); }
            let start = Instant::now();
            let mut latencies = runtime.block_on(async {
                let mut pending = tokio::task::JoinSet::new();
                for (i, request) in requests.iter().cloned().enumerate() {
                    let owner = b.clone();
                    pending.spawn(async move {
                        let t = Instant::now();
                        let e = owner.estimate_async(&request).await.unwrap();
                        (i, e, t.elapsed().as_secs_f64())
                    });
                }
                let mut times = Vec::new();
                while let Some(row) = pending.join_next().await {
                    let (i,e,t) = row.unwrap();
                    assert_eq!((e.tokens,e.pages,e.output),expected[i],"token/page identity drift");
                    times.push(t);
                }
                times
            });
            let seconds = start.elapsed().as_secs_f64();
            latencies.sort_by(f64::total_cmp);
            println!("ROUTER_CPU_BENCH {}", serde_json::json!({
                "slots":slots,"bpe_cache_capacity":cache,"requests":requests.len(),
                "seconds":seconds,"requests_per_second":requests.len() as f64/seconds,
                "batch_latency_p50_s":latencies[latencies.len()/2],
                "batch_latency_max_s":latencies.last(),
                "input_token_mean":expected.iter().map(|e|e.0).sum::<usize>() as f64/expected.len() as f64,
                "exact_tokens_and_pages":true,"inference_requests_sent":0
            }));
        }
    }
    #[test] fn bpe_cache_keeps_exact_ids_and_page_hashes() {
        // A real merge and many repeated words exercise both cache paths.
        let mut vocab = serde_json::Map::new();
        for byte in 32u8..=126 { vocab.insert((byte as char).to_string(), serde_json::json!(byte)); }
        vocab.insert("ab".into(), serde_json::json!(127));
        let model_json = serde_json::json!({
            "type":"BPE", "vocab":vocab, "merges":[["a","b"]]
        }).to_string();
        let model: tokenizers::models::bpe::BPE = serde_json::from_str(&model_json).unwrap();
        let make_balancer = |capacity| {
            let (mut owner, _, _) = fixture();
            let b = Arc::get_mut(&mut owner).unwrap();
            b.tokenizer = Tokenizer::new(model.clone());
            b.tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace::default()));
            set_bpe_cache_capacity(&mut b.tokenizer, capacity);
            owner
        };
        let without = make_balancer(0);
        let with = make_balancer(1024);
        let request = serde_json::json!({"model":"test","prompt":"ab ab abc ab cd ".repeat(512),
            "cache_salt":"salt","max_completion_tokens":17});
        let content = serde_json::to_string(&serde_json::json!({
            "tools": request.get("tools"), "messages": request.get("messages"),
            "prompt": request.get("prompt"), "cache_salt": request.get("cache_salt"),
            "model": request.get("model"), "chat_template_kwargs": request.get("chat_template_kwargs")
        })).unwrap();
        let expected_ids = without.tokenizer.encode(content.clone(), false).unwrap().get_ids().to_vec();
        assert!(expected_ids.len() >= 256);
        assert!(expected_ids.contains(&127));
        for _ in 0..2 {
            assert_eq!(with.tokenizer.encode(content.clone(), false).unwrap().get_ids(), expected_ids);
            let expected = without.estimate(&request).unwrap();
            let actual = with.estimate(&request).unwrap();
            assert!(!expected.pages.is_empty());
            assert_eq!(actual.tokens, expected.tokens);
            assert_eq!(actual.pages, expected.pages);
            assert_eq!(actual.output, 17);
        }
    }
    #[tokio::test] async fn cancelled_blocking_work_keeps_its_slot_until_finished() {
        let slots = Arc::new(tokio::sync::Semaphore::new(1));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let work_slots = slots.clone();
        let task = tokio::spawn(async move {
            run_tokenization(work_slots, || move || {
                let _ = started_tx.send(());
                let _ = release_rx.recv();
                Ok(())
            }).await
        });
        tokio::time::timeout(Duration::from_secs(5), started_rx).await.unwrap().unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(slots.available_permits(), 0);
        assert!(slots.clone().try_acquire_owned().is_err());
        release_tx.send(()).unwrap();
        let permit = tokio::time::timeout(Duration::from_secs(5), slots.clone().acquire_owned()).await.unwrap().unwrap();
        drop(permit);
        assert_eq!(slots.available_permits(), 1);
    }
    #[test] fn cancellation_does_not_publish_cache_and_releases_all_demand() {
        let (b,ps,ds)=fixture();
        let (_,_,p,d)=b.select(&ps,&ds,Estimate{tokens:1000,output:400,pages:vec![[2;32]]}).unwrap();
        drop(p);drop(d);
        assert_eq!(b.state.lock().prefixes.len(),0);
        for value in b.state.lock().demand.values() {assert_eq!(value.requests,0);assert_eq!(value.kv,0.0);assert_eq!(value.work,0.0);}
    }
    #[test] fn unavailable_workers_are_not_selected() {
        let (b,ps,ds)=fixture(); ds[0].set_healthy(false);
        assert!(b.select(&ps,&ds,Estimate{tokens:1000,output:400,pages:vec![]}).is_none());
        assert_eq!(b.state.lock().demand["http://p:1"].requests,0);
    }
    #[test] fn prefill_kv_guard_avoids_pressure_with_equal_compute() {
        let cap=Capacity{kv_tokens:2_000_000.0,max_seqs:24.0};
        let d=Demand{requests:3,kv:400_000.0,work:4000.0};
        assert!(prefill_score(&d,&cap,3000.0,130_000.0,0.85,Some(0.96))>
                prefill_score(&d,&cap,3000.0,130_000.0,0.85,Some(0.40)));
        let full=Demand{requests:3,kv:1_700_000.0,work:4000.0};
        assert!(prefill_score(&full,&cap,3000.0,130_000.0,0.85,None)>
                prefill_score(&d,&cap,3000.0,130_000.0,0.85,None));
    }
    #[test] fn live_kv_parser_keeps_ranks_and_rejects_invalid_data() {
        let raw="vllm:kv_cache_usage_perc{engine=\"0\",model_name=\"glm\"} 0.95\nvllm:kv_cache_usage_perc{engine=\"1\"} NaN\nvllm:kv_cache_usage_perc{engine=\"2\"} 1.9\n";
        let rows=live_kv_rows("http://p:1",raw,Instant::now());
        assert_eq!(rows.len(),1);assert_eq!(rows[0].0,"http://p:1@0");assert_eq!(rows[0].1.usage,0.95);
    }
    #[test] fn same_requests_long_kv_avoids_small_ep4() {
        let d=Demand{requests:8,kv:1_200_000.0,work:0.0};
        let small=Capacity{kv_tokens:1_944_075.0,max_seqs:24.0};
        let large=Capacity{kv_tokens:4_821_918.0,max_seqs:24.0};
        assert!(decode_score(&d,&small,150_000.0,0.85)>decode_score(&d,&large,150_000.0,0.85));
    }
    #[test] fn short_count_long_length_tradeoff() {
        let cap=Capacity{kv_tokens:4_000_000.0,max_seqs:24.0};
        let a=Demand{requests:8,kv:2_500_000.0,work:0.0};
        let b=Demand{requests:12,kv:800_000.0,work:0.0};
        assert!(decode_score(&a,&cap,4000.0,0.85)<decode_score(&b,&cap,4000.0,0.85));
        assert!(decode_score(&a,&cap,300_000.0,0.85)>decode_score(&b,&cap,300_000.0,0.85));
    }
    #[test] fn capacity_penalty_and_prefill_prefix_cost() {
        let cap=Capacity{kv_tokens:2_000_000.0,max_seqs:24.0};
        let d=Demand{requests:8,kv:1_700_000.0,work:0.0};
        assert!(decode_score(&d,&cap,200_000.0,0.85)>10.0);
        assert!(prefill_work(2000.0,130_000.0)>prefill_work(2000.0,20_000.0));
        assert!(prefill_work(30_000.0,0.0)>prefill_work(2000.0,130_000.0));
    }
}
