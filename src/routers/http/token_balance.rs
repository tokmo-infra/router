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
}
fn default_headroom() -> f64 { 0.85 }
#[derive(Debug, Default)]
struct Demand { requests: usize, kv: f64, work: f64 }
#[derive(Debug)]
struct State {
    demand: HashMap<String, Demand>,
    // Completed prefill page-prefix observations, with bounded memory and TTL.
    // Shared across P workers because their Mooncake Store is shared.
    prefixes: LruCache<[u8; 32], Instant>,
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
        let mut tokenizer = Tokenizer::from_file(&config.tokenizer_path).map_err(|e| e.to_string())?;
        // BPE caches are thread-local: unbounded blocking-thread fanout multiplies
        // long-prompt strings. Routing does not need a second full BPE word cache.
        if let tokenizers::models::ModelWrapper::BPE(mut model) = tokenizer.get_model().clone() {
            model.resize_cache(0);
            tokenizer.with_model(model);
        }
        tracing::info!(workers=config.workers.len(), kv_fraction=config.kv_fraction, "PD token balance enabled (heuristic cost; estimated token demand)");
        Ok(Some(Arc::new(Self { config, tokenizer, tokenization_slots: Arc::new(tokio::sync::Semaphore::new(4)), state: Mutex::new(State {
            demand: HashMap::new(), prefixes: LruCache::new(NonZeroUsize::new(262144).unwrap()),
        }) })))
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
        let matched = e.pages.iter().take_while(|h| s.prefixes.peek(*h).is_some_and(|t| now.duration_since(*t) < Duration::from_secs(60))).count();
        let c = (matched.saturating_sub(1) * 256).min(e.tokens.saturating_sub(1)) as f64;
        let q = (e.tokens as f64 - c).max(1.0);
        let work = prefill_work(q, c);
        let kv = (e.tokens + e.output) as f64 * 1.10; // template/estimate margin
        let pi = ps.iter().enumerate().filter(|(_,w)|w.is_available())
            .filter_map(|(i,w)| {
                let cap = self.config.workers.get(w.base_url())?;
                let d = s.demand.entry(w.url().into()).or_default();
                Some((i, (d.work + work) / cap.max_seqs + d.requests as f64 * 256.0))
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
        let p = reserve(&ps[pi], "prefill", 0.0, work, e.pages);
        let d = reserve(&ds[di], "decode", kv, 0.0, Vec::new());
        Some((pi, di, p, d))
    }
    pub(super) fn validate_workers(&self, workers: &[Arc<dyn Worker>]) -> Result<(), String> {
        let missing: HashSet<_> = workers.iter().filter(|w| !self.config.workers.contains_key(w.base_url())).map(|w|w.base_url()).collect();
        if missing.is_empty() { Ok(()) } else { Err(format!("missing PD capacity config: {:?}", missing)) }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Arc<TokenBalancer>, Vec<Arc<dyn Worker>>, Vec<Arc<dyn Worker>>) {
        use crate::core::{BasicWorker,WorkerType};
        let ps:Vec<Arc<dyn Worker>>=vec![Arc::new(BasicWorker::new("http://p:1".into(),WorkerType::Regular))];
        let ds:Vec<Arc<dyn Worker>>=vec![Arc::new(BasicWorker::new("http://d:1".into(),WorkerType::Regular))];
        let workers=HashMap::from([("http://p:1".into(),Capacity{kv_tokens:2e6,max_seqs:24.0}),
            ("http://d:1".into(),Capacity{kv_tokens:2e6,max_seqs:24.0})]);
        let b=Arc::new(TokenBalancer{config:Config{tokenizer_path:String::new(),workers,kv_fraction:0.85},
            tokenizer:Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            tokenization_slots:Arc::new(tokio::sync::Semaphore::new(4)),
            state:Mutex::new(State{demand:HashMap::new(),prefixes:LruCache::new(NonZeroUsize::new(20).unwrap())})});
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
