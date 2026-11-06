//! Opt-in regular/coupled routing using live per-DP GPU KV and request load.
//! This does not change benchmark scheduling or claim knowledge of Store hits.
use crate::core::Worker;
use parking_lot::Mutex;
use std::{collections::{HashMap, HashSet}, sync::Arc, time::{Duration, Instant}};

#[derive(Debug, Clone, Copy, Default)]
struct Native { usage: Option<f64>, running: f64, waiting: f64 }
#[derive(Debug, Default)]
pub(super) struct CoupledKv {
    rows: Arc<Mutex<HashMap<String, (Native, Instant)>>>,
    watching: Mutex<HashSet<String>>,
}
fn parse(base: &str, body: &str) -> HashMap<String, Native> {
    let mut rows = HashMap::<String, Native>::new();
    for line in body.lines().filter(|l| l.starts_with("vllm:")) {
        let Some(metric) = line.split('{').next() else { continue };
        if !["vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting"].contains(&metric) { continue; }
        let Some(rank) = line.split("engine=\"").nth(1).and_then(|s| s.split('\"').next()).and_then(|s| s.parse::<usize>().ok()) else { continue };
        let Some(v) = line.split_whitespace().nth(1).and_then(|s| s.parse::<f64>().ok()).filter(|v| v.is_finite() && *v >= 0.0) else { continue };
        let row = rows.entry(format!("{base}@{rank}")).or_default();
        match metric {
            "vllm:kv_cache_usage_perc" if v <= 1.0 => row.usage = Some(v),
            "vllm:num_requests_running" => row.running = v,
            "vllm:num_requests_waiting" => row.waiting = v,
            _ => {}
        }
    }
    rows
}
fn score(inflight: usize, live: Option<Native>) -> f64 {
    let live = live.unwrap_or_default();
    let load = (inflight as f64).max(live.running + live.waiting);
    // Idle cached pages are evictable: do not treat idle usage as active KV.
    let usage = if load > 0.0 { live.usage.unwrap_or(0.0) } else { 0.0 };
    load + 2.0 * live.waiting + 8.0 * usage.powi(2)
        + 20.0 * ((usage - 0.85) / 0.15).max(0.0).powi(2)
}
impl CoupledKv {
    pub(super) fn from_env() -> Option<Self> {
        (std::env::var("VLLM_ROUTER_COUPLED_KV_BALANCE").as_deref() == Ok("1")).then(Self::default)
    }
    pub(super) fn select(&self, workers: &[Arc<dyn Worker>]) -> Option<usize> {
        for w in workers {
            let base = w.base_url().to_string();
            if !self.watching.lock().insert(base.clone()) { continue; }
            let rows = self.rows.clone();
            tokio::spawn(async move {
                let client = match reqwest::Client::builder().no_proxy().timeout(Duration::from_secs(2)).build() { Ok(v) => v, Err(_) => return };
                loop {
                    if let Ok(response) = client.get(format!("{base}/metrics")).send().await {
                        if response.status().is_success() {
                            if let Ok(body) = response.text().await {
                                let now = Instant::now();
                                for (url, row) in parse(&base, &body) {
                                    if let Some(usage) = row.usage { metrics::gauge!("vllm_router_coupled_live_kv_usage", "worker" => url.clone()).set(usage); }
                                    metrics::gauge!("vllm_router_coupled_native_running", "worker" => url.clone()).set(row.running);
                                    metrics::gauge!("vllm_router_coupled_native_waiting", "worker" => url.clone()).set(row.waiting);
                                    rows.lock().insert(url, (row, now));
                                }
                            }
                        }
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            });
        }
        let rows = self.rows.lock();
        let selected = workers.iter().enumerate().filter(|(_, w)| w.is_available())
            .map(|(i, w)| {
                let live = rows.get(w.url()).filter(|(_, t)| t.elapsed() < Duration::from_secs(5)).map(|(r, _)| *r);
                let s = score(w.load(), live);
                metrics::gauge!("vllm_router_coupled_routing_score", "worker" => w.url().to_string()).set(s);
                (i, s, w.processed_requests())
            }).min_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.2.cmp(&b.2)))?.0;
        metrics::counter!("vllm_router_coupled_selected_total", "worker" => workers[selected].url().to_string()).increment(1);
        Some(selected)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn kv_pressure_beats_same_request_count() {
        assert!(score(4, Some(Native { usage: Some(0.96), running: 4.0, waiting: 0.0 }))
            > score(4, Some(Native { usage: Some(0.3), running: 4.0, waiting: 0.0 })));
    }
    #[test] fn waiting_is_not_mistaken_for_free_capacity() {
        assert!(score(5, Some(Native { usage: Some(0.2), running: 1.0, waiting: 4.0 })) > score(4, None));
    }
    #[test] fn idle_reusable_cache_is_not_active_pressure() {
        assert_eq!(score(0, Some(Native { usage: Some(0.99), ..Native::default() })), 0.0);
    }
    #[test] fn parser_preserves_rank_and_rejects_invalid_usage() {
        let r = parse("http://p:1", "vllm:kv_cache_usage_perc{engine=\"0\"} 0.98\nvllm:num_requests_waiting{engine=\"0\"} 3\nvllm:kv_cache_usage_perc{engine=\"1\"} NaN\nvllm:kv_cache_usage_perc{engine=\"2\"} 1.2\n");
        assert_eq!(r["http://p:1@0"].usage, Some(0.98));
        assert_eq!(r["http://p:1@0"].waiting, 3.0);
        assert!(!r.contains_key("http://p:1@1"));
        assert!(r["http://p:1@2"].usage.is_none());
    }
}
