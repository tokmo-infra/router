//! Round-robin load balancing policy

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::{Worker, WorkerType};
use crate::metrics::RouterMetrics;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Round-robin selection policy
///
/// Selects workers in sequential order, cycling through all healthy workers.
#[derive(Debug)]
pub struct RoundRobinPolicy {
    counter: AtomicUsize,
    decode_load_slack: Option<usize>,
}

impl RoundRobinPolicy {
    pub fn new() -> Self {
        // Opt-in for direct PD decode pools. The PD admission lock reserves
        // worker.load() before prefill and releases it at response EOF/drop.
        let decode_load_slack = match std::env::var("VLLM_ROUTER_PD_DECODE_RR_LOAD_SLACK") {
            Ok(value) => Some(
                value
                    .parse::<usize>()
                    .expect("VLLM_ROUTER_PD_DECODE_RR_LOAD_SLACK must be a non-negative integer"),
            ),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => panic!("VLLM_ROUTER_PD_DECODE_RR_LOAD_SLACK: {error}"),
        };
        Self {
            counter: AtomicUsize::new(0),
            decode_load_slack,
        }
    }
}

impl Default for RoundRobinPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancingPolicy for RoundRobinPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Get and increment counter atomically
        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        let mut selected_idx = count % healthy_indices.len();
        if let Some(slack) = self.decode_load_slack.filter(|_| {
            healthy_indices
                .iter()
                .all(|&idx| matches!(workers[idx].worker_type(), WorkerType::Decode))
        }) {
            let loads: Vec<_> = healthy_indices
                .iter()
                .map(|&idx| workers[idx].load())
                .collect();
            let ceiling = loads.iter().min().unwrap().saturating_add(slack);
            // Retain RR among ranks within the count band; never turn an
            // unavailable backend metric into a zero-load admission signal.
            let skipped = (0..loads.len())
                .find(|&offset| loads[(selected_idx + offset) % loads.len()] <= ceiling)
                .unwrap(); // The minimum-load rank is always eligible.
            if skipped > 0 {
                metrics::counter!("vllm_router_pd_decode_rr_rebalanced_total",
                    "worker" => workers[healthy_indices[selected_idx]].url().to_owned())
                .increment(1);
                self.counter.fetch_add(skipped, Ordering::Relaxed);
                selected_idx = (selected_idx + skipped) % healthy_indices.len();
            }
        }
        let worker = workers[healthy_indices[selected_idx]].url();

        RouterMetrics::record_processed_request(worker);
        RouterMetrics::record_policy_decision(self.name(), worker);
        Some(healthy_indices[selected_idx])
    }

    fn name(&self) -> &'static str {
        "round_robin"
    }

    fn reset(&self) {
        self.counter.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    fn guarded(slack: Option<usize>) -> RoundRobinPolicy {
        RoundRobinPolicy {
            counter: AtomicUsize::new(0),
            decode_load_slack: slack,
        }
    }

    fn pool(kind: WorkerType, loads: &[usize]) -> Vec<Arc<dyn Worker>> {
        loads
            .iter()
            .enumerate()
            .map(|(rank, &load)| {
                let worker = BasicWorker::new(format!("http://decode:8000@{rank}"), kind.clone());
                for _ in 0..load {
                    worker.increment_load();
                }
                Arc::new(worker) as Arc<dyn Worker>
            })
            .collect()
    }

    #[test]
    fn decode_guard_skips_hot_rank_rotates_peers_and_recovers() {
        let policy = guarded(Some(32));
        let workers = pool(WorkerType::Decode, &[100, 0, 0]);
        let selected: Vec<_> = (0..12)
            .map(|_| policy.select_worker(&workers, None).unwrap())
            .collect();
        assert_eq!(selected, vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]);
        for _ in 0..100 {
            workers[0].decrement_load();
        }
        assert_eq!(policy.select_worker(&workers, None), Some(0));
    }

    #[test]
    fn decode_guard_retains_rr_within_slack_and_when_disabled() {
        for (slack, loads) in [(Some(32), vec![5, 30, 16]), (None, vec![100, 0, 0])] {
            let policy = guarded(slack);
            let workers = pool(WorkerType::Decode, &loads);
            let selected: Vec<_> = (0..6)
                .map(|_| policy.select_worker(&workers, None).unwrap())
                .collect();
            assert_eq!(selected, vec![0, 1, 2, 0, 1, 2]);
        }
    }

    #[test]
    fn decode_guard_does_not_change_prefill_or_regular_rr() {
        for kind in [
            WorkerType::Regular,
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        ] {
            let policy = guarded(Some(32));
            let workers = pool(kind, &[100, 0, 0]);
            assert_eq!(policy.select_worker(&workers, None), Some(0));
            assert_eq!(policy.select_worker(&workers, None), Some(1));
            assert_eq!(policy.select_worker(&workers, None), Some(2));
        }
    }

    #[test]
    fn decode_guard_uses_healthy_minimum_and_inclusive_threshold() {
        let policy = guarded(Some(32));
        let workers = pool(WorkerType::Decode, &[0, 100, 120]);
        workers[0].set_healthy(false);
        assert_eq!(policy.select_worker(&workers, None), Some(1));
        assert_eq!(policy.select_worker(&workers, None), Some(2));
        let workers = pool(WorkerType::Decode, &[42, 10, 11]);
        policy.reset();
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        workers[0].increment_load();
        policy.reset();
        assert_eq!(policy.select_worker(&workers, None), Some(1));
    }

    #[test]
    fn decode_guard_bounds_skew_with_serialized_concurrent_reservations() {
        let policy = Arc::new(guarded(Some(4)));
        let workers = pool(WorkerType::Decode, &[100, 0, 0, 0]);
        let admission = Arc::new(std::sync::Mutex::new(()));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let policy = policy.clone();
                let workers = workers.clone();
                let admission = admission.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        let _guard = admission.lock().unwrap();
                        let selected = policy.select_worker(&workers, None).unwrap();
                        workers[selected].increment_load();
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        let loads: Vec<_> = workers.iter().map(|w| w.load()).collect();
        assert_eq!(loads.iter().sum::<usize>(), 500);
        assert!(loads.iter().max().unwrap() - loads.iter().min().unwrap() <= 5);
    }

    #[test]
    fn test_round_robin_selection() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w3:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // Should select workers in order: 0, 1, 2, 0, 1, 2, ...
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(1));
        assert_eq!(policy.select_worker(&workers, None), Some(2));
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(1));
    }

    #[test]
    fn test_round_robin_with_unhealthy_workers() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w3:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // Mark middle worker as unhealthy
        workers[1].set_healthy(false);

        // Should skip unhealthy worker: 0, 2, 0, 2, ...
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(2));
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(2));
    }

    #[test]
    fn test_round_robin_reset() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // Advance the counter
        assert_eq!(policy.select_worker(&workers, None), Some(0));
        assert_eq!(policy.select_worker(&workers, None), Some(1));

        // Reset should start from beginning
        policy.reset();
        assert_eq!(policy.select_worker(&workers, None), Some(0));
    }
}
