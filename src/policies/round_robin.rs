//! Round-robin load balancing policy

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Round-robin selection policy
///
/// Selects workers in sequential order, cycling through all healthy workers.
/// DP pools use rank-major, then host-major order so host-by-host registration
/// does not concentrate the first requests on a single host. Non-DP pools keep
/// their input order.
#[derive(Debug, Default)]
pub struct RoundRobinPolicy {
    counter: AtomicUsize,
}

impl RoundRobinPolicy {
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
        }
    }
}

impl LoadBalancingPolicy for RoundRobinPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let mut healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        if workers.iter().any(|worker| worker.dp_rank().is_some()) {
            // Use worker identity rather than Vec position: reordering the same
            // pool must not change its round-robin sequence. Missing ranks sort
            // before DP ranks in a mixed pool; wholly non-DP pools are unsorted.
            healthy_indices.sort_by_key(|&idx| {
                let worker = &workers[idx];
                (worker.dp_rank(), worker.base_url(), worker.url())
            });
        }

        // Get and increment counter atomically
        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        let selected_idx = count % healthy_indices.len();
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
    use crate::core::{BasicWorker, DPAwareWorker, WorkerType};
    use std::collections::{HashMap, HashSet};

    fn dp_workers(ranks_per_host: &[usize]) -> Vec<Arc<dyn Worker>> {
        ranks_per_host
            .iter()
            .enumerate()
            .flat_map(|(host, &dp_size)| {
                (0..dp_size).map(move |rank| {
                    Arc::new(DPAwareWorker::new(
                        format!("http://host{host}:8000"),
                        rank,
                        dp_size,
                        WorkerType::Regular,
                    )) as Arc<dyn Worker>
                })
            })
            .collect()
    }

    fn selected_urls(
        policy: &RoundRobinPolicy,
        workers: &[Arc<dyn Worker>],
        count: usize,
    ) -> Vec<String> {
        (0..count)
            .map(|_| {
                let idx = policy.select_worker(workers, None).unwrap();
                assert!(workers[idx].is_healthy());
                workers[idx].url().to_string()
            })
            .collect()
    }

    fn assert_dp_batch_fairness(concurrency: usize) {
        let policy = RoundRobinPolicy::new();
        // Reproduce registration order: all eight ranks of each host together.
        let workers = dp_workers(&[8, 8, 8, 8]);
        let mut host_counts = HashMap::new();
        let mut rank_counts = HashMap::new();
        let mut seen = HashSet::new();

        for request in 0..concurrency {
            let idx = policy.select_worker(&workers, None).unwrap();
            let worker = &workers[idx];
            assert_eq!(
                worker.url(),
                format!("http://host{}:8000@{}", request % 4, request / 4)
            );
            *host_counts.entry(worker.base_url()).or_insert(0) += 1;
            *rank_counts.entry(worker.dp_rank().unwrap()).or_insert(0) += 1;
            assert!(seen.insert(worker.url()));
        }

        assert_eq!(host_counts.len(), 4);
        assert!(host_counts.values().all(|&count| count == concurrency / 4));
        assert_eq!(rank_counts.len(), concurrency / 4);
        assert!(rank_counts.values().all(|&count| count == 4));
    }

    #[test]
    fn test_round_robin_dp_c8() {
        assert_dp_batch_fairness(8);
    }

    #[test]
    fn test_round_robin_dp_c16() {
        assert_dp_batch_fairness(16);
    }

    #[test]
    fn test_round_robin_dp_c32() {
        assert_dp_batch_fairness(32);
    }

    #[test]
    fn test_round_robin_dp_reordering_preserves_sequence() {
        let policy = RoundRobinPolicy::new();
        let mut workers = dp_workers(&[8, 8, 8, 8]);

        // Change Vec positions on every request, including across wraparound.
        for request in 0..96 {
            workers.rotate_left((request * 7 + 3) % 32);
            if request % 2 == 0 {
                workers.reverse();
            }
            let idx = policy.select_worker(&workers, None).unwrap();
            let position = request % 32;
            assert_eq!(
                workers[idx].url(),
                format!("http://host{}:8000@{}", position % 4, position / 4)
            );
        }

        policy.reset();
        assert_eq!(selected_urls(&policy, &workers, 1), ["http://host0:8000@0"]);
    }

    #[test]
    fn test_round_robin_dp_health_filtering() {
        let policy = RoundRobinPolicy::new();
        let mut workers = dp_workers(&[8, 8, 8, 8]);
        // Remove a whole host plus individual ranks on other hosts.
        for worker in &workers {
            if worker.base_url() == "http://host1:8000"
                || (worker.base_url() == "http://host0:8000" && worker.dp_rank() == Some(0))
                || (worker.base_url() == "http://host2:8000" && worker.dp_rank() == Some(1))
            {
                worker.set_healthy(false);
            }
        }
        let expected: Vec<String> = (0..8)
            .flat_map(|rank| {
                (0..4)
                    .filter(move |&host| {
                        host != 1 && !(host == 0 && rank == 0) && !(host == 2 && rank == 1)
                    })
                    .map(move |host| format!("http://host{host}:8000@{rank}"))
            })
            .collect();
        workers.reverse();
        for _ in 0..2 {
            assert_eq!(selected_urls(&policy, &workers, expected.len()), expected);
        }

        for worker in &workers {
            worker.set_healthy(false);
        }
        policy.reset();
        assert_eq!(policy.select_worker(&workers, None), None);
        for worker in &workers {
            worker.set_healthy(true);
        }
        // An empty healthy pool must not consume the counter.
        assert_eq!(selected_urls(&policy, &workers, 1), ["http://host0:8000@0"]);
    }

    #[test]
    fn test_round_robin_dp_unequal_rank_counts() {
        let policy = RoundRobinPolicy::new();
        // Include ranks >= 10 to distinguish numeric rank from URL ordering.
        let ranks_per_host = [2, 12, 4, 1];
        let mut workers = dp_workers(&ranks_per_host);
        workers.reverse();
        let expected: Vec<String> = (0..12)
            .flat_map(|rank| {
                ranks_per_host
                    .iter()
                    .enumerate()
                    .filter(move |(_, &dp_size)| rank < dp_size)
                    .map(move |(host, _)| format!("http://host{host}:8000@{rank}"))
            })
            .collect();
        assert_eq!(expected.len(), workers.len());
        for _ in 0..2 {
            let selected = selected_urls(&policy, &workers, workers.len());
            assert_eq!(selected, expected);
            assert_eq!(selected.iter().collect::<HashSet<_>>().len(), workers.len());
        }
    }

    #[test]
    fn test_round_robin_non_dp_keeps_unsorted_input_order() {
        let policy = RoundRobinPolicy::new();
        let mut workers: Vec<Arc<dyn Worker>> = ["http://z:8000", "http://a:8000", "http://m:8000"]
            .into_iter()
            .map(|url| Arc::new(BasicWorker::new(url.to_string(), WorkerType::Regular)) as _)
            .collect();
        assert_eq!(
            selected_urls(&policy, &workers, 3),
            ["http://z:8000", "http://a:8000", "http://m:8000"]
        );
        workers.reverse();
        assert_eq!(
            selected_urls(&policy, &workers, 3),
            ["http://m:8000", "http://a:8000", "http://z:8000"]
        );
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
