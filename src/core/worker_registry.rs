//! Worker Registry for multi-router support
//!
//! Provides centralized registry for workers with model-based indexing

use crate::core::{ConnectionMode, Worker, WorkerType};
use dashmap::{mapref::entry::Entry, DashMap};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex, RwLock,
};
use uuid::Uuid;

/// Unique identifier for a worker
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct WorkerId(String);

impl WorkerId {
    /// Create a new worker ID
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create a worker ID from a string
    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    /// Get the ID as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for the model index to reduce complexity
type ModelIndex = Arc<DashMap<String, Arc<RwLock<Vec<Arc<dyn Worker>>>>>>;

/// Worker registry with model-based indexing
#[derive(Debug)]
pub struct WorkerRegistry {
    /// Serialize multi-index mutations; readers never take this lock. Mutation
    /// helpers must not call public mutators, which would re-lock it.
    mutation: Mutex<()>,

    /// All workers indexed by ID
    workers: Arc<DashMap<WorkerId, Arc<dyn Worker>>>,

    /// Workers indexed by model ID (stores WorkerId for reference)
    model_workers: Arc<DashMap<String, Vec<WorkerId>>>,

    /// Optimized model index for O(1) lookups (stores Arc<dyn Worker> directly)
    model_index: ModelIndex,

    /// Workers indexed by worker type
    type_workers: Arc<DashMap<WorkerType, Vec<WorkerId>>>,

    /// Workers indexed by connection mode
    connection_workers: Arc<DashMap<ConnectionMode, Vec<WorkerId>>>,

    /// URL to worker ID mapping (for backward compatibility)
    url_to_id: Arc<DashMap<String, WorkerId>>,

    /// Monotonic topology and availability revision for read-side caches.
    revision: Arc<AtomicU64>,
}

impl WorkerRegistry {
    /// Create a new worker registry
    pub fn new() -> Self {
        Self {
            mutation: Mutex::new(()),
            workers: Arc::new(DashMap::new()),
            model_workers: Arc::new(DashMap::new()),
            model_index: Arc::new(DashMap::new()),
            type_workers: Arc::new(DashMap::new()),
            connection_workers: Arc::new(DashMap::new()),
            url_to_id: Arc::new(DashMap::new()),
            revision: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Register or replace a worker, preserving the ID for an existing URL.
    pub fn register(&self, worker: Arc<dyn Worker>) -> WorkerId {
        let _mutation = self
            .mutation
            .lock()
            .expect("worker registry mutation poisoned");
        self.register_unlocked(worker)
    }

    fn register_unlocked(&self, worker: Arc<dyn Worker>) -> WorkerId {
        // Drop the URL shard guard before touching workers or any other index.
        let existing_id = self.url_to_id.get(worker.url()).map(|id| id.clone());
        let worker_id = existing_id.unwrap_or_default();
        // A replacement may change model, role or connection mode. Remove the
        // old worker's memberships (and Arc) before adding the new ones once.
        self.remove_unlocked(&worker_id);

        // Store worker
        self.workers.insert(worker_id.clone(), worker.clone());

        // Update URL mapping
        self.url_to_id
            .insert(worker.url().to_string(), worker_id.clone());

        // Update model index (both ID-based and optimized)
        let model_id = worker.model_id().to_string();
        self.model_workers
            .entry(model_id.clone())
            .or_default()
            .push(worker_id.clone());

        // Update optimized model index for O(1) lookups
        self.model_index
            .entry(model_id)
            .or_insert_with(|| Arc::new(RwLock::new(Vec::new())))
            .write()
            .expect("RwLock for model_index is poisoned")
            .push(worker.clone());

        // Update type index
        self.type_workers
            .entry(worker.worker_type())
            .or_default()
            .push(worker_id.clone());

        // Update connection mode index
        self.connection_workers
            .entry(worker.connection_mode())
            .or_default()
            .push(worker_id.clone());

        self.notify_worker_state_change();

        worker_id
    }

    /// Remove a worker by ID
    pub fn remove(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        let _mutation = self
            .mutation
            .lock()
            .expect("worker registry mutation poisoned");
        let removed = self.remove_unlocked(worker_id);
        if removed.is_some() {
            self.notify_worker_state_change();
        }
        removed
    }

    fn remove_unlocked(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        let (_, worker) = self.workers.remove(worker_id)?;
        self.url_to_id.remove(worker.url());

        // Entry guards allow retaining and deleting an empty bucket under the
        // same shard lock, without recursively calling remove on that map.
        if let Entry::Occupied(mut entry) = self.model_workers.entry(worker.model_id().to_string())
        {
            entry.get_mut().retain(|id| id != worker_id);
            if entry.get().is_empty() {
                entry.remove();
            }
        }
        if let Entry::Occupied(entry) = self.model_index.entry(worker.model_id().to_string()) {
            let empty = {
                let mut workers = entry
                    .get()
                    .write()
                    .expect("RwLock for model_index is poisoned");
                workers.retain(|w| w.url() != worker.url());
                workers.is_empty()
            };
            if empty {
                entry.remove();
            }
        }
        if let Entry::Occupied(mut entry) = self.type_workers.entry(worker.worker_type()) {
            entry.get_mut().retain(|id| id != worker_id);
            if entry.get().is_empty() {
                entry.remove();
            }
        }
        if let Entry::Occupied(mut entry) = self.connection_workers.entry(worker.connection_mode())
        {
            entry.get_mut().retain(|id| id != worker_id);
            if entry.get().is_empty() {
                entry.remove();
            }
        }
        Some(worker)
    }

    /// Remove a worker by URL
    pub fn remove_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        let _mutation = self
            .mutation
            .lock()
            .expect("worker registry mutation poisoned");
        let removed = self.remove_by_url_unlocked(url);
        if removed.is_some() {
            self.notify_worker_state_change();
        }
        removed
    }

    fn remove_by_url_unlocked(&self, url: &str) -> Option<Arc<dyn Worker>> {
        let worker_id = self.url_to_id.get(url).map(|id| id.clone())?;
        self.remove_unlocked(&worker_id)
    }

    /// Get a worker by ID
    pub fn get(&self, worker_id: &WorkerId) -> Option<Arc<dyn Worker>> {
        self.workers.get(worker_id).map(|entry| entry.clone())
    }

    /// Get a worker by URL
    pub fn get_by_url(&self, url: &str) -> Option<Arc<dyn Worker>> {
        let worker_id = self.url_to_id.get(url).map(|id| id.clone())?;
        self.get(&worker_id)
    }

    /// Get all workers for a model
    pub fn get_by_model(&self, model_id: &str) -> Vec<Arc<dyn Worker>> {
        let ids = self
            .model_workers
            .get(model_id)
            .map(|ids| ids.clone())
            .unwrap_or_default();
        ids.iter().filter_map(|id| self.get(id)).collect()
    }

    /// Get all workers for a model (O(1) optimized version)
    /// This method uses the pre-indexed model_index for fast lookups
    pub fn get_by_model_fast(&self, model_id: &str) -> Vec<Arc<dyn Worker>> {
        self.model_index
            .get(model_id)
            .map(|workers| {
                workers
                    .read()
                    .expect("RwLock for model_index is poisoned")
                    .clone()
            })
            .unwrap_or_default()
    }

    /// Whether the optimized model index contains at least one worker.
    pub fn has_model(&self, model_id: &str) -> bool {
        self.model_index.get(model_id).is_some_and(|workers| {
            !workers
                .read()
                .expect("RwLock for model_index is poisoned")
                .is_empty()
        })
    }

    /// Current worker topology and availability revision.
    pub fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    /// Invalidate target caches after topology or availability changes.
    pub fn notify_worker_state_change(&self) {
        self.revision.fetch_add(1, Ordering::AcqRel);
    }

    /// Get all workers by worker type
    pub fn get_by_type(&self, worker_type: &WorkerType) -> Vec<Arc<dyn Worker>> {
        let ids = self
            .type_workers
            .get(worker_type)
            .map(|ids| ids.clone())
            .unwrap_or_default();
        ids.iter().filter_map(|id| self.get(id)).collect()
    }

    /// Get all prefill workers (regardless of bootstrap_port)
    pub fn get_prefill_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .filter_map(|entry| {
                let worker = entry.value();
                match worker.worker_type() {
                    WorkerType::Prefill { .. } => Some(worker.clone()),
                    _ => None,
                }
            })
            .collect()
    }

    /// Get all decode workers
    pub fn get_decode_workers(&self) -> Vec<Arc<dyn Worker>> {
        self.get_by_type(&WorkerType::Decode)
    }

    /// Get all workers by connection mode
    pub fn get_by_connection(&self, connection_mode: &ConnectionMode) -> Vec<Arc<dyn Worker>> {
        let ids = self
            .connection_workers
            .get(connection_mode)
            .map(|ids| ids.clone())
            .unwrap_or_default();
        ids.iter().filter_map(|id| self.get(id)).collect()
    }

    /// Get all workers
    pub fn get_all(&self) -> Vec<Arc<dyn Worker>> {
        self.workers
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get all workers with their IDs
    pub fn get_all_with_ids(&self) -> Vec<(WorkerId, Arc<dyn Worker>)> {
        self.workers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    /// Get all worker URLs
    pub fn get_all_urls(&self) -> Vec<String> {
        self.workers
            .iter()
            .map(|entry| entry.value().url().to_string())
            .collect()
    }

    /// Get all model IDs with workers
    pub fn get_models(&self) -> Vec<String> {
        self.model_workers
            .iter()
            .filter(|entry| !entry.value().is_empty())
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get workers filtered by multiple criteria
    ///
    /// This method allows flexible filtering of workers based on:
    /// - model_id: Filter by specific model
    /// - worker_type: Filter by worker type (Regular, Prefill, Decode)
    /// - connection_mode: Filter by connection mode
    /// - healthy_only: Only return healthy workers
    pub fn get_workers_filtered(
        &self,
        model_id: Option<&str>,
        worker_type: Option<WorkerType>,
        connection_mode: Option<ConnectionMode>,
        healthy_only: bool,
    ) -> Vec<Arc<dyn Worker>> {
        // Start with the most efficient collection based on filters
        // Use model index when possible as it's O(1) lookup
        let workers = if let Some(model) = model_id {
            self.get_by_model_fast(model)
        } else {
            self.get_all()
        };

        // Apply remaining filters
        workers
            .into_iter()
            .filter(|w| {
                // Check worker_type if specified
                if let Some(ref wtype) = worker_type {
                    if w.worker_type() != *wtype {
                        return false;
                    }
                }

                // Check connection_mode if specified
                if let Some(ref conn) = connection_mode {
                    if w.connection_mode() != *conn {
                        return false;
                    }
                }

                // Check health if required
                if healthy_only && !w.is_healthy() {
                    return false;
                }

                true
            })
            .collect()
    }

    /// Get worker statistics
    pub fn stats(&self) -> WorkerRegistryStats {
        let total_workers = self.workers.len();
        let total_models = self.get_models().len();

        let mut healthy_count = 0;
        let mut total_load = 0;
        let mut regular_count = 0;
        let mut prefill_count = 0;
        let mut decode_count = 0;

        for worker in self.get_all() {
            if worker.is_healthy() {
                healthy_count += 1;
            }
            total_load += worker.load();

            match worker.worker_type() {
                WorkerType::Regular => regular_count += 1,
                WorkerType::Prefill { .. } => prefill_count += 1,
                WorkerType::Decode => decode_count += 1,
            }
        }

        WorkerRegistryStats {
            total_workers,
            total_models,
            healthy_workers: healthy_count,
            total_load,
            regular_workers: regular_count,
            prefill_workers: prefill_count,
            decode_workers: decode_count,
        }
    }

    /// Start a health checker for all workers in the registry
    /// This should be called once after the registry is populated with workers
    pub fn start_health_checker(&self, check_interval_secs: u64) -> crate::core::HealthChecker {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let workers_ref = self.workers.clone();
        let revision = self.revision.clone();

        let handle = tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(check_interval_secs));

            // Health probes must preserve active requests (upstream issue #197).

            loop {
                interval.tick().await;

                // Check for shutdown signal
                if shutdown_clone.load(Ordering::Acquire) {
                    tracing::debug!("Registry health checker shutting down");
                    break;
                }

                // Get all workers from registry
                let workers: Vec<Arc<dyn crate::core::Worker>> = workers_ref
                    .iter()
                    .map(|entry| entry.value().clone())
                    .collect();

                // Perform health checks
                for worker in &workers {
                    let was_available = worker.is_available();
                    let _ = worker.check_health_async().await; // Use async version directly
                    if was_available != worker.is_available() {
                        revision.fetch_add(1, Ordering::AcqRel);
                    }
                }
            }
        });

        crate::core::HealthChecker::new(handle, shutdown)
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for the worker registry
#[derive(Debug, Clone)]
pub struct WorkerRegistryStats {
    pub total_workers: usize,
    pub total_models: usize,
    pub healthy_workers: usize,
    pub total_load: usize,
    pub regular_workers: usize,
    pub prefill_workers: usize,
    pub decode_workers: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, CircuitBreakerConfig, WorkerFactory};
    use std::collections::HashMap;

    fn memory_worker(
        model: &str,
        worker_type: WorkerType,
        mode: ConnectionMode,
    ) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorker::with_connection_mode("http://same:8000".into(), worker_type, mode)
                .with_labels(HashMap::from([("model_id".into(), model.into())])),
        )
    }

    fn assert_registry_empty(registry: &WorkerRegistry) {
        assert!(registry.workers.is_empty());
        assert!(registry.url_to_id.is_empty());
        assert!(registry.model_workers.is_empty());
        assert!(registry.model_index.is_empty());
        assert!(registry.type_workers.is_empty());
        assert!(registry.connection_workers.is_empty());
    }

    #[test]
    fn memory_regression_register_same_url_32_times_is_idempotent() {
        let registry = WorkerRegistry::new();
        let mut first_id = None;
        let mut old_workers = Vec::new();
        for _ in 0..32 {
            let worker = memory_worker("model", WorkerType::Regular, ConnectionMode::Http);
            old_workers.push(Arc::downgrade(&worker));
            let id = registry.register(worker);
            assert_eq!(first_id.get_or_insert(id.clone()), &id);
        }
        assert_eq!(registry.workers.len(), 1);
        assert_eq!(registry.url_to_id.len(), 1);
        assert_eq!(registry.get_by_model("model").len(), 1);
        assert_eq!(registry.get_by_model_fast("model").len(), 1);
        assert_eq!(registry.get_by_type(&WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(&ConnectionMode::Http).len(), 1);
        assert!(old_workers[..31]
            .iter()
            .all(|worker| worker.upgrade().is_none()));
        registry.remove_by_url("http://same:8000");
        assert!(old_workers.iter().all(|worker| worker.upgrade().is_none()));
        assert_registry_empty(&registry);
    }

    #[test]
    fn memory_regression_update_releases_old_model_type_connection_and_worker() {
        let registry = WorkerRegistry::new();
        let prefill = WorkerType::Prefill {
            bootstrap_port: Some(1234),
        };
        let old = memory_worker("old-model", prefill.clone(), ConnectionMode::Http);
        let old_weak = Arc::downgrade(&old);
        let id = registry.register(old);
        let replacement = memory_worker("new-model", WorkerType::Decode, ConnectionMode::Grpc);
        let replacement_weak = Arc::downgrade(&replacement);
        assert_eq!(registry.register(replacement), id);
        assert!(
            old_weak.upgrade().is_none(),
            "old model index retained the worker"
        );
        assert!(!registry.model_workers.contains_key("old-model"));
        assert!(!registry.model_index.contains_key("old-model"));
        assert!(!registry.type_workers.contains_key(&prefill));
        assert!(!registry
            .connection_workers
            .contains_key(&ConnectionMode::Http));
        assert_eq!(registry.get_by_model_fast("new-model").len(), 1);
        assert_eq!(registry.get_decode_workers().len(), 1);
        assert_eq!(registry.get_by_connection(&ConnectionMode::Grpc).len(), 1);
        registry.remove(&id);
        assert!(replacement_weak.upgrade().is_none());
        assert_registry_empty(&registry);
    }

    #[test]
    fn memory_regression_model_churn_does_not_keep_empty_buckets() {
        let registry = WorkerRegistry::new();
        for index in 0..32 {
            registry.register(memory_worker(
                &format!("model-{index}"),
                WorkerType::Regular,
                ConnectionMode::Http,
            ));
            assert_eq!(registry.model_workers.len(), 1);
            assert_eq!(registry.model_index.len(), 1);
        }
        registry.remove_by_url("http://same:8000");
        assert_registry_empty(&registry);
    }

    #[test]
    fn memory_regression_concurrent_same_url_has_one_id_and_one_index_entry() {
        use std::sync::{mpsc, Barrier};
        use std::time::Duration;
        let registry = Arc::new(WorkerRegistry::new());
        let barrier = Arc::new(Barrier::new(32));
        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();
        for _ in 0..32 {
            let (registry, barrier, tx) = (registry.clone(), barrier.clone(), tx.clone());
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                let id = registry.register(memory_worker(
                    "model",
                    WorkerType::Regular,
                    ConnectionMode::Http,
                ));
                tx.send(id).unwrap();
            }));
        }
        drop(tx);
        let ids: std::collections::HashSet<_> = (0..32)
            .map(|_| {
                rx.recv_timeout(Duration::from_secs(10))
                    .expect("registry mutation deadlocked")
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(ids.len(), 1);
        assert_eq!(registry.workers.len(), 1);
        assert_eq!(registry.get_by_model_fast("model").len(), 1);
        assert_eq!(registry.get_by_type(&WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(&ConnectionMode::Http).len(), 1);
        registry.remove_by_url("http://same:8000");
        assert_registry_empty(&registry);
    }

    #[test]
    fn memory_regression_concurrent_register_remove_and_lookup_leave_no_orphans() {
        use std::sync::{mpsc, Barrier};
        use std::time::Duration;
        let registry = Arc::new(WorkerRegistry::new());
        let barrier = Arc::new(Barrier::new(8));
        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();
        for thread in 0..8 {
            let (registry, barrier, tx) = (registry.clone(), barrier.clone(), tx.clone());
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                for round in 0..64 {
                    let model = format!("model-{thread}-{}", round % 2);
                    let id = registry.register(memory_worker(
                        &model,
                        WorkerType::Regular,
                        ConnectionMode::Http,
                    ));
                    registry.get_by_url("http://same:8000");
                    registry.get_by_model(&model);
                    registry.get_by_model_fast(&model);
                    registry.get_by_type(&WorkerType::Regular);
                    registry.get_by_connection(&ConnectionMode::Http);
                    if round % 2 == 0 {
                        registry.remove(&id);
                    } else {
                        registry.remove_by_url("http://same:8000");
                    }
                }
                tx.send(()).unwrap();
            }));
        }
        drop(tx);
        for _ in 0..8 {
            rx.recv_timeout(Duration::from_secs(10))
                .expect("registry lookup/mutation deadlocked");
        }
        for handle in handles {
            handle.join().unwrap();
        }
        registry.remove_by_url("http://same:8000");
        assert_registry_empty(&registry);
    }

    #[test]
    fn test_worker_registry() {
        let registry = WorkerRegistry::new();

        // Create a worker with labels
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "llama-3-8b".to_string());
        labels.insert("priority".to_string(), "50".to_string());
        labels.insert("cost".to_string(), "0.8".to_string());

        let worker = WorkerFactory::create_regular_with_labels(
            "http://worker1:8080".to_string(),
            labels,
            CircuitBreakerConfig::default(),
        );

        // Register worker (WorkerFactory returns Box<dyn Worker>, convert to Arc)
        let worker_id = registry.register(Arc::from(worker));

        // Verify registration
        assert!(registry.get(&worker_id).is_some());
        assert!(registry.get_by_url("http://worker1:8080").is_some());
        assert_eq!(registry.get_by_model("llama-3-8b").len(), 1);
        assert_eq!(registry.get_by_type(&WorkerType::Regular).len(), 1);
        assert_eq!(registry.get_by_connection(&ConnectionMode::Http).len(), 1);

        // Test stats
        let stats = registry.stats();
        assert_eq!(stats.total_workers, 1);
        assert_eq!(stats.total_models, 1);

        // Remove worker
        registry.remove(&worker_id);
        assert!(registry.get(&worker_id).is_none());
    }

    #[test]
    fn test_model_index_fast_lookup() {
        let registry = WorkerRegistry::new();

        // Create workers for different models
        let mut labels1 = HashMap::new();
        labels1.insert("model_id".to_string(), "llama-3".to_string());
        let worker1 = WorkerFactory::create_regular_with_labels(
            "http://worker1:8080".to_string(),
            labels1,
            CircuitBreakerConfig::default(),
        );

        let mut labels2 = HashMap::new();
        labels2.insert("model_id".to_string(), "llama-3".to_string());
        let worker2 = WorkerFactory::create_regular_with_labels(
            "http://worker2:8080".to_string(),
            labels2,
            CircuitBreakerConfig::default(),
        );

        let mut labels3 = HashMap::new();
        labels3.insert("model_id".to_string(), "gpt-4".to_string());
        let worker3 = WorkerFactory::create_regular_with_labels(
            "http://worker3:8080".to_string(),
            labels3,
            CircuitBreakerConfig::default(),
        );

        // Register workers
        registry.register(Arc::from(worker1));
        registry.register(Arc::from(worker2));
        registry.register(Arc::from(worker3));

        // Test get_by_model_fast for llama-3
        let llama_workers = registry.get_by_model_fast("llama-3");
        assert_eq!(llama_workers.len(), 2);
        let urls: Vec<String> = llama_workers.iter().map(|w| w.url().to_string()).collect();
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));

        // Test get_by_model_fast for gpt-4
        let gpt_workers = registry.get_by_model_fast("gpt-4");
        assert_eq!(gpt_workers.len(), 1);
        assert_eq!(gpt_workers[0].url(), "http://worker3:8080");

        // Test get_by_model_fast for non-existent model
        let unknown_workers = registry.get_by_model_fast("unknown-model");
        assert_eq!(unknown_workers.len(), 0);

        // Test that both get_by_model and get_by_model_fast return same results
        let llama_workers_slow = registry.get_by_model("llama-3");
        assert_eq!(llama_workers.len(), llama_workers_slow.len());

        // Test removal updates the model index
        registry.remove_by_url("http://worker1:8080");
        let llama_workers_after = registry.get_by_model_fast("llama-3");
        assert_eq!(llama_workers_after.len(), 1);
        assert_eq!(llama_workers_after[0].url(), "http://worker2:8080");
    }
}
