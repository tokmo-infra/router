//! Bounded API-key validation results. TTLs are fixed at insertion, not refreshed
//! by hits. Only definitive upstream denials may be cached as negative results.
//!
//! This module deliberately uses only std so its production implementation can
//! also be exercised with rustc --test, without network or router dependencies.

use std::{
    collections::{BTreeSet, HashMap},
    fmt,
    sync::Arc,
    time::{Duration, Instant},
};

pub const DEFAULT_CAPACITY: usize = 4096;
pub const DEFAULT_POSITIVE_TTL: Duration = Duration::from_secs(300);
pub const DEFAULT_NEGATIVE_TTL: Duration = Duration::from_secs(30);
pub const DEFAULT_MAX_KEY_BYTES: usize = 8192;

struct Entry {
    valid: bool,
    expires_at: Instant,
}

/// A hard-bounded cache, evicting the soonest-expiring entry when full.
///
/// The expiry index shares key allocations with the map and has exactly one
/// record per entry (no growing queue of stale replacement/eviction records).
/// Oversized keys bypass the cache, but must still be validated by the caller.
pub struct AuthKeyCache {
    entries: HashMap<Arc<str>, Entry>,
    expiry: BTreeSet<(Instant, Arc<str>)>,
    capacity: usize,
    positive_ttl: Duration,
    negative_ttl: Duration,
    max_key_bytes: usize,
}

impl Default for AuthKeyCache {
    fn default() -> Self {
        Self::with_limits(
            DEFAULT_CAPACITY,
            DEFAULT_POSITIVE_TTL,
            DEFAULT_NEGATIVE_TTL,
            DEFAULT_MAX_KEY_BYTES,
        )
    }
}

// Never expose cached credentials through Debug, including when tests fail.
impl fmt::Debug for AuthKeyCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuthKeyCache")
            .field("entries", &self.entries.len())
            .field("capacity", &self.capacity)
            .field("positive_ttl", &self.positive_ttl)
            .field("negative_ttl", &self.negative_ttl)
            .field("max_key_bytes", &self.max_key_bytes)
            .finish()
    }
}

impl AuthKeyCache {
    fn with_limits(
        capacity: usize,
        positive_ttl: Duration,
        negative_ttl: Duration,
        max_key_bytes: usize,
    ) -> Self {
        Self {
            entries: HashMap::new(),
            expiry: BTreeSet::new(),
            capacity,
            positive_ttl,
            negative_ttl,
            max_key_bytes,
        }
    }

    pub fn get(&mut self, key: &str) -> Option<bool> {
        self.get_at(key, Instant::now())
    }

    /// Returns whether the result was retained. Skipping caching must not alter
    /// the caller's authorization result, including for oversized keys.
    pub fn insert(&mut self, key: &str, valid: bool) -> bool {
        self.insert_at(key, valid, Instant::now())
    }

    fn get_at(&mut self, key: &str, now: Instant) -> Option<bool> {
        self.remove_expired(now);
        if key.len() > self.max_key_bytes {
            return None;
        }
        self.entries.get(key).map(|entry| entry.valid)
    }

    fn insert_at(&mut self, key: &str, valid: bool, now: Instant) -> bool {
        self.remove_expired(now);
        if self.capacity == 0 || key.len() > self.max_key_bytes {
            return false;
        }
        let ttl = if valid {
            self.positive_ttl
        } else {
            self.negative_ttl
        };
        let Some(expires_at) = now.checked_add(ttl).filter(|expiry| *expiry > now) else {
            return false;
        };

        if let Some((stored_key, entry)) = self.entries.remove_entry(key) {
            self.expiry.remove(&(entry.expires_at, stored_key));
        }
        while self.entries.len() >= self.capacity {
            if let Some((_, oldest_key)) = self.expiry.pop_first() {
                self.entries.remove(oldest_key.as_ref());
            }
        }

        let key: Arc<str> = Arc::from(key);
        self.expiry.insert((expires_at, Arc::clone(&key)));
        self.entries.insert(key, Entry { valid, expires_at });
        true
    }

    fn remove_expired(&mut self, now: Instant) {
        while let Some((expires_at, _)) = self.expiry.first() {
            if *expires_at > now {
                break;
            }
            if let Some((_, key)) = self.expiry.pop_first() {
                self.entries.remove(key.as_ref());
            }
        }
    }
}

/// Aggregate all validation endpoints without retaining credentials or errors.
/// Preserve the original rule: any HTTP 200 authorizes; otherwise deny this
/// request. Only an all-401/403 denial is safe to cache for the short TTL.
#[derive(Default)]
pub(super) struct ValidationSummary {
    validated: bool,
    denied: bool,
    transient_failure: bool,
}

impl ValidationSummary {
    pub(super) fn record_status(&mut self, status: u16) {
        match status {
            200 => self.validated = true,
            401 | 403 => self.denied = true,
            _ => self.transient_failure = true,
        }
    }

    pub(super) fn record_transport_error(&mut self) {
        self.transient_failure = true;
    }

    pub(super) fn is_valid(&self) -> bool {
        self.validated
    }

    pub(super) fn cache_value(&self) -> Option<bool> {
        if self.validated {
            Some(true)
        } else if self.denied && !self.transient_failure {
            Some(false)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cache(capacity: usize) -> AuthKeyCache {
        AuthKeyCache::with_limits(
            capacity,
            DEFAULT_POSITIVE_TTL,
            DEFAULT_NEGATIVE_TTL,
            DEFAULT_MAX_KEY_BYTES,
        )
    }

    fn assert_bounded(cache: &AuthKeyCache) {
        assert!(cache.entries.len() <= cache.capacity);
        assert_eq!(cache.entries.len(), cache.expiry.len());
        for (expires_at, key) in &cache.expiry {
            assert_eq!(
                cache.entries.get(key.as_ref()).unwrap().expires_at,
                *expires_at
            );
        }
    }

    #[test]
    fn defaults_are_explicit() {
        let cache = AuthKeyCache::default();
        assert_eq!(cache.capacity, 4096);
        assert_eq!(cache.positive_ttl, Duration::from_secs(300));
        assert_eq!(cache.negative_ttl, Duration::from_secs(30));
        assert_eq!(cache.max_key_bytes, 8192);
        assert_eq!(cache.entries.len(), 0);
        assert_bounded(&cache);
    }

    #[test]
    fn positive_expires_at_exact_insert_deadline() {
        let mut cache = AuthKeyCache::default();
        let now = Instant::now();
        assert!(cache.insert_at("positive", true, now));
        assert_eq!(
            cache.get_at("positive", now + Duration::from_secs(299)),
            Some(true)
        );
        assert_eq!(
            cache.get_at("positive", now + Duration::from_secs(300)),
            None
        );
        assert!(cache.entries.is_empty());
        assert_bounded(&cache);
    }

    #[test]
    fn negative_expires_at_exact_insert_deadline() {
        let mut cache = AuthKeyCache::default();
        let now = Instant::now();
        cache.insert_at("negative", false, now);
        assert_eq!(
            cache.get_at("negative", now + Duration::from_secs(29)),
            Some(false)
        );
        assert_eq!(
            cache.get_at("negative", now + Duration::from_secs(30)),
            None
        );
        assert_bounded(&cache);
    }

    #[test]
    fn repeated_hits_do_not_refresh_either_ttl() {
        for (valid, ttl) in [(true, 300), (false, 30)] {
            let mut cache = AuthKeyCache::default();
            let now = Instant::now();
            cache.insert_at("fixed", valid, now);
            for second in 0..ttl {
                assert_eq!(
                    cache.get_at("fixed", now + Duration::from_secs(second)),
                    Some(valid)
                );
            }
            assert_eq!(cache.get_at("fixed", now + Duration::from_secs(ttl)), None);
        }
    }

    #[test]
    fn expired_result_can_be_revalidated_with_opposite_outcome() {
        for (old, ttl) in [(true, 300), (false, 30)] {
            let mut cache = AuthKeyCache::default();
            let now = Instant::now();
            cache.insert_at("revalidate", old, now);
            let later = now + Duration::from_secs(ttl);
            // A miss makes authorize_request visit its upstream-validation loop.
            assert_eq!(cache.get_at("revalidate", later), None);
            cache.insert_at("revalidate", !old, later);
            assert_eq!(cache.get_at("revalidate", later), Some(!old));
            assert_bounded(&cache);
        }
    }

    #[test]
    fn expired_entries_are_reclaimed_even_on_unrelated_miss() {
        let mut cache = AuthKeyCache::default();
        let now = Instant::now();
        cache.insert_at("short", false, now);
        cache.insert_at("long", true, now);
        assert_eq!(cache.get_at("absent", now + Duration::from_secs(30)), None);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(
            cache.get_at("long", now + Duration::from_secs(30)),
            Some(true)
        );
        cache.get_at("absent", now + Duration::from_secs(300));
        assert!(cache.entries.is_empty());
        assert_bounded(&cache);
    }

    #[test]
    fn capacity_and_expiry_index_stay_bounded_under_key_churn() {
        let mut cache = AuthKeyCache::default();
        let now = Instant::now();
        for i in 0..20_000 {
            cache.insert_at(&format!("key-{i}"), i % 2 == 0, now);
            assert_eq!(cache.expiry.len(), cache.entries.len());
            assert!(cache.entries.len() <= DEFAULT_CAPACITY);
        }
        assert_eq!(cache.entries.len(), DEFAULT_CAPACITY);
        assert_bounded(&cache);
    }

    #[test]
    fn replacement_has_one_index_entry_and_renews_from_insert() {
        let mut cache = small_cache(2);
        let now = Instant::now();
        cache.insert_at("replace", false, now);
        for second in 1..1000 {
            cache.insert_at("replace", true, now + Duration::from_secs(second));
            assert_eq!(cache.expiry.len(), 1);
            assert_eq!(cache.entries.len(), 1);
        }
        assert_eq!(
            cache.get_at("replace", now + Duration::from_secs(1298)),
            Some(true)
        );
        assert_eq!(
            cache.get_at("replace", now + Duration::from_secs(1299)),
            None
        );
        assert_bounded(&cache);
    }

    #[test]
    fn replacement_does_not_evict_an_unrelated_entry() {
        let mut cache = small_cache(2);
        let now = Instant::now();
        cache.insert_at("a", true, now);
        cache.insert_at("b", true, now);
        cache.insert_at("a", false, now);
        assert_eq!(cache.get_at("a", now), Some(false));
        assert_eq!(cache.get_at("b", now), Some(true));
        assert_bounded(&cache);
    }

    #[test]
    fn eviction_prefers_soonest_expiry_and_expiration_precedes_eviction() {
        let mut cache = small_cache(2);
        let now = Instant::now();
        cache.insert_at("positive", true, now);
        cache.insert_at("negative", false, now);
        cache.insert_at("new", true, now);
        assert_eq!(cache.get_at("negative", now), None);
        assert_eq!(cache.get_at("positive", now), Some(true));
        cache.insert_at("short", false, now);
        cache.insert_at("after-expiry", true, now + Duration::from_secs(30));
        assert_eq!(
            cache.get_at("positive", now + Duration::from_secs(30)),
            Some(true)
        );
        assert_bounded(&cache);
    }

    #[test]
    fn key_limit_counts_bytes_including_utf8() {
        let mut cache = AuthKeyCache::default();
        let now = Instant::now();
        let at_limit = "x".repeat(8192);
        assert!(cache.insert_at(&at_limit, true, now));
        assert_eq!(cache.get_at(&at_limit, now), Some(true));
        let oversized = "x".repeat(8193);
        assert!(!cache.insert_at(&oversized, false, now));
        assert_eq!(cache.get_at(&oversized, now), None);
        let utf8_limit = "é".repeat(4096);
        assert!(cache.insert_at(&utf8_limit, false, now));
        assert!(!cache.insert_at(&(utf8_limit + "x"), true, now));
        assert_eq!(cache.entries.len(), 2);
        assert_bounded(&cache);
    }

    #[test]
    fn oversized_key_bypass_does_not_evict_usable_results() {
        let mut cache = small_cache(1);
        let now = Instant::now();
        cache.insert_at("keep", true, now);
        let oversized = "x".repeat(DEFAULT_MAX_KEY_BYTES + 1);
        for valid in [true, false] {
            assert!(!cache.insert_at(&oversized, valid, now));
            assert_eq!(cache.get_at("keep", now), Some(true));
        }
        assert_bounded(&cache);
    }

    #[test]
    fn disabled_capacity_and_zero_ttl_do_not_retain_results() {
        let now = Instant::now();
        let mut disabled = small_cache(0);
        assert!(!disabled.insert_at("key", true, now));
        let mut zero_ttl = AuthKeyCache::with_limits(1, Duration::ZERO, Duration::ZERO, 10);
        for valid in [true, false] {
            assert!(!zero_ttl.insert_at("key", valid, now));
        }
        assert_bounded(&disabled);
        assert_bounded(&zero_ttl);
    }

    #[test]
    fn debug_never_contains_key_material() {
        let mut cache = AuthKeyCache::default();
        cache.insert_at("SECRET-API-KEY-never-log", true, Instant::now());
        let debug = format!("{cache:?}");
        assert!(!debug.contains("SECRET-API-KEY"));
        assert!(debug.contains("entries: 1"));
    }

    #[test]
    fn public_clock_methods_use_same_cache() {
        let mut cache = AuthKeyCache::default();
        assert!(cache.insert("public", true));
        assert_eq!(cache.get("public"), Some(true));
        assert_bounded(&cache);
    }

    #[test]
    fn only_http_200_authorizes() {
        for status in [200, 201, 204, 301, 400, 401, 403, 404, 408, 429, 500, 503] {
            let mut result = ValidationSummary::default();
            result.record_status(status);
            assert_eq!(result.is_valid(), status == 200);
        }
    }

    #[test]
    fn only_all_definitive_denials_are_negative_cached() {
        let mut result = ValidationSummary::default();
        assert_eq!(result.cache_value(), None);
        for status in [401, 403, 401] {
            result.record_status(status);
            assert!(!result.is_valid());
            assert_eq!(result.cache_value(), Some(false));
        }
    }

    #[test]
    fn service_errors_rate_limits_and_unexpected_statuses_are_not_cached() {
        for status in [201, 204, 302, 400, 404, 408, 429, 500, 502, 503] {
            for denied_first in [true, false] {
                let mut result = ValidationSummary::default();
                if denied_first {
                    result.record_status(401);
                }
                result.record_status(status);
                result.record_status(403);
                assert!(!result.is_valid());
                assert_eq!(result.cache_value(), None);
            }
        }
    }

    #[test]
    fn transport_error_is_not_negative_cached_even_with_definitive_denial() {
        let mut result = ValidationSummary::default();
        result.record_transport_error();
        assert_eq!(result.cache_value(), None);
        result.record_status(401);
        assert!(!result.is_valid());
        assert_eq!(result.cache_value(), None);
    }

    #[test]
    fn later_success_authorizes_and_is_cached_despite_previous_failures() {
        let mut result = ValidationSummary::default();
        result.record_status(401);
        result.record_status(503);
        result.record_transport_error();
        result.record_status(200);
        assert!(result.is_valid());
        assert_eq!(result.cache_value(), Some(true));
    }
}
