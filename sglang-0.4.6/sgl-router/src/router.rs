use crate::tree::Tree;
use actix_web::http::header::{HeaderValue, CONTENT_TYPE};
use actix_web::{HttpRequest, HttpResponse};
use bytes::Bytes;
use futures_util::{StreamExt, TryStreamExt};
use log::{debug, error, info, warn};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;
use crossbeam_channel::{unbounded, Sender};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug)]
struct UpdateObs {
    worker: String,
    text: String,
    e2e_ms: f64,
    cached_eff_at_route: usize,
    uncached_eff_at_route: usize,
    cur_ms_at_route: f64, 
    inc_ms_at_route: u64,
}

const LEN_SCALE: f64 = 1000.0;          
const BASE_INTERCEPT_MS: f64 = 0.0;    
const SLOPE_CACHED_MS: f64 = 0.0;     
const SLOPE_UNCACHED_MS: f64 = 1000.0; 

const ALPHA_QUEUE: f64 = 1.0; 

const DECAY_NUM: u64 = 31;
const DECAY_DEN: u64 = 32;
const DECAY_INTERVAL_MS: u64 = 20;



#[inline]
fn base_ms(cached: usize, uncached: usize) -> f64 {
    let v = BASE_INTERCEPT_MS
        + SLOPE_CACHED_MS  * (cached as f64 / LEN_SCALE)
        + SLOPE_UNCACHED_MS* (uncached as f64 / LEN_SCALE);
    v.max(1.0)  
}

#[inline]
fn remaining_after_decay(inc_ms: u64, reserve_at: std::time::Instant, now: std::time::Instant) -> u64 {
    let elapsed_ms = now.saturating_duration_since(reserve_at).as_millis() as u64;
    let ticks = (elapsed_ms / DECAY_INTERVAL_MS) as i32;
    if ticks <= 0 { return inc_ms; }
    let factor = (DECAY_NUM as f64 / DECAY_DEN as f64).powi(ticks);
    ((inc_ms as f64) * factor).round() as u64
}



#[inline]
fn phi_bias() -> [f64; 4] { [1.0, 0.0, 0.0, 0.0] }

#[inline]
fn atomic_saturating_sub(cell: &AtomicU64, v: u64) {
    let mut cur = cell.load(Ordering::Relaxed);
    loop {
        let next = cur.saturating_sub(v);
        match cell.compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(x) => cur = x,
        }
    }
}


#[derive(Debug, Clone)]
struct Rls4 {
    theta: [f64; 4],
    p: [[f64; 4]; 4],
    lambda: f64,
    learn_mask: [bool; 4],            
}

impl Rls4 {
    const DEFAULT_LAMBDA: f64 = 0.95;
    const DEFAULT_P0: f64 = 1e6;
    const DEFAULT_THETA: [f64; 4] = [0.0, 0.0, 0.0, 0.0];

    fn new(lambda: f64, p0: f64) -> Self {
        Self {
            theta: Self::DEFAULT_THETA,
            p: [[p0,0.0,0.0,0.0],[0.0,p0,0.0,0.0],[0.0,0.0,p0,0.0],[0.0,0.0,0.0,p0]],
            lambda,
            learn_mask: [true, false, false, false],       
        }
    }

    #[inline] fn predict(&self, phi: [f64; 4]) -> f64 {
        self.theta[0]*phi[0] + self.theta[1]*phi[1] + self.theta[2]*phi[2] + self.theta[3]*phi[3]
    }

    fn update(&mut self, phi: [f64; 4], y: f64) {
        // P * phi
        let mut pphi = [0.0; 4];
        for i in 0..4 { pphi[i] = self.p[i][0]*phi[0] + self.p[i][1]*phi[1] + self.p[i][2]*phi[2] + self.p[i][3]*phi[3]; }
        let denom = self.lambda + phi[0]*pphi[0] + phi[1]*pphi[1] + phi[2]*pphi[2] + phi[3]*pphi[3];
        let mut k = [pphi[0]/denom, pphi[1]/denom, pphi[2]/denom, pphi[3]/denom];
        
        for i in 0..4 { if !self.learn_mask[i] { k[i] = 0.0; } }

        let err = y - self.predict(phi);
        for i in 0..4 { if self.learn_mask[i] { self.theta[i] += k[i]*err; } }

        let mut kphi_p = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                if self.learn_mask[i] {
                    kphi_p[i][j] = k[i] * (phi[0]*self.p[0][j] + phi[1]*self.p[1][j] + phi[2]*self.p[2][j] + phi[3]*self.p[3][j]);
                }
            }
        }
        for i in 0..4 { for j in 0..4 { self.p[i][j] = (self.p[i][j] - kphi_p[i][j]) / self.lambda; } }
    }
}



fn copy_request_headers(req: &HttpRequest) -> Vec<(String, String)> {
    req.headers()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|v| (name.to_string(), v.to_string()))
        })
        .collect()
}

#[derive(Debug)]
pub enum Router {
    RoundRobin {
        worker_urls: Arc<RwLock<Vec<String>>>,
        current_index: AtomicUsize,
        timeout_secs: u64,
        interval_secs: u64,
    },
    Random {
        worker_urls: Arc<RwLock<Vec<String>>>,
        timeout_secs: u64,
        interval_secs: u64,
    },
    CacheAware {
        worker_urls: Arc<RwLock<Vec<String>>>,
        tree: Arc<Mutex<Tree>>,
        running_queue: Arc<Mutex<HashMap<String, usize>>>,
        processed_queue: Arc<Mutex<HashMap<String, usize>>>,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        timeout_secs: u64,
        interval_secs: u64,
        _eviction_thread: Option<thread::JoinHandle<()>>,
    },
    EtaOnline3D {
        worker_urls: Arc<RwLock<Vec<String>>>,
        models: Arc<RwLock<HashMap<String, Rls4>>>,       
        prefix_index: Arc<RwLock<Tree>>,                 
        timeout_secs: u64,
        interval_secs: u64,
        rls_lambda: f64,   
        rls_p0: f64,       
        eviction_interval_secs: u64,            
        max_tree_size: usize,                     
        _eviction_thread: Option<thread::JoinHandle<()>>, 
        update_tx: Sender<UpdateObs>,
        _update_thread: Option<thread::JoinHandle<()>>,
        workloads: Arc<RwLock<Vec<Arc<AtomicU64>>>>,
        w_index: Arc<RwLock<HashMap<String, usize>>>,
    },
}

#[derive(Debug, Clone)]
pub enum PolicyConfig {
    RandomConfig {
        timeout_secs: u64,
        interval_secs: u64,
    },
    RoundRobinConfig {
        timeout_secs: u64,
        interval_secs: u64,
    },
    CacheAwareConfig {
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
        timeout_secs: u64,
        interval_secs: u64,
    },
    EtaOnline3DConfig {
        lambda: f64,
        p0: f64,
        timeout_secs: u64,
        interval_secs: u64,
        eviction_interval_secs: u64,   
        max_tree_size: usize, 
    },
}


fn sse_pop_one_event(buf: &mut Vec<u8>) -> Option<Vec<u8>> {
    if let Some(pos) = twoway::find_bytes(&buf, b"\n\n") {
        let msg = buf[..pos].to_vec();
        buf.drain(..pos+2);
        return Some(msg);
    }
    if let Some(pos) = twoway::find_bytes(&buf, b"\r\n\r\n") {
        let msg = buf[..pos].to_vec();
        buf.drain(..pos+4);
        return Some(msg);
    }
    None
}


impl Router {
    pub fn new(worker_urls: Vec<String>, policy_config: PolicyConfig) -> Result<Self, String> {
        // Get timeout and interval from policy config
        let (timeout_secs, interval_secs) = match &policy_config {
            PolicyConfig::RandomConfig {
                timeout_secs,
                interval_secs,
            } => (*timeout_secs, *interval_secs),
            PolicyConfig::RoundRobinConfig {
                timeout_secs,
                interval_secs,
            } => (*timeout_secs, *interval_secs),
            PolicyConfig::CacheAwareConfig {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
            PolicyConfig::EtaOnline3DConfig { timeout_secs, interval_secs, .. 
            } => (*timeout_secs, *interval_secs),
        };

        // Wait until all workers are healthy
        Self::wait_for_healthy_workers(&worker_urls, timeout_secs, interval_secs)?;

        // Create router based on policy...
        Ok(match policy_config {
            PolicyConfig::RandomConfig {
                timeout_secs,
                interval_secs,
            } => Router::Random {
                worker_urls: Arc::new(RwLock::new(worker_urls)),
                timeout_secs,
                interval_secs,
            },
            PolicyConfig::RoundRobinConfig {
                timeout_secs,
                interval_secs,
            } => Router::RoundRobin {
                worker_urls: Arc::new(RwLock::new(worker_urls)),
                current_index: std::sync::atomic::AtomicUsize::new(0),
                timeout_secs,
                interval_secs,
            },
            PolicyConfig::CacheAwareConfig {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
                timeout_secs,
                interval_secs,
            } => {
                let mut running_queue = HashMap::new();
                for url in &worker_urls {
                    running_queue.insert(url.clone(), 0);
                }

                let mut processed_queue = HashMap::new();
                for url in &worker_urls {
                    processed_queue.insert(url.clone(), 0);
                }

                let tree = Arc::new(Mutex::new(Tree::new()));
                let running_queue = Arc::new(Mutex::new(running_queue));
                let processed_queue = Arc::new(Mutex::new(processed_queue));

                // Create background eviction thread
                let tree_clone = Arc::clone(&tree);
                let processed_queue_clone = Arc::clone(&processed_queue);
                let running_queue_clone = Arc::clone(&running_queue);
                let eviction_thread = thread::spawn(move || {
                    loop {
                        // Sleep for the specified interval
                        thread::sleep(Duration::from_secs(eviction_interval_secs));

                        let locked_tree_clone = tree_clone.lock().unwrap();
                        // Run eviction
                        locked_tree_clone.evict_tenant_by_size(max_tree_size);

                        // Print the process queue
                        let locked_processed_queue = processed_queue_clone.lock().unwrap();
                        info!("Processed Queue: {:?}", locked_processed_queue);

                        // Print the running queue
                        let locked_running_queue = running_queue_clone.lock().unwrap();
                        info!("Running Queue: {:?}", locked_running_queue);
                    }
                });

                for url in &worker_urls {
                    tree.lock().unwrap().insert(&"".to_string(), url);
                }

                Router::CacheAware {
                    worker_urls: Arc::new(RwLock::new(worker_urls)),
                    tree,
                    running_queue,
                    processed_queue,
                    cache_threshold,
                    balance_abs_threshold,
                    balance_rel_threshold,
                    timeout_secs,
                    interval_secs,
                    _eviction_thread: Some(eviction_thread),
                }
            },
            PolicyConfig::EtaOnline3DConfig {
                lambda, p0, timeout_secs, interval_secs, eviction_interval_secs, max_tree_size,
            } => {

                let mut m = HashMap::new();
                for url in &worker_urls {
                     m.insert(url.clone(), Rls4::new(lambda, p0));
                }
                let models = Arc::new(RwLock::new(m));
            
                let mut idx = HashMap::new();
                let mut slots = Vec::with_capacity(worker_urls.len());
                for (i, url) in worker_urls.iter().enumerate() {
                    idx.insert(url.clone(), i);
                    slots.push(Arc::new(AtomicU64::new(0)));
                }
                let workloads = Arc::new(RwLock::new(slots));
                let w_index   = Arc::new(RwLock::new(idx));

                let t = Tree::new();
                for url in &worker_urls { t.insert(&"".to_string(), url); }
                let prefix_index: Arc<RwLock<Tree>> = Arc::new(RwLock::new(t));

                let (update_tx, update_rx) = unbounded::<UpdateObs>();

                let t_clone = Arc::clone(&prefix_index);
                let ev_thread = thread::spawn(move || {
                    loop {
                        thread::sleep(Duration::from_secs(eviction_interval_secs));
                        if let Ok(t) = t_clone.try_write() {
                            t.evict_tenant_by_size(max_tree_size);
                        }
                    }
                });

                let models_for_update = Arc::clone(&models);
                let prefix_for_update = Arc::clone(&prefix_index);

                let upd_thread = thread::spawn(move || {
                    while let Ok(obs) = update_rx.recv() {
                        let UpdateObs {
                            worker, text, e2e_ms,
                            cached_eff_at_route,
                            uncached_eff_at_route,
                            cur_ms_at_route,
                            inc_ms_at_route: _,
                        } = obs;

                        let base = base_ms(cached_eff_at_route, uncached_eff_at_route);

                        let mut y_resid = e2e_ms - (base + ALPHA_QUEUE * cur_ms_at_route);
                        y_resid = y_resid.max(1.0);

                        if let Ok(mut mdl) = models_for_update.write() {
                            if let Some(m) = mdl.get_mut(&worker) {
                                m.update(phi_bias(), y_resid);
                            }
                        }

                        if !text.is_empty() {
                            if let Ok(t) = prefix_for_update.try_write() {
                                t.insert(&text, &worker);
                            }
                        }
                    }
                });

                let workloads_for_decay = Arc::clone(&workloads);
                let _decay_thread = thread::spawn(move || {
                    loop {
                        thread::sleep(Duration::from_millis(DECAY_INTERVAL_MS));
                        if let Ok(slots) = workloads_for_decay.read() {
                            for a in slots.iter() {
                                let cur = a.load(Ordering::Relaxed) as u128;
                                let dec = (cur * DECAY_NUM as u128 / DECAY_DEN as u128) as u64;
                                a.store(dec, Ordering::Relaxed);
                            }
                        }
                    }
                });

                Router::EtaOnline3D {
                    worker_urls: Arc::new(RwLock::new(worker_urls)),
                    models,
                    prefix_index,
                    timeout_secs, interval_secs,
                    rls_lambda: lambda,
                    rls_p0: p0,
                    eviction_interval_secs,
                    max_tree_size,
                    _eviction_thread: Some(ev_thread),
                    update_tx,
                    _update_thread: Some(upd_thread),
                    workloads,
                    w_index
                }
            }
        })
    }
    
    fn wait_for_healthy_workers(
        worker_urls: &[String],
        timeout_secs: u64,
        interval_secs: u64,
    ) -> Result<(), String> {
        let start_time = std::time::Instant::now();
        let sync_client = reqwest::blocking::Client::new();

        loop {
            if start_time.elapsed() > Duration::from_secs(timeout_secs) {
                error!(
                    "Timeout {}s waiting for workers {:?} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    timeout_secs, worker_urls
                );
                return Err(format!(
                    "Timeout {}s waiting for workers {:?} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    timeout_secs, worker_urls
                ));
            }

            let mut all_healthy = true;
            let mut unhealthy_workers = Vec::new();

            for url in worker_urls {
                match sync_client.get(&format!("{}/health", url)).send() {
                    Ok(res) => {
                        if !res.status().is_success() {
                            let msg = format!(
                                "Worker heatlh check is pending with status {}",
                                res.status()
                            );
                            info!("{}", msg);
                            all_healthy = false;
                            unhealthy_workers.push((url, msg));
                        }
                    }
                    Err(_) => {
                        let msg = format!("Worker is not ready yet");
                        info!("{}", msg);
                        all_healthy = false;
                        unhealthy_workers.push((url, msg));
                    }
                }
            }

            if all_healthy {
                info!("All workers are healthy");
                return Ok(());
            } else {
                info!("Initializing workers:");
                for (url, reason) in &unhealthy_workers {
                    info!("  {} - {}", url, reason);
                }
                thread::sleep(Duration::from_secs(interval_secs));
            }
        }
    }

    fn select_first_worker(&self) -> Result<String, String> {
        match self {
            Router::RoundRobin { worker_urls, .. }
            | Router::Random { worker_urls, .. }
            | Router::CacheAware { worker_urls, .. } => {
                if worker_urls.read().unwrap().is_empty() {
                    Err("No workers are available".to_string())
                } else {
                    Ok(worker_urls.read().unwrap()[0].clone())
                }
            }
            | Router::EtaOnline3D { worker_urls, .. } => {              
                if worker_urls.read().unwrap().is_empty() {
                    Err("No workers are available".to_string())
                } else {
                    Ok(worker_urls.read().unwrap()[0].clone())
                }
            }
        }
    }

    async fn send_request(
        &self,
        client: &reqwest::Client,
        worker_url: &str,
        route: &str,
        req: &HttpRequest,
    ) -> HttpResponse {
        let mut request_builder = client.get(format!("{}{}", worker_url, route));

        // Copy all headers from original request except for /health because it does not need authorization
        if route != "/health" {
            for (name, value) in copy_request_headers(req) {
                request_builder = request_builder.header(name, value);
            }
        }

        match request_builder.send().await {
            Ok(res) => {
                let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

                match res.bytes().await {
                    Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                    Err(e) => HttpResponse::InternalServerError()
                        .body(format!("Failed to read response body: {}", e)),
                }
            }
            Err(e) => HttpResponse::InternalServerError().body(format!(
                "Failed to send request to worker {}: {}",
                worker_url, e
            )),
        }
    }

    pub async fn route_to_first(
        &self,
        client: &reqwest::Client,
        route: &str,
        req: &HttpRequest,
    ) -> HttpResponse {
        const MAX_REQUEST_RETRIES: u32 = 3;
        const MAX_TOTAL_RETRIES: u32 = 6;
        let mut total_retries = 0;

        while total_retries < MAX_TOTAL_RETRIES {
            match self.select_first_worker() {
                Ok(worker_url) => {
                    let mut request_retries = 0;

                    // Try the same worker multiple times
                    while request_retries < MAX_REQUEST_RETRIES {
                        if total_retries >= 1 {
                            info!("Retrying request after {} failed attempts", total_retries);
                        }

                        let response = self.send_request(client, &worker_url, route, req).await;

                        if response.status().is_success() {
                            return response;
                        } else {
                            // if the worker is healthy, it means the request is bad, so return the error response
                            let health_response =
                                self.send_request(client, &worker_url, "/health", req).await;
                            if health_response.status().is_success() {
                                return response;
                            }
                        }

                        warn!(
                            "Request to {} failed (attempt {}/{})",
                            worker_url,
                            request_retries + 1,
                            MAX_REQUEST_RETRIES
                        );

                        request_retries += 1;
                        total_retries += 1;

                        if request_retries == MAX_REQUEST_RETRIES {
                            warn!("Removing failed worker: {}", worker_url);
                            self.remove_worker(&worker_url);
                            break;
                        }
                    }
                }
                Err(e) => return HttpResponse::InternalServerError().body(e),
            }
        }

        HttpResponse::InternalServerError().body("All retry attempts failed")
    }

    fn get_text_from_request(&self, body: &Bytes, route: &str) -> String {
        // Convert body to JSON
        let json: Value = match serde_json::from_slice(body) {
            Ok(j) => j,
            Err(_) => {
                warn!("Failed to parse JSON from request body.");
                return String::new();
            }
        };

        match route {
            "/generate" => {
                // For /generate, always use the "text" field.
                match json.get("text").and_then(Value::as_str) {
                    Some(text) => text.to_string(),
                    None => {
                        warn!("No 'text' field found in request body for route /generate.");
                        String::new()
                    }
                }
            }
            "/v1/chat/completions" | "/v1/completions" => {
                // For these routes, try "messages", then "prompt", then "text".
                if let Some(messages) = json.get("messages") {
                    serde_json::to_string(messages).unwrap_or_default()
                } else if let Some(prompt) = json.get("prompt").and_then(Value::as_str) {
                    prompt.to_string()
                } else {
                    warn!("Failed to find 'messages', 'prompt' in request body.");
                    String::new()
                }
            }
            _ => {
                warn!("Unknown route: {} - defaulting to fallback string", route);
                String::new()
            }
        }
    }

    // TODO: return Result<String, String> instead of panicking
    fn select_generate_worker(&self, body: &Bytes, route: &str) -> (String, Option<(f64, usize, usize, f64, u64, std::time::Instant)>) {
        let text = self.get_text_from_request(body, route);

        let worker_url = match self {
            Router::RoundRobin {
                worker_urls,
                current_index,
                ..
            } => {
                let idx = current_index
                    .fetch_update(
                        std::sync::atomic::Ordering::SeqCst,
                        std::sync::atomic::Ordering::SeqCst,
                        |x| Some((x + 1) % worker_urls.read().unwrap().len()),
                    )
                    .unwrap();
                (worker_urls.read().unwrap()[idx].clone(), None)
            }

            Router::Random { worker_urls, .. } => {
                (worker_urls.read().unwrap()
                    [rand::random::<usize>() % worker_urls.read().unwrap().len()]
                .clone(), None)
            }

            Router::CacheAware {
                worker_urls,
                tree,
                running_queue,
                processed_queue,
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                ..
            } => {
                // TODO: delay scheduling if cache hit rate is high because it may cause imbalance. prioritize low hit rate ones
                // let decision_start = std::time::Instant::now();
                let tree = tree.lock().unwrap();
                let mut running_queue = running_queue.lock().unwrap();

                // Get current load statistics
                let max_load = *running_queue.values().max().unwrap_or(&0);
                let min_load = *running_queue.values().min().unwrap_or(&0);

                // Load is considered imbalanced if:
                // 1. (max - min) > abs_threshold AND
                // 2. max > rel_threshold * min
                let is_imbalanced = max_load.saturating_sub(min_load) > *balance_abs_threshold
                    && (max_load as f32) > (min_load as f32 * balance_rel_threshold);

                let selected_url = if is_imbalanced {
                    // Log load balancing trigger and current queue state
                    info!(
                        "Load balancing triggered due to workload imbalance:\n\
                        Max load: {}, Min load: {}\n\
                        Current running queue: {:?}",
                        max_load, min_load, running_queue
                    );

                    // Use shortest queue routing when load is imbalanced
                    running_queue
                        .iter()
                        .min_by_key(|(_url, &count)| count)
                        .map(|(url, _)| url.clone())
                        .unwrap_or_else(|| worker_urls.read().unwrap()[0].clone())
                } else {
                    // Use cache-aware routing when load is balanced
                    let (matched_text, matched_worker) = tree.prefix_match(&text);
                    let matched_rate =
                        matched_text.chars().count() as f32 / text.chars().count() as f32;

                    if matched_rate > *cache_threshold {
                        matched_worker.to_string()
                    } else {
                        tree.get_smallest_tenant()
                    }
                };

                // Update queues and tree
                *running_queue.get_mut(&selected_url).unwrap() += 1;

                *processed_queue
                    .lock()
                    .unwrap()
                    .get_mut(&selected_url)
                    .unwrap() += 1;
                tree.insert(&text, &selected_url);

                // let decision_duration = decision_start.elapsed(); 
                // info!("ETA decision took: {:?}", decision_duration);

                (selected_url, None)
            }

            Router::EtaOnline3D { .. } => {
                let (u, pred, cached_eff, uncached_eff, cur_ms, inc_ms, reserve_at) = self.e2e_reserve_min_ms(&text);
               (u, Some((pred, cached_eff, uncached_eff, cur_ms, inc_ms, reserve_at)))
            }

        };

        worker_url
    }

    async fn send_generate_request(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
        body: &Bytes,
        route: &str,
        worker_url: &str,
        route_meta: Option<(f64, usize, usize, f64, u64, std::time::Instant)>,
        // router_start_opt: Option<std::time::Instant>,
    ) -> HttpResponse {
        let is_stream = serde_json::from_slice::<serde_json::Value>(body.as_ref())
            .map(|v| v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false))
            .unwrap_or(false);

        let (predicted_e2e_ms, cached_eff_at_route, uncached_eff_at_route, cur_ms_at_route, inc_ms_at_route, reserve_at) =route_meta.unwrap_or((0.0, 0, 0, 0.0, 0, std::time::Instant::now()));

        let mut eta3_active = false;
        let mut text_for_update: Option<String> = None;
        let mut t0_opt: Option<std::time::Instant> = None;

        if let Router::EtaOnline3D { .. } = self {
            eta3_active = true;
            text_for_update = Some(self.get_text_from_request(body, route));
            t0_opt = Some(std::time::Instant::now());
        }

        let mut request_builder = client
            .post(format!("{}{}", worker_url, route))
            .body(body.to_vec());

        // Copy all headers from original request
        for (name, value) in copy_request_headers(req) {
            request_builder = request_builder.header(name, value);
        }


        let res = match request_builder.send().await {
        Ok(res) => res,
        Err(_) => {
            if let Router::EtaOnline3D { w_index, workloads, .. } = self {
                if let (Ok(idx_map), Ok(slots)) = (w_index.read(), workloads.read()) {
                    if let Some(&idx) = idx_map.get(worker_url) {
                        let now = std::time::Instant::now();
                        let rem = remaining_after_decay(inc_ms_at_route, reserve_at, now);
                        atomic_saturating_sub(&slots[idx], rem);
                    }
                }
            }
            return HttpResponse::InternalServerError().finish();
        }
    };



        let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

        
        if !status.is_success() {
            let body_bytes = res.bytes().await.unwrap_or_default();
            if let Router::EtaOnline3D { w_index, workloads, .. } = self {
                if let (Ok(idx_map), Ok(slots)) = (w_index.read(), workloads.read()) {
                    if let Some(&idx) = idx_map.get(worker_url) {
                        let now = std::time::Instant::now();
                        let rem = remaining_after_decay(inc_ms_at_route, reserve_at, now);
                        atomic_saturating_sub(&slots[idx], rem);
                    }
                }
            }
            return HttpResponse::build(status).body(body_bytes.to_vec());
        }

        if !is_stream {
            // For non-streaming requests, get response first
            let response = match res.bytes().await {
                Ok(body_bytes) => {
                    let mut builder = HttpResponse::build(status);
                    builder.insert_header((
                        "X-SGLang-Predicted-E2E-Ms",
                        HeaderValue::from_str(&format!("{:.2}", predicted_e2e_ms)).unwrap_or(HeaderValue::from_static("0")),
                    ));
                    builder.insert_header((
                        "X-SGLang-Predicted-Total-Ms",
                        HeaderValue::from_str(&format!("{:.2}", predicted_e2e_ms))
                            .unwrap_or(HeaderValue::from_static("0")),
                    ));
                    builder.body(body_bytes.to_vec())
                }
                Err(e) => {
                    let error_msg = format!("Failed to get response body: {}", e);
                    HttpResponse::InternalServerError().body(error_msg)
                }
            };

            // Then decrement running queue counter if using CacheAware
            if let Router::CacheAware { running_queue, .. } = self {
                if let Ok(mut queue) = running_queue.lock() {
                    if let Some(count) = queue.get_mut(worker_url) {
                        *count = count.saturating_sub(1);
                    }
                }
            }

            // EtaOnline3D
            if eta3_active && response.status().is_success() {
                if let (Some(text), Some(t0)) =
                    (text_for_update.as_ref(), t0_opt.as_ref())
                {
                    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    if let Router::EtaOnline3D { update_tx, .. } = self {
                        let _ = update_tx.try_send(UpdateObs {
                            worker: worker_url.to_string(),
                            text: text.clone(),
                            e2e_ms: latency_ms,
                            cached_eff_at_route,
                            uncached_eff_at_route,
                            cur_ms_at_route,
                            inc_ms_at_route,
                        });
                    }
                }
            }
            
            if let Router::EtaOnline3D { w_index, workloads, .. } = self {
                if let (Ok(idx_map), Ok(slots)) = (w_index.read(), workloads.read()) {
                    if let Some(&idx) = idx_map.get(worker_url) {
                        let now = std::time::Instant::now();
                        let rem = remaining_after_decay(inc_ms_at_route, reserve_at, now);
                        atomic_saturating_sub(&slots[idx], rem);
                    }
                }
            }
            return response;
        } 
        
     
        if let Router::EtaOnline3D { update_tx, workloads, w_index, .. } = self {
                let text  = text_for_update.unwrap_or_default();
                let worker = worker_url.to_string();
                let t0 = t0_opt.unwrap_or_else(std::time::Instant::now);

                let workloads_for_done = Arc::clone(workloads);
                let workloads_for_err  = Arc::clone(workloads);
                let w_index_for_done   = Arc::clone(w_index);
                let w_index_for_err    = Arc::clone(w_index);
                let worker_for_done = worker.clone();
                let worker_for_err  = worker.clone();


                let upd_tx = update_tx.clone();
                let mut buf: Vec<u8> = Vec::new();

                let reserve_at_for_done = reserve_at;
                let reserve_at_for_err  = reserve_at;
                let inc_ms_for_done = inc_ms_at_route;
                let inc_ms_for_err  = inc_ms_at_route;

                return HttpResponse::build(status)
                    .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                    .insert_header(("Cache-Control", HeaderValue::from_static("no-cache")))
                    .insert_header(("Connection", HeaderValue::from_static("keep-alive")))
                    .insert_header(("X-Accel-Buffering", HeaderValue::from_static("no")))
                    .insert_header(("X-SGLang-Predicted-E2E-Ms",
                        HeaderValue::from_str(&format!("{:.2}", predicted_e2e_ms)).unwrap_or(HeaderValue::from_static("0"))))
                    .streaming(
                        res.bytes_stream()
                        .map_err(|_| actix_web::error::ErrorInternalServerError("Failed to read stream"))
                        .inspect_err(move |_| {
                            if let (Ok(idx_map), Ok(slots)) = (w_index_for_err.read(), workloads_for_err.read()) {
                                if let Some(&idx) = idx_map.get(&worker_for_err) {
                                    let now = std::time::Instant::now();
                                    let rem = remaining_after_decay(inc_ms_for_err, reserve_at_for_err, now);
                                    atomic_saturating_sub(&slots[idx], rem);
                                }
                            }
                        })
                        .inspect(move |bytes| {
                            if let Ok(b) = bytes {
                                buf.extend_from_slice(b.as_ref());
                                loop {
                                    if let Some(msg) = sse_pop_one_event(&mut buf) {
                                        const DONE: &[u8] = b"data: [DONE]";
                                        let is_done = msg.windows(DONE.len()).any(|w| w == DONE);
                                        if is_done {
                                            let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

                                            let _ = upd_tx.try_send(UpdateObs {
                                                worker: worker_for_done.clone(),
                                                text: text.clone(),
                                                e2e_ms: latency_ms,
                                                cached_eff_at_route,
                                                uncached_eff_at_route,
                                                cur_ms_at_route,
                                                inc_ms_at_route,
                                            });

                                            if let (Ok(idx_map), Ok(slots)) = (w_index_for_done.read(), workloads_for_done.read()) {
                                                if let Some(&idx) = idx_map.get(&worker_for_done) {
                                                    let now = std::time::Instant::now();
                                                    let rem = remaining_after_decay(inc_ms_for_done, reserve_at_for_done, now);
                                                    atomic_saturating_sub(&slots[idx], rem);
                                                }
                                            }
                                        }
                                    } else { break; }
                                }
                            }
                        }),
                    );
            } else if let Router::CacheAware { running_queue, .. } = self {
                let running_queue = Arc::clone(running_queue);
                let worker_url = worker_url.to_string();

                return HttpResponse::build(status)
                    .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                    .insert_header(("Cache-Control", HeaderValue::from_static("no-cache")))
                    .insert_header(("Connection", HeaderValue::from_static("keep-alive")))
                    .insert_header(("X-Accel-Buffering", HeaderValue::from_static("no")))
                    .streaming(
                        res.bytes_stream()
                            .map_err(|_| {
                                actix_web::error::ErrorInternalServerError("Failed to read stream")
                            })
                            .inspect(move |bytes| {
                                if let Ok(chunk) = bytes {
                                    let b = chunk.as_ref();
                                    if b.windows(12).any(|w| w == b"data: [DONE]") {
                                        let mut locked = running_queue.lock().unwrap();
                                        if let Some(c) = locked.get_mut(&worker_url) {
                                            *c = c.saturating_sub(1);
                                        }
                                        debug!("Streaming is done!!");
                                    }
                                }
                            }),
                );
        } else {
            return HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .insert_header(("Cache-Control", HeaderValue::from_static("no-cache")))
                .insert_header(("Connection", HeaderValue::from_static("keep-alive")))
                .insert_header(("X-Accel-Buffering", HeaderValue::from_static("no")))
                .streaming(res.bytes_stream().map_err(|_| {
                    actix_web::error::ErrorInternalServerError("Failed to read stream")
                }));
        }
    }

    pub async fn route_generate_request(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
        body: &Bytes,
        route: &str,
    ) -> HttpResponse {
        const MAX_REQUEST_RETRIES: u32 = 3;
        const MAX_TOTAL_RETRIES: u32 = 6;
        let mut total_retries = 0;

        while total_retries < MAX_TOTAL_RETRIES {

            let (worker_url, wait_opt) = self.select_generate_worker(body, route);
            let mut request_retries = 0;

            // Try the same worker multiple times
            while request_retries < MAX_REQUEST_RETRIES {
                if total_retries >= 1 {
                    info!("Retrying request after {} failed attempts", total_retries);
                }
                let response = self
                    .send_generate_request(client, req, body, route, &worker_url, wait_opt,) 
                    .await;

                if response.status().is_success() {
                    return response;
                } else {
                    // if the worker is healthy, it means the request is bad, so return the error response
                    let health_response =
                        self.send_request(client, &worker_url, "/health", req).await;
                    if health_response.status().is_success() {
                        return response;
                    }
                }

                warn!(
                    "Generate request to {} failed (attempt {}/{})",
                    worker_url,
                    request_retries + 1,
                    MAX_REQUEST_RETRIES
                );

                request_retries += 1;
                total_retries += 1;

                if request_retries == MAX_REQUEST_RETRIES {
                    warn!("Removing failed worker: {}", worker_url);
                    self.remove_worker(&worker_url);
                    break;
                }
            }
        }

        HttpResponse::InternalServerError().body("All retry attempts failed")
    }

    pub async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
        let (timeout_secs, interval_secs) = match self {
            Router::Random {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
            Router::RoundRobin {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
            Router::CacheAware {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
            Router::EtaOnline3D { 
                timeout_secs, 
                interval_secs, 
                .. 
            } => (*timeout_secs, *interval_secs),
        };

        let start_time = std::time::Instant::now();
        let client = reqwest::Client::new();

        loop {
            if start_time.elapsed() > Duration::from_secs(timeout_secs) {
                error!(
                    "Timeout {}s waiting for worker {} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    timeout_secs, worker_url
                );
                return Err(format!(
                    "Timeout {}s waiting for worker {} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    timeout_secs, worker_url
                ));
            }

            match client.get(&format!("{}/health", worker_url)).send().await {
                Ok(res) => {
                    if res.status().is_success() {
                        match self {
                            Router::RoundRobin { worker_urls, .. }
                            | Router::Random { worker_urls, .. }
                            | Router::CacheAware { worker_urls, .. } => {
                                info!("Worker {} health check passed", worker_url);
                                let mut urls = worker_urls.write().unwrap();
                                if urls.contains(&worker_url.to_string()) {
                                    return Err(format!("Worker {} already exists", worker_url));
                                }
                                info!("Added worker: {}", worker_url);
                                urls.push(worker_url.to_string());
                            }
                            | Router::EtaOnline3D { worker_urls, .. } => {
                                info!("Worker {} health check passed", worker_url);
                                let mut urls = worker_urls.write().unwrap();
                                if urls.contains(&worker_url.to_string()) {
                                    return Err(format!("Worker {} already exists", worker_url));
                                }
                                info!("Added worker: {}", worker_url);
                                urls.push(worker_url.to_string());
                            }
                        }

                        // If cache aware, initialize the queues for the new worker
                        if let Router::CacheAware {
                            running_queue,
                            processed_queue,
                            tree,
                            ..
                        } = self
                        {
                            // Add worker to running queue with initial count of 0
                            running_queue
                                .lock()
                                .unwrap()
                                .insert(worker_url.to_string(), 0);

                            // Add worker to processed queue with initial count of 0
                            processed_queue
                                .lock()
                                .unwrap()
                                .insert(worker_url.to_string(), 0);

                            // Add worker to tree
                            tree.lock().unwrap().insert(&"".to_string(), &worker_url);
                        }

                        if let Router::EtaOnline3D {
                            models,
                            prefix_index,
                            rls_lambda,
                            rls_p0,
                            workloads, 
                            w_index,
                            ..
                        } = self
                        {

                            prefix_index
                                .write()
                                .unwrap()
                                .insert(&"".to_string(), &worker_url);

                            models
                                .write()
                                .unwrap()
                                .insert(worker_url.to_string(), Rls4::new(*rls_lambda, *rls_p0));

                            let mut slots = workloads.write().unwrap();
                            let mut map   = w_index.write().unwrap();
                            let idx = slots.len();
                            slots.push(Arc::new(AtomicU64::new(0)));
                            map.insert(worker_url.to_string(), idx);
                        }

                        return Ok(format!("Successfully added worker: {}", worker_url));
                    } else {
                        info!(
                            "Worker {} health check is pending with status: {}.",
                            worker_url,
                            res.status()
                        );
                        // if the url does not have http or https prefix, warn users
                        if !worker_url.starts_with("http://") && !worker_url.starts_with("https://")
                        {
                            warn!("The worker url {} does not have http or https prefix. Please add the prefix to the url.", worker_url);
                        }

                        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                        continue;
                    }
                }
                Err(e) => {
                    info!(
                        "Worker {} health check is pending with error: {}",
                        worker_url, e
                    );

                    // if the url does not have http or https prefix, warn users
                    if !worker_url.starts_with("http://") && !worker_url.starts_with("https://") {
                        warn!("The worker url {} does not have http or https prefix. Please add the prefix to the url.", worker_url);
                    }

                    tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                    continue;
                }
            }
        }
    }

    pub fn remove_worker(&self, worker_url: &str) {
        match self {
            Router::RoundRobin { worker_urls, .. }
            | Router::Random { worker_urls, .. }
            | Router::CacheAware { worker_urls, .. } => {
                let mut urls = worker_urls.write().unwrap();
                if let Some(index) = urls.iter().position(|url| url == &worker_url) {
                    urls.remove(index);
                    info!("Removed worker: {}", worker_url);
                } else {
                    warn!("Worker {} not found, skipping removal", worker_url);
                    return;
                }
            }
            | Router::EtaOnline3D { worker_urls, .. } => {
                let mut urls = worker_urls.write().unwrap();
                if let Some(index) = urls.iter().position(|url| url == worker_url) {
                    urls.remove(index);
                    info!("Removed worker: {}", worker_url);
                } else {
                    warn!("Worker {} not found, skipping removal", worker_url);
                    return;
                }
            }
        }

        // if cache aware, remove the worker from the tree
        if let Router::CacheAware {
            tree,
            running_queue,
            processed_queue,
            ..
        } = self
        {
            tree.lock().unwrap().remove_tenant(&worker_url);
            running_queue
                .lock()
                .unwrap()
                .remove(&worker_url.to_string());
            processed_queue
                .lock()
                .unwrap()
                .remove(&worker_url.to_string());
            info!(
                "Removed worker from tree and cleaned up queues: {}",
                worker_url
            );
        }
        if let Router::EtaOnline3D {
            models,
            prefix_index,
            workloads,
            w_index,
            ..
        } = self
        {   
            prefix_index.write().unwrap().remove_tenant(worker_url);
            models.write().unwrap().remove(worker_url);
            let mut map = w_index.write().unwrap();
            if let Some(idx) = map.remove(worker_url) {
                let mut slots = workloads.write().unwrap();
                let last = slots.len() - 1;
                slots.swap_remove(idx);
                if idx != last {
                    if let Some((moved_worker, _)) = map.iter().find(|(_, &i)| i == last).map(|(u, i)| (u.clone(), i)) {
                        map.insert(moved_worker, idx);
                    }
                }
            }
            info!("Removed worker from ETA/model/prefix: {}", worker_url);
        }

    }
}


impl Router {  
    fn e2e_reserve_min_ms(&self, text: &str) -> (String, f64, usize, usize, f64, u64, std::time::Instant) {
        let (urls, total, cached_per_url) = if let Router::EtaOnline3D { worker_urls, prefix_index, .. } = self {
            let urls = worker_urls.read().unwrap().clone();
            let total = text.chars().count();
            let cached_per_url = {
                let t = prefix_index.read().unwrap();
                urls.iter().map(|u| t.prefix_len_tenant_no_touch(text, u)).collect::<Vec<_>>()
            };
            (urls, total, cached_per_url)
        } else { unreachable!() };

        if let Router::EtaOnline3D { models, workloads, prefix_index, .. } = self {
            let m = models.read().unwrap();
            let slots = workloads.read().unwrap().clone();
            let mut best: Option<(usize, f64, usize, usize, f64)> = None;
            let mut best_any: Option<(usize, f64, usize, usize, f64)> = None;

            for (i, u) in urls.iter().enumerate() {
                let cached0 = cached_per_url[i];
                let cur_ms  = slots[i].load(Ordering::Relaxed) as f64;

                let cached_eff = cached0;
                let uncached_eff = total.saturating_sub(cached_eff);
                let base_eff = base_ms(cached_eff, uncached_eff);
                let pred_lin = base_eff + ALPHA_QUEUE * cur_ms;
                let resid    = m.get(u).map(|mm| mm.predict(phi_bias())).unwrap_or(0.0);
                let pred = (pred_lin + resid).max(1.0);

                match best_any {
                    None => best_any = Some((i, pred, cached_eff, uncached_eff, cur_ms)),
                    Some((_bi, bp, ..)) if pred + 1.0 < bp => best_any = Some((i, pred, cached_eff, uncached_eff, cur_ms)),
                    _ => {}
                }

                match best {
                    None => best = Some((i, pred, cached_eff, uncached_eff, cur_ms)),
                    Some((_bi, bp, ..)) if pred + 1.0 < bp => best = Some((i, pred, cached_eff, uncached_eff, cur_ms)),
                    _ => {}
                }
            }

            let (k, pred_e2e_ms, cached_eff, uncached_eff, cur_ms_at_route) = best.or(best_any).expect("no worker");
            let chosen = urls[k].clone();

            let inc_ms = base_ms(cached_eff, uncached_eff).round() as u64;
            let reserve_at = std::time::Instant::now(); 
            slots[k].fetch_add(inc_ms, Ordering::Relaxed);

            if !text.is_empty() {
                if let Ok(t) = prefix_index.try_write() {
                    t.insert(text, &chosen);
                }
            }

            return (chosen, pred_e2e_ms, cached_eff, uncached_eff, cur_ms_at_route, inc_ms, reserve_at);
        }
        unreachable!()
    }

}
