mod trip;

use anyhow::{Context, Result, anyhow};
use autoagents::async_trait;
use autoagents::core::agent::memory::{MemoryProvider, SlidingWindowMemory};
use autoagents::core::agent::prebuilt::executor::{ReActAgent, ReActAgentOutput};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{
    AgentBuilder, AgentHooks, AgentOutputT, Context as AgentContext, DirectAgent,
};
use autoagents::core::tool::{ToolCallError, ToolCallResult, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::LLMProvider;
use autoagents::llm::ToolCall;
use autoagents::llm::backends::openai::OpenAI;
use autoagents::llm::builder::LLMBuilder;
use autoagents::llm::chat::{
    ChatMessage, ChatResponse, StreamChunk, StreamResponse, StructuredOutputFormat, Tool,
    ToolChoice,
};
use autoagents::llm::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
use autoagents::llm::embedding::EmbeddingProvider;
use autoagents::llm::error::LLMError;
use autoagents::llm::models::{ModelListRequest, ModelListResponse, ModelsProvider};
use autoagents_derive::{AgentOutput, ToolInput, agent, tool};
use futures::stream::{FuturesUnordered, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::fs;
use std::mem::MaybeUninit;
use std::path::Path;
use std::pin::Pin;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::task::{Context as TaskContext, Poll};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio::sync::Mutex as TokioMutex;

use crate::trip::compute_average_trip_duration;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct TripDataProcessingArgs {}

#[tool(
    name = "TripDataProcessingTool",
    description = "Calculate average trip duration from the TLC trip_data.parquet dataset",
    input = TripDataProcessingArgs,
)]
pub struct TripDataProcessorTool {}

impl Default for TripDataProcessorTool {
    fn default() -> Self {
        Self {}
    }
}

struct ToolTimingGuard {
    start: Instant,
}

impl ToolTimingGuard {
    fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}

impl Drop for ToolTimingGuard {
    fn drop(&mut self) {
        record_tool_timing(self.start.elapsed());
    }
}

static TOOL_TIMINGS: OnceLock<Mutex<Vec<Duration>>> = OnceLock::new();
static TOOL_START_TIMES: OnceLock<Mutex<HashMap<usize, Instant>>> = OnceLock::new();

fn tool_start_times() -> &'static Mutex<HashMap<usize, Instant>> {
    TOOL_START_TIMES.get_or_init(|| Mutex::new(HashMap::new()))
}

tokio::task_local! {
    static REQUEST_ID: usize;
}

tokio::task_local! {
    static REQUEST_MEMORY: Arc<TokioMutex<Box<dyn MemoryProvider>>>;
}

fn tool_timings() -> &'static Mutex<Vec<Duration>> {
    TOOL_TIMINGS.get_or_init(|| Mutex::new(Vec::new()))
}

fn reset_tool_timings() {
    if let Ok(mut timings) = tool_timings().lock() {
        timings.clear();
    }
}

fn record_tool_timing(duration: Duration) {
    if let Ok(mut timings) = tool_timings().lock() {
        timings.push(duration);
    }
}

fn snapshot_tool_timings() -> Vec<Duration> {
    tool_timings()
        .lock()
        .map(|timings| timings.clone())
        .unwrap_or_default()
}

#[derive(Default)]
struct LlmTimingRecorder {
    call_durations: Mutex<Vec<Duration>>,
    request_durations: Mutex<HashMap<usize, Vec<Duration>>>,
}

impl LlmTimingRecorder {
    fn record(&self, duration: Duration) {
        if let Ok(mut durations) = self.call_durations.lock() {
            durations.push(duration);
        }

        if let Ok(request_id) = REQUEST_ID.try_with(|id| *id) {
            if let Ok(mut per_request) = self.request_durations.lock() {
                per_request.entry(request_id).or_default().push(duration);
            }
        }
    }

    fn reset(&self) {
        if let Ok(mut durations) = self.call_durations.lock() {
            durations.clear();
        }
        if let Ok(mut per_request) = self.request_durations.lock() {
            per_request.clear();
        }
    }

    fn snapshot_request_totals(&self) -> Vec<Duration> {
        let mut totals = Vec::new();
        if let Ok(per_request) = self.request_durations.lock() {
            for durations in per_request.values() {
                let total = durations
                    .iter()
                    .copied()
                    .fold(Duration::from_secs(0), |acc, next| acc + next);
                totals.push(total);
            }
        }
        totals
    }
}

#[derive(Clone)]
struct TimedLlm {
    inner: Arc<dyn LLMProvider>,
    timings: Arc<LlmTimingRecorder>,
}

impl TimedLlm {
    fn new(inner: Arc<dyn LLMProvider>, timings: Arc<LlmTimingRecorder>) -> Self {
        Self { inner, timings }
    }
}

struct TimedStream<T> {
    inner: Pin<Box<dyn Stream<Item = Result<T, LLMError>> + Send>>,
    start: Instant,
    timings: Arc<LlmTimingRecorder>,
    recorded: bool,
}

impl<T> TimedStream<T> {
    fn new(
        inner: Pin<Box<dyn Stream<Item = Result<T, LLMError>> + Send>>,
        start: Instant,
        timings: Arc<LlmTimingRecorder>,
    ) -> Self {
        Self {
            inner,
            start,
            timings,
            recorded: false,
        }
    }

    fn record_once(&mut self) {
        if !self.recorded {
            self.recorded = true;
            self.timings.record(self.start.elapsed());
        }
    }
}

impl<T> Stream for TimedStream<T> {
    type Item = Result<T, LLMError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let poll = self.inner.as_mut().poll_next(cx);
        if matches!(poll, Poll::Ready(None)) {
            self.record_once();
        }
        poll
    }
}

impl<T> Drop for TimedStream<T> {
    fn drop(&mut self) {
        self.record_once();
    }
}

impl<T> Unpin for TimedStream<T> {}

#[async_trait]
impl autoagents::llm::chat::ChatProvider for TimedLlm {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let start = Instant::now();
        let result = self
            .inner
            .chat_with_tools(messages, tools, json_schema)
            .await;
        self.timings.record(start.elapsed());
        result
    }

    async fn chat_with_web_search(&self, input: String) -> Result<Box<dyn ChatResponse>, LLMError> {
        let start = Instant::now();
        let result = self.inner.chat_with_web_search(input).await;
        self.timings.record(start.elapsed());
        result
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        let start = Instant::now();
        match self.inner.chat_stream(messages, json_schema).await {
            Ok(stream) => Ok(Box::pin(TimedStream::new(
                stream,
                start,
                self.timings.clone(),
            ))),
            Err(err) => {
                self.timings.record(start.elapsed());
                Err(err)
            }
        }
    }

    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
    {
        let start = Instant::now();
        match self
            .inner
            .chat_stream_struct(messages, tools, json_schema)
            .await
        {
            Ok(stream) => Ok(Box::pin(TimedStream::new(
                stream,
                start,
                self.timings.clone(),
            ))),
            Err(err) => {
                self.timings.record(start.elapsed());
                Err(err)
            }
        }
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        let start = Instant::now();
        match self
            .inner
            .chat_stream_with_tools(messages, tools, json_schema)
            .await
        {
            Ok(stream) => Ok(Box::pin(TimedStream::new(
                stream,
                start,
                self.timings.clone(),
            ))),
            Err(err) => {
                self.timings.record(start.elapsed());
                Err(err)
            }
        }
    }
}

#[async_trait]
impl CompletionProvider for TimedLlm {
    async fn complete(
        &self,
        req: &CompletionRequest,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        let start = Instant::now();
        let result = self.inner.complete(req, json_schema).await;
        self.timings.record(start.elapsed());
        result
    }
}

#[async_trait]
impl EmbeddingProvider for TimedLlm {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        self.inner.embed(input).await
    }
}

#[async_trait]
impl ModelsProvider for TimedLlm {
    async fn list_models(
        &self,
        request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        self.inner.list_models(request).await
    }
}

impl LLMProvider for TimedLlm {}

#[derive(Default)]
struct TaskLocalMemory;

#[async_trait]
impl MemoryProvider for TaskLocalMemory {
    async fn remember(&mut self, message: &ChatMessage) -> Result<(), LLMError> {
        let memory = REQUEST_MEMORY
            .try_with(|mem| mem.clone())
            .map_err(|_| LLMError::ProviderError("request memory not set".to_string()))?;
        let mut guard = memory.lock().await;
        guard.remember(message).await
    }

    async fn recall(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ChatMessage>, LLMError> {
        let memory = REQUEST_MEMORY
            .try_with(|mem| mem.clone())
            .map_err(|_| LLMError::ProviderError("request memory not set".to_string()))?;
        let guard = memory.lock().await;
        guard.recall(query, limit).await
    }

    async fn clear(&mut self) -> Result<(), LLMError> {
        let memory = REQUEST_MEMORY
            .try_with(|mem| mem.clone())
            .map_err(|_| LLMError::ProviderError("request memory not set".to_string()))?;
        let mut guard = memory.lock().await;
        guard.clear().await
    }

    fn memory_type(&self) -> autoagents::core::agent::memory::MemoryType {
        autoagents::core::agent::memory::MemoryType::SlidingWindow
    }

    fn size(&self) -> usize {
        0
    }

    fn clone_box(&self) -> Box<dyn MemoryProvider> {
        Box::new(Self)
    }
}

#[async_trait]
impl ToolRuntime for TripDataProcessorTool {
    async fn execute(&self, _args: Value) -> Result<Value, ToolCallError> {
        let path =
            std::env::var("TRIP_DATA_PATH").unwrap_or_else(|_| "trip_data.parquet".to_string());

        // Offload CPU-heavy work to a blocking thread
        let aggregate =
            tokio::task::spawn_blocking(move || compute_average_trip_duration(Path::new(&path)))
                .await
                .map_err(|err| ToolCallError::RuntimeError(Box::new(err)))? // Join error
                .map_err(|err| ToolCallError::RuntimeError(Box::new(err)))?; // Your compute error

        // println!("Aggregate: {:?}", aggregate);

        let summary = format!(
            "Average trip duration is {:.2} minutes across {} trips.",
            aggregate.average_trip_duration_minutes, aggregate.row_count
        );

        Ok(Value::String(summary))
    }
}

/// Math agent output with Value and Explanation
#[derive(Debug, Serialize, Deserialize, AgentOutput, schemars::JsonSchema)]
pub struct SimpleAgentOutput {
    #[output(description = "The Average Trip Duration")]
    value: f64,
}

fn extract_last_number(text: &str) -> Option<f64> {
    let bytes = text.as_bytes();
    let mut idx = 0;
    let mut last_value = None;
    while idx < bytes.len() {
        let b = bytes[idx];
        let is_start = b.is_ascii_digit() || b == b'-';
        if !is_start {
            idx += 1;
            continue;
        }

        let start = idx;
        idx += 1;
        while idx < bytes.len() {
            let b = bytes[idx];
            if b.is_ascii_digit() || b == b'.' {
                idx += 1;
            } else {
                break;
            }
        }

        if let Ok(value) = text[start..idx].parse::<f64>() {
            last_value = Some(value);
        }
    }
    last_value
}

fn extract_number_before_minutes(text: &str) -> Option<f64> {
    let lower = text.to_ascii_lowercase();
    let minutes_idx = lower.rfind("minutes")?;
    let bytes = text.as_bytes();
    let mut end = minutes_idx.min(bytes.len());
    while end > 0 {
        let b = bytes[end - 1];
        if b.is_ascii_digit() || b == b'.' || b == b'-' {
            break;
        }
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let mut start = end;
    while start > 0 {
        let b = bytes[start - 1];
        if b.is_ascii_digit() || b == b'.' || b == b'-' {
            start -= 1;
        } else {
            break;
        }
    }
    if start >= end {
        return None;
    }
    text[start..end].trim().parse::<f64>().ok()
}

fn parse_value(text: &str) -> Option<f64> {
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
        for key in ["value", "answer", "average_trip_duration_minutes"] {
            if let Some(val) = parsed.get(key) {
                if let Some(num) = val.as_f64() {
                    return Some(num);
                }
                if let Some(num_str) = val.as_str() {
                    if let Ok(num) = num_str.parse::<f64>() {
                        return Some(num);
                    }
                }
            }
        }
    }
    if let Some(value) = extract_number_before_minutes(text) {
        return Some(value);
    }
    extract_last_number(text)
}

impl From<ReActAgentOutput> for SimpleAgentOutput {
    fn from(output: ReActAgentOutput) -> Self {
        let resp = output.response;
        if output.done && !resp.trim().is_empty() {
            // Try to parse as structured JSON first
            if let Ok(value) = serde_json::from_str::<SimpleAgentOutput>(&resp) {
                return value;
            }
            if let Some(value) = parse_value(&resp) {
                return SimpleAgentOutput { value };
            }
        }
        if output.done && !output.tool_calls.is_empty() {
            for tool_call in output.tool_calls.iter().rev() {
                if !tool_call.success {
                    continue;
                }
                match &tool_call.result {
                    Value::Number(num) => {
                        if let Some(value) = num.as_f64() {
                            return SimpleAgentOutput { value };
                        }
                    }
                    Value::String(text) => {
                        if let Some(value) = parse_value(text) {
                            return SimpleAgentOutput { value };
                        }
                    }
                    Value::Object(map) => {
                        for key in ["value", "answer", "average_trip_duration_minutes"] {
                            if let Some(val) = map.get(key) {
                                if let Some(value) = val.as_f64() {
                                    return SimpleAgentOutput { value };
                                }
                                if let Some(val_str) = val.as_str() {
                                    if let Ok(value) = val_str.parse::<f64>() {
                                        return SimpleAgentOutput { value };
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        // For streaming chunks or unparseable content, create a default response
        SimpleAgentOutput { value: 0.0 }
    }
}

#[agent(
    name = "data_processing_agent",
    description = "You are a helpful assistant, provide the answer to the user's question in structured format",
    tools = [TripDataProcessorTool],
    output = SimpleAgentOutput,
)]
#[derive(Default, Clone)]
pub struct SimpleAgent {}

#[async_trait]
impl AgentHooks for SimpleAgent {
    async fn on_tool_start(&self, _tool_call: &ToolCall, _ctx: &AgentContext) {
        if let Ok(req_id) = REQUEST_ID.try_with(|id| *id) {
            if let Ok(mut starts) = tool_start_times().lock() {
                starts.insert(req_id, Instant::now());
            }
        }
    }

    async fn on_tool_result(
        &self,
        _tool_call: &ToolCall,
        _result: &ToolCallResult,
        _ctx: &AgentContext,
    ) {
        if let Ok(req_id) = REQUEST_ID.try_with(|id| *id) {
            if let Ok(mut starts) = tool_start_times().lock() {
                if let Some(start) = starts.remove(&req_id) {
                    record_tool_timing(start.elapsed());
                }
            }
        }
    }

    async fn on_tool_error(&self, _tool_call: &ToolCall, _err: Value, _ctx: &AgentContext) {
        if let Ok(req_id) = REQUEST_ID.try_with(|id| *id) {
            if let Ok(mut starts) = tool_start_times().lock() {
                if let Some(start) = starts.remove(&req_id) {
                    record_tool_timing(start.elapsed());
                }
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
struct BenchmarkConfig {
    total_requests: usize,
    concurrency: usize,
    prompt_template: String,
    model: String,
    #[serde(default = "default_request_timeout_secs")]
    request_timeout_secs: u64,
}

fn default_request_timeout_secs() -> u64 {
    120
}

#[derive(Debug)]
struct BenchmarkResult {
    name: &'static str,
    total_requests: usize,
    concurrency: usize,
    total_duration: Duration,
    setup_duration_s: f64,
    throughput_rps: f64,
    throughput_success_rps: f64,
    average_latency_ms: f64,
    p95_latency_ms: f64,
    p99_latency_ms: f64,
    average_queue_ms: f64,
    p95_queue_ms: f64,
    average_call_ms: f64,
    p95_call_ms: f64,
    average_processing_ms: f64,
    p95_processing_ms: f64,
    average_tool_ms: f64,
    p95_tool_ms: f64,
    average_llm_total_ms: f64,
    p95_llm_total_ms: f64,
    average_framework_overhead_ms: f64,
    p95_framework_overhead_ms: f64,
    average_framework_overhead_corrected_ms: f64,
    p95_framework_overhead_corrected_ms: f64,
    total_success: usize,
    total_failure: usize,
    cpu_usage_percent: f64,
    memory_peak_mb: f64,
    determinism_rate: f64,
}

#[derive(Debug)]
struct TimingBreakdown {
    total: Duration,
    queue_wait: Duration,
    call: Duration,
    status: bool,
}

#[derive(Copy, Clone, Debug)]
enum BenchmarkMode {
    Tool,
    LlmOnly,
}

struct PerformanceMonitor {
    start: Instant,
    start_cpu_seconds: f64,
    peak_rss_bytes: Arc<AtomicU64>,
    stop_flag: Arc<AtomicBool>,
    sampler: Option<JoinHandle<()>>,
}

impl PerformanceMonitor {
    fn start() -> Result<Self> {
        let peak_rss_bytes = Arc::new(AtomicU64::new(current_rss_bytes().unwrap_or(0)));
        let stop_flag = Arc::new(AtomicBool::new(false));
        let sampler = spawn_rss_sampler(stop_flag.clone(), peak_rss_bytes.clone());
        Ok(Self {
            start: Instant::now(),
            start_cpu_seconds: cpu_seconds()?,
            peak_rss_bytes,
            stop_flag,
            sampler,
        })
    }

    fn stop(mut self) -> Result<PerformanceStats> {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.sampler.take() {
            let _ = handle.join();
        }
        let wall_time = self.start.elapsed().as_secs_f64();
        let end_cpu = cpu_seconds()?;
        let cpu_time = (end_cpu - self.start_cpu_seconds).max(0.0);
        let cpu_usage_percent = if wall_time > 0.0 {
            (cpu_time / wall_time) * 100.0
        } else {
            0.0
        };
        let memory_peak_mb = if cfg!(target_os = "linux") {
            self.peak_rss_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
        } else {
            peak_rss_mb()?
        };
        Ok(PerformanceStats {
            cpu_usage_percent,
            memory_peak_mb,
        })
    }
}

struct PerformanceStats {
    cpu_usage_percent: f64,
    memory_peak_mb: f64,
}

fn cpu_seconds() -> Result<f64> {
    let usage = get_rusage()?;
    let user = usage.ru_utime;
    let sys = usage.ru_stime;
    let user_secs = user.tv_sec as f64 + (user.tv_usec as f64 / 1_000_000.0);
    let sys_secs = sys.tv_sec as f64 + (sys.tv_usec as f64 / 1_000_000.0);
    Ok(user_secs + sys_secs)
}

fn peak_rss_mb() -> Result<f64> {
    let usage = get_rusage()?;
    let rss = usage.ru_maxrss as f64;
    #[cfg(target_os = "macos")]
    let mb = rss / (1024.0 * 1024.0);
    #[cfg(not(target_os = "macos"))]
    let mb = rss / 1024.0;
    Ok(mb)
}

#[cfg(target_os = "linux")]
fn spawn_rss_sampler(stop_flag: Arc<AtomicBool>, peak: Arc<AtomicU64>) -> Option<JoinHandle<()>> {
    Some(std::thread::spawn(move || {
        while !stop_flag.load(Ordering::Relaxed) {
            if let Some(rss) = current_rss_bytes() {
                let mut prev = peak.load(Ordering::Relaxed);
                while rss > prev {
                    match peak.compare_exchange(prev, rss, Ordering::Relaxed, Ordering::Relaxed) {
                        Ok(_) => break,
                        Err(next) => prev = next,
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }))
}

#[cfg(not(target_os = "linux"))]
fn spawn_rss_sampler(_stop_flag: Arc<AtomicBool>, _peak: Arc<AtomicU64>) -> Option<JoinHandle<()>> {
    None
}

#[cfg(target_os = "linux")]
fn current_rss_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let mut parts = statm.split_whitespace();
    let _size = parts.next()?;
    let rss_pages: u64 = parts.next()?.parse().ok()?;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    Some(rss_pages * page_size)
}

#[cfg(not(target_os = "linux"))]
fn current_rss_bytes() -> Option<u64> {
    None
}

fn get_rusage() -> Result<libc::rusage> {
    let mut usage = MaybeUninit::<libc::rusage>::uninit();
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return Err(anyhow!("getrusage failed with code {}", result));
    }
    Ok(unsafe { usage.assume_init() })
}

fn compute_expected_value() -> Result<f64> {
    let path = std::env::var("TRIP_DATA_PATH").unwrap_or_else(|_| "trip_data.parquet".to_string());
    let aggregate = compute_average_trip_duration(Path::new(&path))?;
    Ok((aggregate.average_trip_duration_minutes * 100.0).round() / 100.0)
}

fn build_llm_only_prompt(request_id: usize, expected_value: f64) -> String {
    format!(
        "Request {request_id}. The average trip duration in minutes is {expected_value:.2}. \
Return JSON only in the format {{\"value\": {expected_value:.2}}}."
    )
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    async_main().await
}

async fn async_main() -> Result<()> {
    // CLI argument parsing.
    // Usage:
    //   cargo run                          -> run all frameworks, both modes
    //   cargo run -- autoagents            -> run only AutoAgents, both modes
    //   cargo run -- rig                   -> run only Rig, both modes
    //   cargo run -- all --mode tool       -> run all frameworks, tool mode only
    //   cargo run -- autoagents --mode llm -> run AutoAgents, llm-only mode
    //   cargo run -- rig --mode both       -> run Rig, both modes (explicit)
    let args: Vec<String> = std::env::args().collect();

    let framework = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "all".to_string())
        .to_lowercase();

    // Parse optional --mode <tool|llm|both> flag anywhere after arg[1]
    let mode_arg = args
        .windows(2)
        .find(|w| w[0] == "--mode")
        .map(|w| w[1].to_lowercase());

    let (run_llm, run_tool) = match mode_arg.as_deref() {
        Some("tool") => (false, true),
        Some("llm") => (true, false),
        Some("both") | None => (true, true),
        Some(other) => {
            return Err(anyhow!(
                "Unknown --mode '{}'. Valid options: tool, llm, both",
                other
            ));
        }
    };

    let run_autoagents = framework == "all" || framework == "autoagents";
    let run_rig = framework == "all" || framework == "rig";

    if !run_autoagents && !run_rig {
        return Err(anyhow!(
            "Unknown framework '{}'. Valid options: autoagents, rig, all",
            framework
        ));
    }

    let config_path = std::env::var("BENCH_CONFIG").unwrap_or_else(|_| "benchmark.yaml".into());
    let config = load_config(Path::new(&config_path))
        .with_context(|| format!("failed to load benchmark config from {config_path}"))?;

    if config.concurrency == 0 || config.total_requests == 0 {
        return Err(anyhow!(
            "Benchmark requires at least one request and concurrency > 0"
        ));
    }

    let expected_value = tokio::task::spawn_blocking(compute_expected_value)
        .await
        .context("failed to join expected value task")?
        .context("failed to compute expected value")?;

    println!(
        "Preparing benchmark: {} requests with concurrency {} (framework: {})",
        config.total_requests, config.concurrency, framework
    );

    if run_autoagents {
        println!("\n=== AutoAgents Results ===");

        let autoagents_llm = if run_llm {
            let r = bench_autoagents(&config, expected_value, BenchmarkMode::LlmOnly).await?;
            r.print("llm");
            persist_result(&r, "benchmark_results_llm.json")
                .context("failed to persist autoagents llm results")?;
            Some(r)
        } else {
            None
        };

        if run_tool {
            let mut autoagents_tool =
                bench_autoagents(&config, expected_value, BenchmarkMode::Tool).await?;
            if let Some(ref llm) = autoagents_llm {
                apply_overhead_metrics(&mut autoagents_tool, llm);
            }
            autoagents_tool.print("tool");
            persist_result(&autoagents_tool, "benchmark_results_tool.json")
                .context("failed to persist autoagents tool results")?;
        }
    }

    if run_rig {
        println!("\n=== Rig Results ===");

        let rig_llm = if run_llm {
            let r = bench_rig(&config, expected_value, BenchmarkMode::LlmOnly).await?;
            r.print("llm");
            persist_result(&r, "benchmark_results_llm.json")
                .context("failed to persist rig llm results")?;
            Some(r)
        } else {
            None
        };

        if run_tool {
            let mut rig_tool = bench_rig(&config, expected_value, BenchmarkMode::Tool).await?;
            if let Some(ref llm) = rig_llm {
                apply_overhead_metrics(&mut rig_tool, llm);
            }
            rig_tool.print("tool");
            persist_result(&rig_tool, "benchmark_results_tool.json")
                .context("failed to persist rig tool results")?;
        }
    }

    Ok(())
}

fn request_timeout(config: &BenchmarkConfig) -> Option<Duration> {
    if config.request_timeout_secs == 0 {
        None
    } else {
        Some(Duration::from_secs(config.request_timeout_secs))
    }
}

async fn bench_autoagents(
    config: &BenchmarkConfig,
    expected_value: f64,
    mode: BenchmarkMode,
) -> Result<BenchmarkResult> {
    let setup_start = Instant::now();
    let api_key = std::env::var("OPENAI_API_KEY")
        .context("OPENAI_API_KEY environment variable is required for AutoAgents benchmark")?;

    let mut llm_builder = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model(config.model.clone());
    if matches!(mode, BenchmarkMode::Tool) {
        llm_builder = llm_builder.tool_choice(ToolChoice::Auto);
    }
    let llm = llm_builder
        .build()
        .context("failed to build AutoAgents LLM client")?;
    let llm_timings = Arc::new(LlmTimingRecorder::default());
    let llm: Arc<dyn LLMProvider> = llm;
    let timed_llm: Arc<dyn LLMProvider> = Arc::new(TimedLlm::new(llm, llm_timings.clone()));

    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(config.concurrency));
    let request_timeout = request_timeout(config);
    let (breakdowns, elapsed, perf, setup_duration_s) = match mode {
        BenchmarkMode::Tool => {
            let agent_handle = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(SimpleAgent {}))
                .llm(timed_llm.clone())
                .memory(Box::new(TaskLocalMemory::default()))
                .build()
                .await?;
            let agent = Arc::new(agent_handle.agent);
            let setup_duration_s = setup_start.elapsed().as_secs_f64();
            reset_tool_timings();
            llm_timings.reset();

            let monitor =
                PerformanceMonitor::start().context("failed to start performance monitor")?;
            let start = Instant::now();
            let mut breakdowns = Vec::with_capacity(config.total_requests);
            let mut tasks = FuturesUnordered::new();
            for req_id in 0..config.total_requests {
                let permit = semaphore.clone();
                let agent = Arc::clone(&agent);
                let prompt = build_prompt(&config.prompt_template, req_id);
                let request_memory: Arc<TokioMutex<Box<dyn MemoryProvider>>> =
                    Arc::new(TokioMutex::new(
                        Box::new(SlidingWindowMemory::new(10)) as Box<dyn MemoryProvider>
                    ));

                tasks.push(tokio::spawn(async move {
                    let task = Task::new(&prompt);
                    let submitted = Instant::now();
                    let _permit = permit
                        .acquire_owned()
                        .await
                        .expect("semaphore closed unexpectedly");
                    let dequeued = Instant::now();
                    let queue_wait = dequeued.duration_since(submitted);

                    let call_start = Instant::now();
                    let call_future = REQUEST_ID.scope(req_id, async {
                        REQUEST_MEMORY
                            .scope(request_memory, async { agent.run(task).await })
                            .await
                    });
                    let result: Result<SimpleAgentOutput> = if let Some(timeout) = request_timeout {
                        match tokio::time::timeout(timeout, call_future).await {
                            Ok(inner) => inner.map_err(|e| anyhow!(e)),
                            Err(_) => Err(anyhow!(
                                "request {req_id} timed out after {}s",
                                timeout.as_secs()
                            )),
                        }
                    } else {
                        call_future.await.map_err(|e| anyhow!(e))
                    };

                    let call_duration = call_start.elapsed();
                    let total_duration = submitted.elapsed();

                    let status = match result {
                        Ok(result) => {
                            let rounded = (result.value * 100.0).round() / 100.0;
                            let success = (rounded - expected_value).abs() <= 0.01;
                            success
                        }
                        Err(e) => {
                            println!("ERROR: {e}");
                            false
                        }
                    };

                    TimingBreakdown {
                        total: total_duration,
                        queue_wait,
                        call: call_duration,
                        status,
                    }
                }));
            }

            while let Some(next) = tasks.next().await {
                match next {
                    Ok(inner) => breakdowns.push(inner),
                    Err(join_err) => return Err(anyhow!("Task join error: {}", join_err)),
                }
            }

            let elapsed = start.elapsed();
            let perf = monitor
                .stop()
                .context("failed to stop performance monitor")?;
            (breakdowns, elapsed, perf, setup_duration_s)
        }
        BenchmarkMode::LlmOnly => {
            let output_schema = SimpleAgentOutput::structured_output_format();
            let output_schema: StructuredOutputFormat = serde_json::from_value(output_schema)
                .context("failed to build structured output schema for llm-only")?;
            let setup_duration_s = setup_start.elapsed().as_secs_f64();
            llm_timings.reset();

            let monitor =
                PerformanceMonitor::start().context("failed to start performance monitor")?;
            let start = Instant::now();
            let mut breakdowns = Vec::with_capacity(config.total_requests);
            let mut tasks = FuturesUnordered::new();
            for req_id in 0..config.total_requests {
                let permit = semaphore.clone();
                let llm = timed_llm.clone();
                let prompt = build_llm_only_prompt(req_id, expected_value);
                let schema = output_schema.clone();

                tasks.push(tokio::spawn(async move {
                    let submitted = Instant::now();
                    let _permit = permit
                        .acquire_owned()
                        .await
                        .expect("semaphore closed unexpectedly");
                    let dequeued = Instant::now();
                    let queue_wait = dequeued.duration_since(submitted);

                    let call_start = Instant::now();
                    let call_future = REQUEST_ID.scope(req_id, async {
                        let messages = vec![ChatMessage::user().content(prompt).build()];
                        llm.chat(&messages, Some(schema)).await
                    });
                    let result: Result<Box<dyn ChatResponse>> =
                        if let Some(timeout) = request_timeout {
                            match tokio::time::timeout(timeout, call_future).await {
                                Ok(inner) => inner.map_err(|e| anyhow!(e)),
                                Err(_) => Err(anyhow!(
                                    "request {req_id} timed out after {}s",
                                    timeout.as_secs()
                                )),
                            }
                        } else {
                            call_future.await.map_err(|e| anyhow!(e))
                        };

                    let call_duration = call_start.elapsed();
                    let total_duration = submitted.elapsed();

                    let status = match result {
                        Ok(response) => response
                            .text()
                            .and_then(|text| parse_value(&text))
                            .map(|value| {
                                let rounded = (value * 100.0).round() / 100.0;
                                (rounded - expected_value).abs() <= 0.01
                            })
                            .unwrap_or(false),
                        Err(_) => false,
                    };

                    TimingBreakdown {
                        total: total_duration,
                        queue_wait,
                        call: call_duration,
                        status,
                    }
                }));
            }

            while let Some(next) = tasks.next().await {
                match next {
                    Ok(inner) => breakdowns.push(inner),
                    Err(join_err) => return Err(anyhow!("Task join error: {}", join_err)),
                }
            }

            let elapsed = start.elapsed();
            let perf = monitor
                .stop()
                .context("failed to stop performance monitor")?;
            (breakdowns, elapsed, perf, setup_duration_s)
        }
    };
    let mut tool_durations = match mode {
        BenchmarkMode::Tool => snapshot_tool_timings(),
        BenchmarkMode::LlmOnly => Vec::new(),
    };
    tool_durations.sort_unstable();
    let tool_divisor = tool_durations.len().max(1) as f64;
    let avg_tool_ms = if tool_durations.is_empty() {
        0.0
    } else {
        tool_durations.iter().map(|d| d.as_secs_f64()).sum::<f64>() / tool_divisor * 1_000.0
    };
    let p95_tool_ms = if tool_durations.is_empty() {
        0.0
    } else {
        percentile_ms(&tool_durations, 0.95)
    };
    let mut llm_request_totals = llm_timings.snapshot_request_totals();
    llm_request_totals.sort_unstable();
    let llm_total_divisor = llm_request_totals.len().max(1) as f64;
    let avg_llm_total_ms = if llm_request_totals.is_empty() {
        0.0
    } else {
        llm_request_totals
            .iter()
            .map(|d| d.as_secs_f64())
            .sum::<f64>()
            / llm_total_divisor
            * 1_000.0
    };
    let p95_llm_total_ms = if llm_request_totals.is_empty() {
        0.0
    } else {
        percentile_ms(&llm_request_totals, 0.95)
    };
    let mut result = summarize(
        "AutoAgents",
        config,
        breakdowns,
        elapsed,
        setup_duration_s,
        perf,
    );
    result.average_tool_ms = avg_tool_ms;
    result.p95_tool_ms = p95_tool_ms;
    let llm_totals_available = !llm_request_totals.is_empty();
    result.average_llm_total_ms = if llm_totals_available {
        avg_llm_total_ms
    } else {
        result.average_call_ms
    };
    result.p95_llm_total_ms = if llm_totals_available {
        p95_llm_total_ms
    } else {
        result.p95_call_ms
    };
    result.average_framework_overhead_ms = 0.0;
    result.p95_framework_overhead_ms = 0.0;
    result.average_framework_overhead_corrected_ms = 0.0;
    result.p95_framework_overhead_corrected_ms = 0.0;
    Ok(result)
}

fn load_config(path: &Path) -> Result<BenchmarkConfig> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read config file {}", path.display()))?;
    let config: BenchmarkConfig = serde_yaml::from_str(&raw)
        .with_context(|| format!("failed to parse YAML config at {}", path.display()))?;
    Ok(config)
}

fn build_prompt(template: &str, request_id: usize) -> String {
    format!(
        "{}\nReturn JSON only in the format {{\"value\": <number>}}.",
        template.replace("{i}", &request_id.to_string())
    )
}

fn summarize(
    name: &'static str,
    config: &BenchmarkConfig,
    breakdowns: Vec<TimingBreakdown>,
    elapsed: Duration,
    setup_duration_s: f64,
    perf: PerformanceStats,
) -> BenchmarkResult {
    let total_requests = breakdowns.len();
    if total_requests == 0 {
        return BenchmarkResult {
            name,
            total_requests: 0,
            concurrency: config.concurrency,
            total_duration: elapsed,
            setup_duration_s,
            throughput_rps: 0.0,
            throughput_success_rps: 0.0,
            average_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            average_queue_ms: 0.0,
            p95_queue_ms: 0.0,
            average_call_ms: 0.0,
            p95_call_ms: 0.0,
            average_processing_ms: 0.0,
            p95_processing_ms: 0.0,
            average_tool_ms: 0.0,
            p95_tool_ms: 0.0,
            average_llm_total_ms: 0.0,
            p95_llm_total_ms: 0.0,
            average_framework_overhead_ms: 0.0,
            p95_framework_overhead_ms: 0.0,
            average_framework_overhead_corrected_ms: 0.0,
            p95_framework_overhead_corrected_ms: 0.0,
            total_success: 0,
            total_failure: 0,
            cpu_usage_percent: perf.cpu_usage_percent,
            memory_peak_mb: perf.memory_peak_mb,
            determinism_rate: 0.0,
        };
    }

    let total_secs = elapsed.as_secs_f64().max(f64::EPSILON);
    let throughput_rps = total_requests as f64 / total_secs;

    let mut total_latencies: Vec<Duration> = breakdowns.iter().map(|b| b.total).collect();
    let mut queue_waits: Vec<Duration> = breakdowns.iter().map(|b| b.queue_wait).collect();
    let mut call_latencies: Vec<Duration> = breakdowns.iter().map(|b| b.call).collect();
    let mut processing_latencies: Vec<Duration> = breakdowns
        .iter()
        .map(|b| b.total.saturating_sub(b.call))
        .collect();

    total_latencies.sort_unstable();
    queue_waits.sort_unstable();
    call_latencies.sort_unstable();
    processing_latencies.sort_unstable();

    let divisor = total_requests as f64;
    let sum_totals = total_latencies.iter().map(|d| d.as_secs_f64()).sum::<f64>();
    let sum_queue = queue_waits.iter().map(|d| d.as_secs_f64()).sum::<f64>();
    let sum_call = call_latencies.iter().map(|d| d.as_secs_f64()).sum::<f64>();
    let sum_processing = processing_latencies
        .iter()
        .map(|d| d.as_secs_f64())
        .sum::<f64>();

    let total_success = breakdowns.iter().filter(|b| b.status == true).count();
    let total_failure = total_requests - total_success;
    let determinism_rate = total_success as f64 / total_requests as f64;
    let throughput_success_rps = total_success as f64 / total_secs;

    BenchmarkResult {
        name,
        total_requests,
        concurrency: config.concurrency,
        total_duration: elapsed,
        setup_duration_s,
        throughput_rps,
        throughput_success_rps,
        average_latency_ms: (sum_totals / divisor) * 1_000.0,
        p95_latency_ms: percentile_ms(&total_latencies, 0.95),
        p99_latency_ms: percentile_ms(&total_latencies, 0.99),
        average_queue_ms: (sum_queue / divisor) * 1_000.0,
        p95_queue_ms: percentile_ms(&queue_waits, 0.95),
        average_call_ms: (sum_call / divisor) * 1_000.0,
        p95_call_ms: percentile_ms(&call_latencies, 0.95),
        average_processing_ms: (sum_processing / divisor) * 1_000.0,
        p95_processing_ms: percentile_ms(&processing_latencies, 0.95),
        average_tool_ms: 0.0,
        p95_tool_ms: 0.0,
        average_llm_total_ms: 0.0,
        p95_llm_total_ms: 0.0,
        average_framework_overhead_ms: 0.0,
        p95_framework_overhead_ms: 0.0,
        average_framework_overhead_corrected_ms: 0.0,
        p95_framework_overhead_corrected_ms: 0.0,
        total_success,
        total_failure,
        cpu_usage_percent: perf.cpu_usage_percent,
        memory_peak_mb: perf.memory_peak_mb,
        determinism_rate,
    }
}

fn percentile_ms(durations: &[Duration], percentile: f64) -> f64 {
    if durations.is_empty() {
        return 0.0;
    }

    let capped = percentile.clamp(0.0, 1.0);
    let rank = (capped * durations.len() as f64).ceil().max(1.0) as usize - 1;
    let idx = rank.min(durations.len() - 1);
    durations[idx].as_secs_f64() * 1_000.0
}

fn persist_result(result: &BenchmarkResult, output_path: &str) -> Result<()> {
    let path = Path::new(output_path);

    let mut data = fs::read_to_string(path)
        .ok()
        .and_then(|content| serde_json::from_str::<Map<String, Value>>(&content).ok())
        .unwrap_or_default();

    data.insert(
        result.name.to_string(),
        json!({
            "name": result.name,
            "total_requests": result.total_requests,
            "concurrency": result.concurrency,
            "total_duration": result.total_duration.as_secs_f64(),
            "setup_duration_s": result.setup_duration_s,
            "throughput_rps": result.throughput_rps,
            "throughput_success_rps": result.throughput_success_rps,
            "average_latency_ms": result.average_latency_ms,
            "p95_latency_ms": result.p95_latency_ms,
            "p99_latency_ms": result.p99_latency_ms,
            "cold_start_ms": result.setup_duration_s * 1000.0,
            "average_queue_ms": result.average_queue_ms,
            "p95_queue_ms": result.p95_queue_ms,
            "average_call_ms": result.average_call_ms,
            "p95_call_ms": result.p95_call_ms,
            "average_processing_ms": result.average_processing_ms,
            "p95_processing_ms": result.p95_processing_ms,
            "average_tool_ms": result.average_tool_ms,
            "p95_tool_ms": result.p95_tool_ms,
            "average_llm_total_ms": result.average_llm_total_ms,
            "p95_llm_total_ms": result.p95_llm_total_ms,
            "average_framework_overhead_ms": result.average_framework_overhead_ms,
            "p95_framework_overhead_ms": result.p95_framework_overhead_ms,
            "average_framework_overhead_corrected_ms": result.average_framework_overhead_corrected_ms,
            "p95_framework_overhead_corrected_ms": result.p95_framework_overhead_corrected_ms,
            "total_success": result.total_success,
            "total_failure": result.total_failure,
            "cpu_usage_percent": result.cpu_usage_percent,
            "memory_peak_mb": result.memory_peak_mb,
            "determinism_rate": result.determinism_rate,
        }),
    );

    let output_value = Value::Object(data);
    let serialized = serde_json::to_string_pretty(&output_value)
        .context("failed to serialise benchmark results to JSON")?;
    fs::write(path, serialized)
        .with_context(|| format!("failed to write benchmark results to {}", path.display()))?;

    Ok(())
}

impl BenchmarkResult {
    fn print(&self, mode: &str) {
        println!("--- {} ({}) ---", self.name, mode);
        println!("requests      : {}", self.total_requests);
        println!("concurrency   : {}", self.concurrency);
        println!("setup time    : {:.3} s", self.setup_duration_s);
        println!("total time    : {:.3} s", self.total_duration.as_secs_f64());
        println!("throughput    : {:.2} req/s", self.throughput_rps);
        println!("throughput ok : {:.2} req/s", self.throughput_success_rps);
        println!("cold start    : {:.2} ms", self.setup_duration_s * 1000.0);
        println!("avg latency   : {:.2} ms", self.average_latency_ms);
        println!("p95 latency   : {:.2} ms", self.p95_latency_ms);
        println!("p99 latency   : {:.2} ms", self.p99_latency_ms);
        println!(
            "queue wait    : avg {:.2} ms | p95 {:.2} ms",
            self.average_queue_ms, self.p95_queue_ms
        );
        println!(
            "llm latency   : avg {:.2} ms | p95 {:.2} ms",
            self.average_call_ms, self.p95_call_ms
        );
        println!(
            "framework ovh : avg {:.2} ms | p95 {:.2} ms",
            self.average_processing_ms, self.p95_processing_ms
        );
        println!(
            "tool exec     : avg {:.2} ms | p95 {:.2} ms",
            self.average_tool_ms, self.p95_tool_ms
        );
        println!(
            "llm total     : avg {:.2} ms | p95 {:.2} ms",
            self.average_llm_total_ms, self.p95_llm_total_ms
        );
        println!(
            "framework ovh : avg {:.2} ms | p95 {:.2} ms (tool+llm subtracted)",
            self.average_framework_overhead_ms, self.p95_framework_overhead_ms
        );
        println!(
            "framework ovh : avg {:.2} ms | p95 {:.2} ms (llm_total+tool subtracted)",
            self.average_framework_overhead_corrected_ms, self.p95_framework_overhead_corrected_ms
        );
        println!("cpu usage    : {:.2} %", self.cpu_usage_percent);
        println!("peak memory  : {:.2} MB", self.memory_peak_mb);
        println!("determinism  : {:.2} %", self.determinism_rate * 100.0);
        println!("Total Success: {}", self.total_success);
        println!("Total Failure: {}", self.total_failure);
    }
}

fn apply_overhead_metrics(tool_result: &mut BenchmarkResult, llm_result: &BenchmarkResult) {
    tool_result.average_framework_overhead_ms =
        (tool_result.average_call_ms - llm_result.average_call_ms - tool_result.average_tool_ms)
            .max(0.0);
    tool_result.p95_framework_overhead_ms =
        (tool_result.p95_call_ms - llm_result.p95_call_ms - tool_result.p95_tool_ms).max(0.0);
    tool_result.average_framework_overhead_corrected_ms = (tool_result.average_call_ms
        - tool_result.average_llm_total_ms
        - tool_result.average_tool_ms)
        .max(0.0);
    tool_result.p95_framework_overhead_corrected_ms =
        (tool_result.p95_call_ms - tool_result.p95_llm_total_ms - tool_result.p95_tool_ms).max(0.0);
}

// =============================================================================
// Rig benchmark
// =============================================================================
//
// Note on LLM timing: rig 0.31's `CompletionModel` trait requires implementing
// streaming, client, and other provider-level components, making a transparent
// timing wrapper impractical.  We therefore set `avg_llm_total_ms = avg_call_ms`
// (i.e. the full agent.prompt() wall-clock time is attributed to LLM time).
// The `apply_overhead_metrics()` call still correctly derives basic framework
// overhead by comparing tool-mode vs llm-only-mode call latencies.

/// Args for the rig trip-data tool (no parameters needed).
#[derive(Deserialize)]
struct RigTripToolArgs {}

/// Error type for the rig trip-data tool.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct RigTripToolError(String);

/// Rig `Tool` implementation that calls the TLC parquet trip-duration computation.
struct RigTripTool;

impl rig::tool::Tool for RigTripTool {
    const NAME: &'static str = "trip_data_average_duration";

    type Error = RigTripToolError;
    type Args = RigTripToolArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> rig::completion::ToolDefinition {
        rig::completion::ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Compute the average TLC trip duration in minutes from the \
                          trip_data.parquet dataset."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        // ToolTimingGuard::start() records elapsed time on drop (end of this fn).
        let _timing = ToolTimingGuard::start();
        let path =
            std::env::var("TRIP_DATA_PATH").unwrap_or_else(|_| "trip_data.parquet".to_string());
        let agg =
            tokio::task::spawn_blocking(move || compute_average_trip_duration(Path::new(&path)))
                .await
                .map_err(|e| RigTripToolError(e.to_string()))?
                .map_err(|e| RigTripToolError(e.to_string()))?;
        Ok(format!(
            "Average trip duration is {:.2} minutes across {} trips.",
            agg.average_trip_duration_minutes, agg.row_count
        ))
    }
}

async fn bench_rig(
    config: &BenchmarkConfig,
    expected_value: f64,
    mode: BenchmarkMode,
) -> Result<BenchmarkResult> {
    use rig::client::{CompletionClient as _, ProviderClient as _};
    use rig::completion::Prompt as _;

    let setup_start = Instant::now();

    // Validate the key is present before trying to use it.
    std::env::var("OPENAI_API_KEY")
        .context("OPENAI_API_KEY environment variable is required for Rig benchmark")?;

    // Use the Chat Completions API (CompletionsClient) rather than the default
    // Responses API client. The Responses API client deserialises the
    // `service_tier` field as a strict enum that doesn't include the
    // `"priority"` variant OpenAI now returns, causing a JsonError at runtime.
    let openai = rig::providers::openai::CompletionsClient::from_env();

    println!(
        "Preparing Rig benchmark ({:?}): {} requests with concurrency {}",
        mode, config.total_requests, config.concurrency
    );

    let semaphore = Arc::new(tokio::sync::Semaphore::new(config.concurrency));
    let request_timeout = request_timeout(config);

    match mode {
        BenchmarkMode::Tool => {
            // AgentBuilder::new(model).tool(T) transitions to AgentBuilderSimple.
            let agent = Arc::new(
                openai
                    .agent(&config.model)
                    .preamble(
                         "You are a helpful assistant, provide the answer to the user's question in structured format"
                    )
                    .tool(RigTripTool)
                    .output_schema::<SimpleAgentOutput>()
                    .build(),
            );

            let setup_duration_s = setup_start.elapsed().as_secs_f64();
            reset_tool_timings();

            let monitor =
                PerformanceMonitor::start().context("failed to start rig tool monitor")?;
            let start = Instant::now();
            let mut tasks = FuturesUnordered::new();

            for req_id in 0..config.total_requests {
                let permit = semaphore.clone();
                let agent = Arc::clone(&agent);
                let prompt = build_prompt(&config.prompt_template, req_id);

                tasks.push(tokio::spawn(async move {
                    let submitted = Instant::now();
                    let _permit = permit.acquire_owned().await.expect("semaphore closed");
                    let dequeued = Instant::now();
                    let queue_wait = dequeued.duration_since(submitted);

                    let call_start = Instant::now();
                    let call_future =
                        REQUEST_ID.scope(req_id, async { agent.prompt(prompt.as_str()).await });
                    let result: Result<String> = if let Some(timeout) = request_timeout {
                        match tokio::time::timeout(timeout, call_future).await {
                            Ok(inner) => inner.map_err(|e| anyhow!(e)),
                            Err(_) => Err(anyhow!("request {req_id} timed out")),
                        }
                    } else {
                        call_future.await.map_err(|e| anyhow!(e))
                    };

                    let call_duration = call_start.elapsed();
                    let total_duration = submitted.elapsed();

                    let status = match result {
                        Ok(ref text) => parse_value(text)
                            .map(|v| {
                                let rounded = (v * 100.0).round() / 100.0;
                                (rounded - expected_value).abs() <= 0.01
                            })
                            .unwrap_or(false),
                        Err(ref e) => {
                            println!("RIG TOOL ERROR: {e}");
                            false
                        }
                    };

                    TimingBreakdown {
                        total: total_duration,
                        queue_wait,
                        call: call_duration,
                        status,
                    }
                }));
            }

            let mut breakdowns = Vec::with_capacity(config.total_requests);
            while let Some(next) = tasks.next().await {
                match next {
                    Ok(inner) => breakdowns.push(inner),
                    Err(e) => return Err(anyhow!("Rig task join error: {e}")),
                }
            }

            let elapsed = start.elapsed();
            let perf = monitor.stop().context("failed to stop rig tool monitor")?;

            let mut tool_durations = snapshot_tool_timings();
            tool_durations.sort_unstable();
            let tool_divisor = tool_durations.len().max(1) as f64;
            let avg_tool_ms = if tool_durations.is_empty() {
                0.0
            } else {
                tool_durations.iter().map(|d| d.as_secs_f64()).sum::<f64>() / tool_divisor * 1_000.0
            };
            let p95_tool_ms = if tool_durations.is_empty() {
                0.0
            } else {
                percentile_ms(&tool_durations, 0.95)
            };

            let mut result = summarize("Rig", config, breakdowns, elapsed, setup_duration_s, perf);
            result.average_tool_ms = avg_tool_ms;
            result.p95_tool_ms = p95_tool_ms;
            // LLM timing approximation: full agent.prompt() wall-clock ≈ LLM time.
            result.average_llm_total_ms = result.average_call_ms;
            result.p95_llm_total_ms = result.p95_call_ms;
            result.average_framework_overhead_ms = 0.0;
            result.p95_framework_overhead_ms = 0.0;
            result.average_framework_overhead_corrected_ms = 0.0;
            result.p95_framework_overhead_corrected_ms = 0.0;
            Ok(result)
        }

        BenchmarkMode::LlmOnly => {
            let agent = Arc::new(
                openai
                    .agent(&config.model)
                    .preamble(
                        "You are a helpful assistant. \
                         Return JSON only in the format {\"value\": <float>}.",
                    )
                    .build(),
            );

            let setup_duration_s = setup_start.elapsed().as_secs_f64();

            let monitor = PerformanceMonitor::start().context("failed to start rig llm monitor")?;
            let start = Instant::now();
            let mut tasks = FuturesUnordered::new();

            for req_id in 0..config.total_requests {
                let permit = semaphore.clone();
                let agent = Arc::clone(&agent);
                let prompt = build_llm_only_prompt(req_id, expected_value);

                tasks.push(tokio::spawn(async move {
                    let submitted = Instant::now();
                    let _permit = permit.acquire_owned().await.expect("semaphore closed");
                    let dequeued = Instant::now();
                    let queue_wait = dequeued.duration_since(submitted);

                    let call_start = Instant::now();
                    let call_future =
                        REQUEST_ID.scope(req_id, async { agent.prompt(prompt.as_str()).await });
                    let result: Result<String> = if let Some(timeout) = request_timeout {
                        match tokio::time::timeout(timeout, call_future).await {
                            Ok(inner) => inner.map_err(|e| anyhow!(e)),
                            Err(_) => Err(anyhow!("request {req_id} timed out")),
                        }
                    } else {
                        call_future.await.map_err(|e| anyhow!(e))
                    };

                    let call_duration = call_start.elapsed();
                    let total_duration = submitted.elapsed();

                    let status = match result {
                        Ok(ref text) => parse_value(text)
                            .map(|v| {
                                let rounded = (v * 100.0).round() / 100.0;
                                (rounded - expected_value).abs() <= 0.01
                            })
                            .unwrap_or(false),
                        Err(ref e) => {
                            println!("RIG LLM ERROR: {e}");
                            false
                        }
                    };

                    TimingBreakdown {
                        total: total_duration,
                        queue_wait,
                        call: call_duration,
                        status,
                    }
                }));
            }

            let mut breakdowns = Vec::with_capacity(config.total_requests);
            while let Some(next) = tasks.next().await {
                match next {
                    Ok(inner) => breakdowns.push(inner),
                    Err(e) => return Err(anyhow!("Rig task join error: {e}")),
                }
            }

            let elapsed = start.elapsed();
            let perf = monitor.stop().context("failed to stop rig llm monitor")?;

            let mut result = summarize("Rig", config, breakdowns, elapsed, setup_duration_s, perf);
            result.average_tool_ms = 0.0;
            result.p95_tool_ms = 0.0;
            // LLM timing approximation: full agent.prompt() wall-clock ≈ LLM time.
            result.average_llm_total_ms = result.average_call_ms;
            result.p95_llm_total_ms = result.p95_call_ms;
            result.average_framework_overhead_ms = 0.0;
            result.p95_framework_overhead_ms = 0.0;
            result.average_framework_overhead_corrected_ms = 0.0;
            result.p95_framework_overhead_corrected_ms = 0.0;
            Ok(result)
        }
    }
}
