use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::oneshot;

use crate::engine::{Decided, Engine};
use crate::types::SystemOneRequest;

struct Job {
    request: SystemOneRequest,
    reply: oneshot::Sender<Result<Decided>>,
}

#[derive(Clone)]
pub struct Batcher {
    tx: Sender<Job>,
}

impl Batcher {
    pub fn spawn(engine: Arc<dyn Engine>) -> Self {
        let (tx, rx) = mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("clef-engine".into())
            .spawn(move || run(engine, rx))
            .expect("spawning the engine thread");
        std::mem::drop(handle);
        Batcher { tx }
    }

    pub async fn submit(&self, request: SystemOneRequest) -> Result<Decided> {
        let (reply, answer) = oneshot::channel();
        self.tx
            .send(Job { request, reply })
            .map_err(|_| anyhow::anyhow!("the engine thread has stopped"))?;
        answer
            .await
            .map_err(|_| anyhow::anyhow!("the engine thread dropped this request without answering"))?
    }
}

fn run(engine: Arc<dyn Engine>, rx: Receiver<Job>) {
    let cap = crate::engine::max_batch().max(1);

    let mut in_flight: Vec<(usize, Job)> = Vec::new();

    let mut deferred: std::collections::VecDeque<Job> = std::collections::VecDeque::new();
    loop {
        if in_flight.is_empty() {
            engine.on_idle();
        }

        let mut jobs: Vec<Job> = Vec::with_capacity(cap);

        while jobs.len() < cap {
            match deferred.pop_front() {
                Some(j) => jobs.push(j),
                None => break,
            }
        }

        let idle = in_flight.is_empty() && jobs.is_empty();
        if !idle {
            while jobs.len() < cap {
                match rx.try_recv() {
                    Ok(job) => jobs.push(job),
                    Err(_) => break,
                }
            }
        } else {
            match rx.recv() {
                Ok(job) => jobs.push(job),
                Err(_) => return,
            }
            while jobs.len() < cap {
                match rx.try_recv() {
                    Ok(job) => jobs.push(job),
                    Err(_) => break,
                }
            }
        }

        for job in jobs {
            if in_flight.len() < engine.long_slots() {
                match engine.long_begin(&job.request) {
                    Ok(slot) => in_flight.push((slot, job)),
                    Err(e) => {
                        let _ = job.reply.send(Err(e));
                    }
                }
            } else {
                deferred.push_back(job);
            }
        }

        if !in_flight.is_empty() {
            let slots: Vec<usize> = in_flight.iter().map(|(s, _)| *s).collect();

            let t_turn = std::time::Instant::now();
            let advanced = engine.long_advance_many(&slots);
            if std::env::var("CLEF_SERVE_TRACE").is_ok() {
                eprintln!("    serve: long_advance_many({}) in {:>8.1} ms",
                          slots.len(), t_turn.elapsed().as_secs_f64() * 1e3);
            }
            match advanced {
                Ok(dones) => {
                    let mut still = Vec::with_capacity(in_flight.len());
                    for ((slot, job), done) in in_flight.drain(..).zip(dones) {
                        if done {
                            let _ = job.reply.send(engine.long_finish(slot));
                        } else {
                            still.push((slot, job));
                        }
                    }
                    in_flight = still;

                    if !in_flight.is_empty() {
                        in_flight.rotate_left(1);
                    }
                }
                Err(e) => {
                    for (slot, job) in in_flight.drain(..) {
                        let _ = job.reply.send(Err(anyhow::anyhow!("{e}")));
                        let _ = slot;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Health, Question, QuestionType};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    struct Recorder {
        batches: Mutex<Vec<usize>>,
        gate: Mutex<Option<Receiver<()>>>,
        calls: AtomicUsize,
        order: Mutex<Vec<String>>,
        labels: Mutex<Vec<String>>,
        states: Mutex<Vec<String>>,
        long_passes: Mutex<Vec<usize>>,
        long_gate: Mutex<Option<Receiver<()>>>,
        slots: usize,
        next_slot: AtomicUsize,
        pass_widths: Mutex<Vec<usize>>,
    }

    impl Recorder {
        fn new(gate: Option<Receiver<()>>) -> Self {
            Recorder { batches: Mutex::new(Vec::new()), gate: Mutex::new(gate),
                       calls: AtomicUsize::new(0), order: Mutex::new(Vec::new()),
                       long_passes: Mutex::new(vec![0; 8]), long_gate: Mutex::new(None),
                       slots: 1, next_slot: AtomicUsize::new(0),
                       pass_widths: Mutex::new(Vec::new()),
                       labels: Mutex::new(Vec::new()), states: Mutex::new(Vec::new()) }
        }
        fn batches(&self) -> Vec<usize> {
            self.batches.lock().unwrap().clone()
        }
        fn order(&self) -> Vec<String> {
            self.order.lock().unwrap().clone()
        }
    }

    impl Engine for Recorder {
        fn health(&self) -> Health {
            Health {
                ok: true,
                model: "recorder".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                device: "none".into(),
                backend: "none".into(),
                quant: "none".into(),
                checkpoint: "none".into(),
                max_batch: crate::engine::max_batch(),
                max_tokens: crate::engine::max_input_tokens(),
                graphs: 0,
                graphs_chunked: 0,
                kv_slots: 1,
                chunked_passes: 0,
                chunked_width_max: 0,
                chunked_passes_sharing: 0,
                graph_pool_mib: 0,
                captures_pending: 0,
                head_graphs: 0,
                chunk_tokens: crate::gpu::max_chunk_size(),
                att_keys: crate::cuda::att_keys(),
                stream_priority: 0,
                kv_tokens: 0,
                kv_mib: 0,
                refused_over_ceiling: 0,
            }
        }

        fn long_slots(&self) -> usize {
            self.slots
        }

        fn long_begin(&self, request: &SystemOneRequest) -> Result<usize> {
            if request.state == "fail" {
                anyhow::bail!("this request is invalid on its own");
            }
            let slot = self.next_slot.fetch_add(1, Ordering::SeqCst);
            let label = if request.state.starts_with("long") {
                let c = request.state.split_whitespace().nth(1).and_then(|w| w.chars().next()).unwrap_or('x');
                format!("long_{c}")
            } else {
                "short".to_string()
            };
            let mut l = self.labels.lock().unwrap();
            while l.len() <= slot { l.push("none".to_string()); }
            l[slot] = label;
            let mut st = self.states.lock().unwrap();
            while st.len() <= slot { st.push(String::new()); }
            st[slot] = request.state.clone();
            Ok(slot)
        }

        fn long_advance_many(&self, slots: &[usize]) -> Result<Vec<bool>> {
            self.batches.lock().unwrap().push(slots.len());
            self.pass_widths.lock().unwrap().push(slots.len());
            let mut out = Vec::with_capacity(slots.len());
            for &slot in slots {
                out.push(self.long_advance(slot)?);
            }
            Ok(out)
        }

        fn long_advance(&self, slot: usize) -> Result<bool> {
            let label = self.labels.lock().unwrap().get(slot).cloned().unwrap_or_else(|| "none".into());
            self.order.lock().unwrap().push(label.clone());

            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                if let Some(gate) = self.gate.lock().unwrap().take() {
                    let _ = gate.recv();
                }
            }

            if let Some(gate) = self.long_gate.lock().unwrap().take() {
                let _ = gate.recv();
            }
            let mut counts = self.long_passes.lock().unwrap();
            counts[slot.min(7)] += 1;

            let need = if label == "short" { 1 } else { 3 };
            Ok(counts[slot.min(7)] >= need)
        }

        fn long_finish(&self, slot: usize) -> Result<Decided> {
            let state = self.states.lock().unwrap().get(slot).cloned().unwrap_or_default();
            Ok(Decided {
                probabilities: vec![(state.clone(), std::collections::BTreeMap::new())],
                input_tokens: state.len(),
            })
        }

        fn decide(&self, _request: &SystemOneRequest) -> Result<Decided> {
            anyhow::bail!("the batcher must not call decide — one builder serves every request")
        }
    }

    fn request(text: &str) -> SystemOneRequest {
        SystemOneRequest {
            state: text.into(),
            questions: vec![(
                "q".to_string(),
                Question { kind: QuestionType::Noul, instructions: None, criteria: None },
            )],
            images: None,
            videos: None,
            model: None,
        }
    }

    #[tokio::test]
    async fn two_long_requests_are_served_in_turn() {
        let (open, gate) = mpsc::channel();
        let mut engine = Recorder::new(None);
        engine.slots = 2;
        engine.long_gate = Mutex::new(Some(gate));
        let engine = Arc::new(engine);
        let batcher = Batcher::spawn(engine.clone());
        let a = { let b = batcher.clone(); tokio::spawn(async move { b.submit(request("long aaa")).await }) };
        while engine.order().is_empty() {
            tokio::task::yield_now().await;
        }

        let b = { let b = batcher.clone(); tokio::spawn(async move { b.submit(request("long bbb")).await }) };
        tokio::task::yield_now().await;
        open.send(()).unwrap();
        let (ra, rb) = (a.await.unwrap(), b.await.unwrap());
        assert!(ra.is_ok() && rb.is_ok(), "both answered");

        let order = engine.order();
        eprintln!("  order = {order:?}");
        let na = order.iter().filter(|o| o.as_str() == "long_a").count();
        let nb = order.iter().filter(|o| o.as_str() == "long_b").count();
        assert!(na >= 3 && nb >= 3, "both requests ran their passes: {order:?}");

        let first_b = order.iter().position(|o| *o == "long_b").expect("a pass of B");
        let last_a = order.iter().rposition(|o| *o == "long_a").expect("a pass of A");
        assert!(first_b < last_a, "the second long request waited for the first: {order:?}");
    }

    #[tokio::test]
    async fn two_long_requests_share_one_pass() {
        let (open, gate) = mpsc::channel();
        let mut engine = Recorder::new(None);
        engine.slots = 2;
        engine.long_gate = Mutex::new(Some(gate));
        let engine = Arc::new(engine);
        let batcher = Batcher::spawn(engine.clone());
        let a = { let b = batcher.clone(); tokio::spawn(async move { b.submit(request("long aaa")).await }) };
        while engine.order().is_empty() {
            tokio::task::yield_now().await;
        }
        let b = { let b = batcher.clone(); tokio::spawn(async move { b.submit(request("long bbb")).await }) };
        tokio::task::yield_now().await;
        open.send(()).unwrap();
        let (ra, rb) = (a.await.unwrap(), b.await.unwrap());
        assert!(ra.is_ok() && rb.is_ok(), "both answered");

        let widths = engine.pass_widths.lock().unwrap().clone();
        eprintln!("  pass widths = {widths:?}");
        assert!(widths.contains(&2), "no pass ever carried both requests: {widths:?}");

        assert!(widths.len() <= 4, "the shared pass did not reduce the pass count: {widths:?}");
    }

    #[tokio::test]
    async fn a_deferred_long_request_is_not_stranded() {
        let (open, gate) = mpsc::channel();
        let mut engine = Recorder::new(None);
        engine.slots = 1;
        engine.long_gate = Mutex::new(Some(gate));
        let engine = Arc::new(engine);
        let batcher = Batcher::spawn(engine.clone());

        let a = { let b = batcher.clone(); tokio::spawn(async move { b.submit(request("long aaa")).await }) };
        while engine.order().is_empty() {
            tokio::task::yield_now().await;
        }
        let (reply_b, answer_b) = oneshot::channel();
        batcher.tx.send(Job { request: request("long bbb"), reply: reply_b })
            .expect("the engine thread is alive");
        open.send(()).unwrap();
        let both = tokio::time::timeout(std::time::Duration::from_secs(20), async {
            (a.await.unwrap(), answer_b.await)
        })
        .await
        .expect("both long requests must be answered — the deferred one was stranded");
        assert!(both.0.is_ok(), "the first long request answered: {:?}", both.0.err());
        assert!(both.1.is_ok(), "the DEFERRED long request answered: {:?}", both.1.err());
        let widths = engine.pass_widths.lock().unwrap().clone();
        assert!(widths.iter().all(|w| *w == 1), "one slot cannot carry two: {widths:?}");
        eprintln!("  deferred request answered; pass widths = {widths:?}");
    }

    #[tokio::test]
    async fn a_long_request_is_served_a_pass_at_a_time() {
        let (open, gate) = mpsc::channel();
        let mut engine = Recorder::new(None);

        engine.slots = 2;
        engine.long_gate = Mutex::new(Some(gate));
        let engine = Arc::new(engine);
        let batcher = Batcher::spawn(engine.clone());

        let long = {
            let b = batcher.clone();
            tokio::spawn(async move { b.submit(request("long long long")).await })
        };

        while engine.order().is_empty() {
            tokio::task::yield_now().await;
        }
        let short = {
            let b = batcher.clone();
            tokio::spawn(async move { b.submit(request("short one")).await })
        };
        tokio::task::yield_now().await;
        open.send(()).unwrap();
        let (l, sh) = (long.await.unwrap(), short.await.unwrap());
        assert!(l.is_ok(), "the long request answered: {}", l.err().map(|e| e.to_string()).unwrap_or_default());
        assert!(sh.is_ok(), "the short request answered: {}", sh.err().map(|e| e.to_string()).unwrap_or_default());

        let order = engine.order();
        eprintln!("  order = {order:?}");

        let longs = order.iter().filter(|o| o.starts_with("long")).count();
        assert_eq!(longs, 3, "the long request needed three passes: {order:?}");

        let shorts = order.iter().filter(|o| o.as_str() == "short").count();
        assert_eq!(shorts, 1, "a short request must finish in ONE pass: {order:?}");

        let last_long = order.iter().rposition(|o| o.starts_with("long")).expect("a long pass");
        let first_short = order.iter().position(|o| o.as_str() == "short").expect("a short pass");
        assert!(
            first_short < last_long,
            "the short request waited for the long one to finish: {order:?}"
        );
    }

    #[tokio::test]
    async fn the_queue_accumulates_while_the_engine_is_busy() {
        let (open, gate) = mpsc::channel();

        let mut engine = Recorder::new(Some(gate));
        engine.slots = 3;
        let engine = Arc::new(engine);
        let batcher = Batcher::spawn(engine.clone());

        let first = {
            let b = batcher.clone();
            tokio::spawn(async move { b.submit(request("a")).await })
        };

        while engine.calls.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }

        let second = { let b = batcher.clone(); tokio::spawn(async move { b.submit(request("b")).await }) };
        let third = { let b = batcher.clone(); tokio::spawn(async move { b.submit(request("c")).await }) };
        tokio::task::yield_now().await;

        open.send(()).unwrap();
        let a = first.await.unwrap().unwrap();
        let b = second.await.unwrap().unwrap();
        let c = third.await.unwrap().unwrap();

        let widths = engine.batches();
        eprintln!("  pass widths = {widths:?}");
        assert_eq!(widths.first(), Some(&1), "the first pass carries one request: {widths:?}");
        assert_eq!(
            widths.iter().skip(1).copied().collect::<Vec<_>>(),
            vec![2],
            "the two queued requests must share ONE pass: {widths:?}"
        );

        assert_eq!(a.probabilities[0].0, "a");
        assert_eq!(b.probabilities[0].0, "b");
        assert_eq!(c.probabilities[0].0, "c");
    }

    #[tokio::test]
    async fn one_bad_request_does_not_fail_the_batch() {
        let engine = Arc::new(Recorder::new(None));
        let batcher = Batcher::spawn(engine.clone());
        let good = { let b = batcher.clone(); tokio::spawn(async move { b.submit(request("fine")).await }) };
        let bad = { let b = batcher.clone(); tokio::spawn(async move { b.submit(request("fail")).await }) };
        let (g, f) = (good.await.unwrap(), bad.await.unwrap());
        assert!(g.is_ok(), "the neighbour of a failing request must still be answered");
        assert!(f.is_err(), "and the failing request must report its own error");
        assert_eq!(g.unwrap().probabilities[0].0, "fine");
    }

    #[tokio::test]
    async fn a_stopped_engine_thread_fails_the_request() {
        let (tx, rx) = mpsc::channel::<Job>();
        drop(rx);
        let batcher = Batcher { tx };
        let err = match batcher.submit(request("x")).await {
            Ok(_) => panic!("a stopped engine thread must not answer"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("engine thread has stopped"), "got: {err}");
    }

    #[tokio::test]
    async fn a_caller_that_goes_away_does_not_stop_the_service() {
        let engine = Arc::new(Recorder::new(None));
        let batcher = Batcher::spawn(engine.clone());
        {
            let mut pending = Box::pin(batcher.submit(request("gone")));
            let _ = futures_poll_once(&mut pending);
        }

        let ok = batcher.submit(request("after")).await.unwrap();
        assert_eq!(ok.probabilities[0].0, "after");
    }

    fn futures_poll_once<F: std::future::Future + Unpin>(f: &mut F) -> Option<F::Output> {
        use std::task::{Context, Poll, Waker};
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        match std::pin::Pin::new(f).poll(&mut cx) {
            Poll::Ready(v) => Some(v),
            Poll::Pending => None,
        }
    }
}
