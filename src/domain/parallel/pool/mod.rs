//! Threads that stay alive between the jobs they are given.
//!
//! This is the only place in `parallel` that is `unsafe`. A job borrows the data of its caller, so
//! its lifetime is erased to hand it to threads that outlive the call. That is sound because
//! [`Pool::run`] does not return, not even by unwinding, until every thread is done with the job.

#[cfg(test)]
mod test;

use alloc::sync::Arc;
use core::mem::transmute;
use core::panic::AssertUnwindSafe;
use std::{
    panic::catch_unwind,
    sync::{Condvar, Mutex, MutexGuard, PoisonError},
    thread::{Builder, JoinHandle},
};

type Job = &'static (dyn Fn() + Sync);

#[derive(Default)]
struct State {
    job: Option<Job>,
    generation: u64,
    remaining: usize,
    panicked: bool,
    shutdown: bool,
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    done: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The calling thread and the threads that it started, which run a job together.
pub struct Pool {
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
    busy: Mutex<()>,
}

impl Pool {
    /// A pool of the given number of threads, which includes the one that calls [`run`](Self::run).
    pub fn new(threads: usize) -> Result<Self, String> {
        if threads == 0 {
            return Err("There must be at least one thread.".to_string());
        }
        let mut pool = Self {
            shared: Arc::default(),
            handles: Vec::with_capacity(threads - 1),
            busy: Mutex::new(()),
        };
        for index in 1..threads {
            let shared = pool.shared.clone();
            let handle = Builder::new()
                .name(format!("conspire-worker-{index}"))
                .spawn(move || worker(&shared))
                .map_err(|error| format!("Could not start a thread: {error}."))?;
            pool.handles.push(handle);
        }
        Ok(pool)
    }
    /// Calls `job` once on every thread of the pool and returns when all of them are done.
    ///
    /// A job must not call `run` of the same pool, which would wait on itself.
    pub fn run(&self, job: &(dyn Fn() + Sync)) {
        if self.handles.is_empty() {
            return job();
        }
        let _busy = self.busy.lock().unwrap_or_else(PoisonError::into_inner);
        // SAFETY: `Finish` waits until every thread has finished the job, also when `job` panics
        // on this thread, and takes the job back, so that nothing uses it after this call.
        let job: Job = unsafe { transmute::<&(dyn Fn() + Sync), Job>(job) };
        {
            let mut state = self.shared.lock();
            state.job = Some(job);
            state.generation += 1;
            state.remaining = self.handles.len();
            state.panicked = false;
        }
        self.shared.wake.notify_all();
        let finish = Finish(&self.shared);
        job();
        drop(finish);
        assert!(
            !self.shared.lock().panicked,
            "A thread of the pool panicked."
        );
    }
}

/// Waits for the threads of a job and takes the job back, even when unwinding.
struct Finish<'a>(&'a Shared);

impl Drop for Finish<'_> {
    fn drop(&mut self) {
        let mut state = self.0.lock();
        while state.remaining > 0 {
            state = self
                .0
                .done
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.job = None;
    }
}

fn worker(shared: &Shared) {
    let mut seen = 0;
    loop {
        let job = {
            let mut state = shared.lock();
            loop {
                if state.shutdown {
                    return;
                }
                if state.generation != seen {
                    seen = state.generation;
                    break state.job.expect("a job for a new generation");
                }
                state = shared
                    .wake
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        };
        let result = catch_unwind(AssertUnwindSafe(job));
        let mut state = shared.lock();
        state.panicked |= result.is_err();
        state.remaining -= 1;
        if state.remaining == 0 {
            shared.done.notify_one();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.lock().shutdown = true;
        self.shared.wake.notify_all();
        self.handles.drain(..).for_each(|handle| {
            let _ = handle.join();
        });
    }
}
