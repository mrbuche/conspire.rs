use super::Pool;
use crate::math::FxHashSet;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::{panic::AssertUnwindSafe, time::Duration};
use std::{
    panic::catch_unwind,
    sync::Mutex,
    thread::{current, sleep, yield_now},
};

#[test]
fn needs_a_thread() {
    assert!(Pool::new(0).is_err());
}

#[test]
fn every_thread_runs_the_job_once() {
    let pool = Pool::new(4).unwrap();
    let count = AtomicUsize::new(0);
    pool.run(&|| {
        count.fetch_add(1, Ordering::SeqCst);
    });
    assert_eq!(count.load(Ordering::SeqCst), 4);
}

#[test]
fn the_threads_are_the_same_between_jobs() {
    let pool = Pool::new(3).unwrap();
    let ids = Mutex::new(FxHashSet::default());
    (0..20).for_each(|_| {
        pool.run(&|| {
            ids.lock().unwrap().insert(current().id());
        })
    });
    assert_eq!(ids.lock().unwrap().len(), 3);
}

#[test]
fn a_job_finishes_before_run_returns() {
    let pool = Pool::new(4).unwrap();
    (0..50).for_each(|_| {
        let count = AtomicUsize::new(0);
        pool.run(&|| {
            yield_now();
            count.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(count.load(Ordering::SeqCst), 4);
    });
}

#[test]
fn a_job_borrows_what_changes_between_jobs() {
    let pool = Pool::new(3).unwrap();
    (0..10).for_each(|round| {
        let sum = AtomicUsize::new(0);
        let data = [round; 8];
        pool.run(&|| {
            sum.fetch_add(data.iter().sum(), Ordering::SeqCst);
        });
        assert_eq!(sum.load(Ordering::SeqCst), 3 * 8 * round);
    });
}

#[test]
fn a_panic_of_a_thread_is_reported_and_the_pool_survives() {
    let pool = Pool::new(3).unwrap();
    let main = current().id();
    assert!(
        catch_unwind(AssertUnwindSafe(|| pool.run(&|| {
            if current().id() != main {
                panic!("a worker");
            }
        })))
        .is_err()
    );
    let count = AtomicUsize::new(0);
    pool.run(&|| {
        count.fetch_add(1, Ordering::SeqCst);
    });
    assert_eq!(count.load(Ordering::SeqCst), 3);
}

#[test]
fn a_panic_of_the_calling_thread_waits_for_the_others() {
    let pool = Pool::new(3).unwrap();
    let main = current().id();
    let done = AtomicUsize::new(0);
    assert!(
        catch_unwind(AssertUnwindSafe(|| pool.run(&|| {
            if current().id() == main {
                panic!("the caller");
            }
            sleep(Duration::from_millis(50));
            done.fetch_add(1, Ordering::SeqCst);
        })))
        .is_err()
    );
    assert_eq!(done.load(Ordering::SeqCst), 2);
}

#[test]
fn one_thread_runs_the_job_on_the_caller() {
    let pool = Pool::new(1).unwrap();
    let main = current().id();
    pool.run(&|| assert_eq!(current().id(), main));
}
