use super::parallel_map;
use std::{
    collections::HashSet,
    sync::Mutex,
    thread::{available_parallelism, current, sleep},
    time::Duration,
};

#[test]
fn parallel_map_keeps_item_order_under_uneven_work() {
    let items: Vec<usize> = (0..53).collect();
    let mapped = parallel_map(&items, 4, |&item| {
        if item % 7 == 0 {
            sleep(Duration::from_millis(3));
        }
        item * item
    });
    assert_eq!(mapped, items.iter().map(|&i| i * i).collect::<Vec<_>>());
}

#[test]
fn parallel_map_uses_no_more_than_the_thread_cap() {
    let items: Vec<usize> = (0..64).collect();
    let threads = Mutex::new(HashSet::new());
    let max_threads = 4;
    parallel_map(&items, max_threads, |_| {
        threads.lock().unwrap().insert(current().id());
        sleep(Duration::from_millis(2));
    });
    let used = threads.lock().unwrap().len();
    assert!(used <= max_threads, "used {used} threads");
    if available_parallelism().map_or(1, |n| n.get()) > 1 {
        assert!(used > 1, "never left the calling thread");
    }
}
