use super::{parallel_map_init, thread_count};
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    thread::{available_parallelism, sleep},
    time::Duration,
};

#[test]
fn thread_count_is_at_least_one_and_at_most_the_parallelism() {
    let available = available_parallelism().map_or(1, |threads| threads.get());
    assert_eq!(thread_count(0), 1);
    assert_eq!(thread_count(1), 1);
    assert_eq!(thread_count(usize::MAX), available);
}

#[test]
fn results_keep_the_order_of_the_items() {
    let items: Vec<usize> = (0..300).collect();
    let squares = parallel_map_init(
        &items,
        usize::MAX,
        || (),
        |_, &item| {
            sleep(Duration::from_micros(item as u64 % 7 * 40));
            item * item
        },
    );
    assert_eq!(
        squares,
        items.iter().map(|item| item * item).collect::<Vec<_>>()
    );
}

#[test]
fn threads_agree_with_the_serial_map() {
    let items: Vec<usize> = (0..257).collect();
    let work = |state: &mut usize, &item: &usize| {
        *state += 1;
        item * 3 + 1
    };
    assert_eq!(
        parallel_map_init(&items, 1, || 0, work),
        parallel_map_init(&items, usize::MAX, || 0, work)
    );
}

#[test]
fn state_is_built_once_per_thread() {
    let items: Vec<usize> = (0..500).collect();
    let built = AtomicUsize::new(0);
    parallel_map_init(
        &items,
        usize::MAX,
        || built.fetch_add(1, Ordering::Relaxed),
        |_, &item| item,
    );
    let built = built.into_inner();
    assert!((1..=thread_count(usize::MAX)).contains(&built), "{built}");
}

#[test]
fn nothing_to_map() {
    let items: Vec<usize> = Vec::new();
    assert!(parallel_map_init(&items, usize::MAX, || (), |_, &item| item).is_empty());
}
