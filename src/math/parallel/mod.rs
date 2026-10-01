#[cfg(test)]
mod test;

use std::{
    sync::atomic::{AtomicUsize, Ordering},
    thread::{available_parallelism, scope},
};

pub(crate) fn thread_count(requested: usize) -> usize {
    requested.clamp(
        1,
        available_parallelism().map_or(1, |threads| threads.get()),
    )
}

pub(crate) fn parallel_map_init<T, S, R>(
    items: &[T],
    threads: usize,
    init: impl Fn() -> S + Sync,
    f: impl Fn(&mut S, &T) -> R + Sync,
) -> Vec<R>
where
    T: Sync,
    R: Send,
{
    let threads = thread_count(threads).min(items.len());
    if threads <= 1 {
        let mut state = init();
        return items.iter().map(|item| f(&mut state, item)).collect();
    }
    let next = AtomicUsize::new(0);
    let mut results: Vec<(usize, R)> = scope(|scope| {
        (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut state = init();
                    let mut done = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= items.len() {
                            break done;
                        }
                        done.push((index, f(&mut state, &items[index])));
                    }
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flat_map(|handle| handle.join().expect("worker thread panicked"))
            .collect()
    });
    results.sort_unstable_by_key(|&(index, _)| index);
    results.into_iter().map(|(_, result)| result).collect()
}
