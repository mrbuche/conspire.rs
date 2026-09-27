#[cfg(test)]
mod test;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::{available_parallelism, scope};

pub(crate) fn thread_count(max_threads: usize) -> usize {
    max_threads.min(available_parallelism().map_or(1, |threads| threads.get()))
}

/// `items.iter().map(f).collect()` spread over up to `max_threads` threads.
/// Threads pull the next unclaimed item, not a fixed chunk, because
/// subdomains cost different amounts (an interior subdomain carries more dual
/// dofs than a corner one), and results come back in item order.
pub(crate) fn parallel_map<T, R>(
    items: &[T],
    max_threads: usize,
    f: impl Fn(&T) -> R + Sync,
) -> Vec<R>
where
    T: Sync,
    R: Send,
{
    let threads = thread_count(max_threads).min(items.len());
    if threads <= 1 {
        return items.iter().map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let mut results: Vec<(usize, R)> = scope(|scope| {
        (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= items.len() {
                            break done;
                        }
                        done.push((index, f(&items[index])));
                    }
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flat_map(|handle| handle.join().expect("setup thread panicked"))
            .collect()
    });
    results.sort_unstable_by_key(|&(index, _)| index);
    results.into_iter().map(|(_, result)| result).collect()
}
