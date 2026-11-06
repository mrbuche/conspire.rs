//! What parallel assembly shares between the frameworks.

mod pool;
#[cfg(test)]
mod test;

use crate::domain::block::element::Items;
use core::convert::Infallible;
use pool::Pool;
use std::sync::Mutex;

const GATHER_CHUNK: usize = 512;

/// The threads that the passes of a run of explicit dynamics are computed on.
///
/// The threads are started once and wait between the passes. In a pass they take the chunks of
/// the items from one queue, so that a pass over items of unequal cost still balances. The
/// calling thread takes chunks too. This is the only type that names the threads, and they end
/// when it is dropped.
pub struct Workers {
    pool: Pool,
}

impl Workers {
    /// The workers of the given number of threads, at least one.
    pub fn new(threads: usize) -> Result<Self, String> {
        Pool::new(threads).map(|pool| Self { pool })
    }
    /// Calls `operation` with the position of the first item and the items of each chunk of
    /// `items`, which has at most `chunk` items.
    pub(crate) fn for_each_chunk<T>(
        &self,
        items: &mut [T],
        chunk: usize,
        operation: impl Fn(usize, &mut [T]) + Sync,
    ) where
        T: Send,
    {
        let _ = self.try_for_each_chunk(items, chunk, |first, items| {
            operation(first, items);
            Ok::<(), Infallible>(())
        });
    }
    /// Calls `operation` like [`for_each_chunk`](Self::for_each_chunk), and returns the error
    /// of the first chunk that has one, whichever thread gets to it first.
    pub(crate) fn try_for_each_chunk<T, E>(
        &self,
        items: &mut [T],
        chunk: usize,
        operation: impl Fn(usize, &mut [T]) -> Result<(), E> + Sync,
    ) -> Result<(), E>
    where
        T: Send,
        E: Send,
    {
        assert!(chunk > 0, "A chunk must have items.");
        let chunks = items.len().div_ceil(chunk);
        let queue = Mutex::new(items.chunks_mut(chunk).enumerate());
        let failure = Mutex::new(None::<(usize, E)>);
        let work = || {
            loop {
                let next = queue.lock().expect("the chunks").next();
                let Some((index, items)) = next else { break };
                if let Err(error) = operation(index * chunk, items) {
                    let mut failure = failure.lock().expect("the failure");
                    if failure.as_ref().is_none_or(|(first, _)| index < *first) {
                        *failure = Some((index, error));
                    }
                }
            }
        };
        if chunks > 1 {
            self.pool.run(&work)
        } else {
            work()
        }
        match failure.into_inner().expect("the failure") {
            Some((_, error)) => Err(error),
            None => Ok(()),
        }
    }
}

/// For each node, the items that contribute to it, and the position of the node among the nodes
/// of each.
///
/// It depends only on the nodes of the items, so it serves everything that is assembled over
/// them, and is built once for a run.
pub struct ItemIndex {
    offsets: Vec<usize>,
    contributions: Vec<(usize, usize)>,
}

impl ItemIndex {
    /// The index of the items, with the contributions to a node in the order of the items.
    pub fn new(items: &(impl Items + ?Sized)) -> Self {
        let number_of_items = items.number_of_items();
        let nodes = (0..number_of_items)
            .flat_map(|item| items.nodes_of(item))
            .max()
            .map_or(0, |&node| node + 1);
        let mut offsets = vec![0; nodes + 1];
        (0..number_of_items)
            .flat_map(|item| items.nodes_of(item))
            .for_each(|&node| offsets[node + 1] += 1);
        (0..nodes).for_each(|node| offsets[node + 1] += offsets[node]);
        let mut next = offsets.clone();
        let mut contributions = vec![(0, 0); offsets[nodes]];
        (0..number_of_items).for_each(|item| {
            items
                .nodes_of(item)
                .iter()
                .enumerate()
                .for_each(|(local, &node)| {
                    contributions[next[node]] = (item, local);
                    next[node] += 1;
                })
        });
        Self {
            offsets,
            contributions,
        }
    }
    /// The number of nodes up to the last node that an item has.
    pub fn number_of_nodes(&self) -> usize {
        self.offsets.len() - 1
    }
    /// The items that contribute to a node, and its position among the nodes of each.
    pub fn contributions(&self, node: usize) -> &[(usize, usize)] {
        &self.contributions[self.offsets[node]..self.offsets[node + 1]]
    }
    /// Calls `add` with the target of each node and each contribution to it, in the order of the
    /// items, so that a sum is that of adding the contributions item by item.
    ///
    /// Each target is only ever touched by one thread, so there is nothing to synchronize.
    pub fn gather<T>(
        &self,
        workers: &Workers,
        targets: &mut [T],
        add: impl Fn(&mut T, usize, usize) + Sync,
    ) where
        T: Send,
    {
        workers.for_each_chunk(
            &mut targets[..self.number_of_nodes()],
            GATHER_CHUNK,
            |first, targets| {
                targets.iter_mut().enumerate().for_each(|(index, target)| {
                    self.contributions(first + index)
                        .iter()
                        .for_each(|&(item, local)| add(target, item, local))
                })
            },
        )
    }
}
