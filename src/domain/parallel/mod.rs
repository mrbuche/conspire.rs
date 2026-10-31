//! What parallel assembly shares between the frameworks.

#[cfg(test)]
mod test;

use crate::domain::block::element::Items;
use rayon::prelude::*;

const GATHER_CHUNK: usize = 512;

/// TEMPORARY EXPERIMENT: the threads of one run of explicit dynamics.
///
/// This is the only type that names the thread pool, so replacing it replaces the backend.
pub struct Workers(rayon::ThreadPool);

impl Workers {
    /// Starts the given number of threads, which stop when the workers are dropped.
    pub fn new(threads: usize) -> Result<Self, String> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map(Self)
            .map_err(|error| error.to_string())
    }
    pub(crate) fn install<R: Send>(&self, operation: impl FnOnce() -> R + Send) -> R {
        self.0.install(operation)
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
        workers.install(|| {
            targets[..self.number_of_nodes()]
                .par_chunks_mut(GATHER_CHUNK)
                .enumerate()
                .for_each(|(chunk, targets)| {
                    targets.iter_mut().enumerate().for_each(|(index, target)| {
                        self.contributions(chunk * GATHER_CHUNK + index)
                            .iter()
                            .for_each(|&(item, local)| add(target, item, local))
                    })
                })
        })
    }
}
