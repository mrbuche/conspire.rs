#[cfg(test)]
mod test;

use crate::math::FxHashMap;
use std::cell::OnceCell;

/// External labels for a collection of items, one label for each item.
pub struct Labels {
    labels: Vec<usize>,
    indices: OnceCell<FxHashMap<usize, usize>>,
}

impl Labels {
    /// The labels of a collection of `number_of_items` items.
    ///
    /// # Panics
    /// Panics if there is not exactly one label for each item.
    pub fn new(labels: Vec<usize>, number_of_items: usize) -> Self {
        assert_eq!(
            labels.len(),
            number_of_items,
            "There must be exactly one label for each item."
        );
        Self {
            labels,
            indices: OnceCell::new(),
        }
    }
    /// The label of each item.
    pub fn as_slice(&self) -> &[usize] {
        &self.labels
    }
    /// The index of the first item carrying a label.
    pub fn index_of(&self, label: usize) -> Option<usize> {
        self.indices
            .get_or_init(|| {
                let mut indices = FxHashMap::default();
                self.labels.iter().enumerate().for_each(|(index, &label)| {
                    indices.entry(label).or_insert(index);
                });
                indices
            })
            .get(&label)
            .copied()
    }
}
