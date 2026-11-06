use super::{ItemIndex, Workers};
use crate::domain::block::element::Items;

#[derive(Debug)]
struct Chain(Vec<Vec<usize>>);

impl Items for Chain {
    fn number_of_items(&self) -> usize {
        self.0.len()
    }
    fn nodes_of(&self, item: usize) -> &[usize] {
        &self.0[item]
    }
}

mod index {
    use super::*;
    #[test]
    fn lists_the_contributions_to_each_node_in_the_order_of_the_items() {
        let index = ItemIndex::new(&Chain(vec![vec![0, 1, 2], vec![2, 3], vec![1, 2]]));
        assert_eq!(index.number_of_nodes(), 4);
        assert_eq!(index.contributions(0), [(0, 0)]);
        assert_eq!(index.contributions(1), [(0, 1), (2, 0)]);
        assert_eq!(index.contributions(2), [(0, 2), (1, 0), (2, 1)]);
        assert_eq!(index.contributions(3), [(1, 1)]);
    }
    #[test]
    fn leaves_out_the_nodes_after_the_last_one_of_an_item() {
        let index = ItemIndex::new(&Chain(vec![vec![1, 2]]));
        assert_eq!(index.number_of_nodes(), 3);
        assert!(index.contributions(0).is_empty());
    }
    #[test]
    fn is_empty_without_items() {
        let index = ItemIndex::new(&Chain(vec![]));
        assert_eq!(index.number_of_nodes(), 0);
    }
}

mod gather {
    use super::*;
    fn long_chain() -> Chain {
        Chain((0..2000).map(|item| vec![item, item + 1]).collect())
    }
    fn gathered(threads: usize) -> Vec<Vec<(usize, usize)>> {
        let chain = long_chain();
        let index = ItemIndex::new(&chain);
        let mut targets = vec![Vec::new(); 2001];
        index.gather(
            &Workers::new(threads).unwrap(),
            &mut targets,
            |target, item, local| target.push((item, local)),
        );
        targets
    }
    #[test]
    fn visits_the_contributions_to_every_node_in_the_order_of_the_items() {
        let targets = gathered(3);
        assert_eq!(targets[0], [(0, 0)]);
        assert_eq!(targets[1000], [(999, 1), (1000, 0)]);
        assert_eq!(targets[2000], [(1999, 1)]);
        assert!(targets.iter().all(|target| target.len() <= 2));
    }
    #[test]
    fn does_not_depend_on_the_number_of_threads() {
        let one = gathered(1);
        assert_eq!(one, gathered(2));
        assert_eq!(one, gathered(4));
    }
    #[test]
    fn leaves_the_targets_beyond_the_last_node_alone() {
        let index = ItemIndex::new(&Chain(vec![vec![0, 1]]));
        let mut targets = vec![0; 5];
        index.gather(&Workers::new(2).unwrap(), &mut targets, |target, _, _| {
            *target += 1
        });
        assert_eq!(targets, [1, 1, 0, 0, 0]);
    }
}

mod workers {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};
    #[test]
    fn at_least_one_thread_is_needed() {
        assert!(Workers::new(0).is_err());
        assert!(Workers::new(1).is_ok());
    }
    #[test]
    fn every_item_is_visited_once_with_the_position_of_its_chunk() {
        [
            (0, 5),
            (1, 5),
            (7, 5),
            (100, 5),
            (100, 1),
            (100, 100),
            (100, 1000),
            (101, 10),
        ]
        .into_iter()
        .for_each(|(length, chunk)| {
            [1, 2, 4, 7].into_iter().for_each(|threads| {
                let mut items = vec![usize::MAX; length];
                Workers::new(threads)
                    .unwrap()
                    .for_each_chunk(&mut items, chunk, |first, items| {
                        assert!(items.len() <= chunk);
                        items.iter_mut().enumerate().for_each(|(offset, item)| {
                            assert_eq!(*item, usize::MAX, "visited twice");
                            *item = first + offset;
                        })
                    });
                assert_eq!(items, (0..length).collect::<Vec<_>>(), "{length}, {chunk}");
            })
        });
    }
    #[test]
    fn more_threads_than_chunks_all_the_same_visit_every_item() {
        let visits = AtomicUsize::new(0);
        let mut items = vec![0; 3];
        Workers::new(16)
            .unwrap()
            .for_each_chunk(&mut items, 2, |_, items| {
                visits.fetch_add(items.len(), Ordering::Relaxed);
            });
        assert_eq!(visits.into_inner(), 3);
    }
    #[test]
    fn the_error_is_that_of_the_first_chunk_that_has_one() {
        (0..20).for_each(|_| {
            let mut items = vec![0; 100];
            let result = Workers::new(4)
                .unwrap()
                .try_for_each_chunk(&mut items, 10, |first, _| match first {
                    30 | 50 => Err(first),
                    _ => Ok(()),
                });
            assert_eq!(result, Err(30));
        })
    }
    #[test]
    fn the_chunks_without_an_error_are_still_done() {
        let mut items = vec![0; 100];
        let _ = Workers::new(4)
            .unwrap()
            .try_for_each_chunk(&mut items, 10, |first, items| {
                items.iter_mut().for_each(|item| *item = 1);
                if first == 30 { Err(()) } else { Ok(()) }
            });
        assert_eq!(items.iter().sum::<i32>(), 100);
    }
}
