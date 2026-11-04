use super::{Criterion, Merging};
use std::cell::Cell;

struct Line {
    bad: Vec<usize>,
    needed: usize,
    maximum: usize,
    certifications: Cell<usize>,
}

impl Line {
    fn new(bad: Vec<usize>, needed: usize, maximum: usize) -> Self {
        Self {
            bad,
            needed,
            maximum,
            certifications: Cell::new(0),
        }
    }
}

impl Criterion for Line {
    fn score(&self, elements: &[usize]) -> Result<f64, String> {
        Ok(elements.len() as f64)
    }
    fn certified(&self, elements: &[usize]) -> Result<bool, String> {
        self.certifications.set(self.certifications.get() + 1);
        Ok(elements.len() >= self.needed || !elements.iter().any(|e| self.bad.contains(e)))
    }
    fn valid(&self, elements: &[usize]) -> bool {
        elements.len() <= self.maximum
    }
}

fn line(number_of_elements: usize) -> Vec<Vec<usize>> {
    (0..number_of_elements)
        .map(|element| {
            [element.checked_sub(1), Some(element + 1)]
                .into_iter()
                .flatten()
                .filter(|&other| other < number_of_elements)
                .collect()
        })
        .collect()
}

fn merging(passes: usize) -> Merging {
    Merging {
        minimum_improvement: 1.0,
        passes,
    }
}

#[test]
fn the_worst_element_is_joined_first() {
    let line_ = line(3);
    let criterion = Line::new(vec![0, 2], 2, 2);
    let first = merging(5)
        .agglomerate(&line_, vec![0.5, 2.0, 1.0], &criterion)
        .unwrap();
    assert_eq!(first.elements_parts, [0, 0, 1]);
    assert_eq!(first.unresolved, [1]);
    let second = merging(5)
        .agglomerate(&line_, vec![1.0, 2.0, 0.5], &criterion)
        .unwrap();
    assert_eq!(second.elements_parts, [0, 1, 1]);
    assert_eq!(second.unresolved, [0]);
}

#[test]
fn a_union_is_only_visited_in_the_next_pass() {
    let line_ = line(4);
    let one = merging(1)
        .agglomerate(&line_, vec![1.0; 4], &Line::new(vec![0], 3, 4))
        .unwrap();
    assert_eq!(one.elements_parts, [0, 0, 1, 2]);
    assert_eq!(one.unresolved, [0]);
    let two = merging(2)
        .agglomerate(&line_, vec![1.0; 4], &Line::new(vec![0], 3, 4))
        .unwrap();
    assert_eq!(two.elements_parts, [0, 0, 0, 1]);
    assert!(two.unresolved.is_empty());
    assert_eq!(two.scores, [3.0, 1.0]);
}

#[test]
fn merging_stops_when_a_pass_changes_nothing() {
    let criterion = Line::new(vec![], 1, 4);
    let result = merging(5)
        .agglomerate(&line(3), vec![1.0; 3], &criterion)
        .unwrap();
    assert_eq!(result.elements_parts, [0, 1, 2]);
    assert_eq!(criterion.certifications.get(), 6);
}

#[test]
fn no_passes_change_nothing() {
    let result = merging(0)
        .agglomerate(&line(3), vec![1.0; 3], &Line::new(vec![0], 3, 4))
        .unwrap();
    assert_eq!(result.elements_parts, [0, 1, 2]);
    assert_eq!(result.unresolved, [0]);
}

#[test]
fn a_union_that_is_not_allowed_is_left_alone() {
    let result = merging(5)
        .agglomerate(&line(3), vec![1.0; 3], &Line::new(vec![0], 3, 1))
        .unwrap();
    assert_eq!(result.elements_parts, [0, 1, 2]);
    assert_eq!(result.unresolved, [0]);
}

#[test]
fn a_union_must_improve_enough() {
    let result = Merging {
        minimum_improvement: 10.0,
        passes: 5,
    }
    .agglomerate(&line(3), vec![1.0; 3], &Line::new(vec![0], 3, 4))
    .unwrap();
    assert_eq!(result.elements_parts, [0, 1, 2]);
    assert_eq!(result.unresolved, [0]);
}

#[test]
fn an_element_with_no_neighbors_stays_alone() {
    let result = merging(5)
        .agglomerate(&[vec![]], vec![1.0], &Line::new(vec![0], 3, 4))
        .unwrap();
    assert_eq!(result.elements_parts, [0]);
    assert_eq!(result.unresolved, [0]);
}

#[test]
#[should_panic(expected = "one score per element")]
fn there_is_a_score_for_each_element() {
    merging(1)
        .agglomerate(&line(3), vec![1.0; 2], &Line::new(vec![], 1, 1))
        .unwrap();
}
