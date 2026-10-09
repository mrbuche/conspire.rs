#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Mesh, Partition},
    math::Scalar,
};
use std::{collections::BTreeSet, mem::take};

/// What a union of elements is worth, for deciding which elements to join.
pub trait Criterion {
    /// A positive score for each element, where higher is better.
    fn score(&self, elements: &[usize]) -> Result<Scalar, String>;
    /// Whether the elements are good enough to be left alone.
    fn certified(&self, elements: &[usize]) -> Result<bool, String>;
    /// Whether the union of the elements is allowed.
    fn valid(&self, elements: &[usize]) -> bool;
}

/// An element that is not certified is joined to the adjacent element with the best score for
/// the union, if that is at least `minimum_improvement` times the smaller of the two scores.
pub struct Merging {
    pub minimum_improvement: Scalar,
    pub passes: usize,
}

pub struct Merged {
    pub elements_parts: Vec<usize>,
    pub scores: Vec<Scalar>,
    pub unresolved: Vec<usize>,
}

impl Merged {
    pub fn partition(&self, mesh: &Mesh<3>) -> Partition {
        Partition::new(mesh, self.elements_parts.clone())
    }
    pub fn mesh(&self, mesh: &Mesh<3>) -> Result<Mesh<3>, &'static str> {
        self.partition(mesh).agglomerate(mesh)
    }
}

impl Merging {
    /// Join the elements, given which are adjacent and the score of each one on its own.
    ///
    /// Each pass visits the elements that are not certified, worst first. A union made in a pass
    /// is not visited again until the next pass, and merging stops once a pass changes nothing.
    pub fn agglomerate(
        &self,
        adjacent: &[Vec<usize>],
        mut scores: Vec<Scalar>,
        criterion: &impl Criterion,
    ) -> Result<Merged, String> {
        let number_of_elements = adjacent.len();
        assert_eq!(scores.len(), number_of_elements, "one score per element");
        let mut groups = (0..number_of_elements)
            .map(|element| vec![element])
            .collect::<Vec<_>>();
        let mut owner = (0..number_of_elements).collect::<Vec<_>>();
        let mut alive = vec![true; number_of_elements];
        for _ in 0..self.passes {
            let mut seeds = unresolved(criterion, &groups, &alive)?;
            seeds.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
            let mut merged = false;
            for seed in seeds {
                if !alive[seed] {
                    continue;
                }
                let neighbors = groups[seed]
                    .iter()
                    .flat_map(|&element| adjacent[element].iter().map(|&other| owner[other]))
                    .filter(|&group| group != seed)
                    .collect::<BTreeSet<_>>();
                let candidates = neighbors
                    .into_iter()
                    .filter_map(|neighbor| {
                        let union = [groups[seed].as_slice(), groups[neighbor].as_slice()].concat();
                        criterion
                            .valid(&union)
                            .then(|| criterion.score(&union).map(|score| (neighbor, score)))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if let Some((neighbor, score)) =
                    candidates.into_iter().max_by(|a, b| a.1.total_cmp(&b.1))
                    && score > self.minimum_improvement * scores[seed].min(scores[neighbor])
                {
                    let absorbed = take(&mut groups[neighbor]);
                    absorbed.iter().for_each(|&element| owner[element] = seed);
                    groups[seed].extend(absorbed);
                    alive[neighbor] = false;
                    scores[seed] = score;
                    merged = true
                }
            }
            if !merged {
                break;
            }
        }
        let unresolved = unresolved(criterion, &groups, &alive)?;
        let mut parts = vec![usize::MAX; number_of_elements];
        let mut part_scores = Vec::new();
        (0..number_of_elements)
            .filter(|&group| alive[group])
            .for_each(|group| {
                parts[group] = part_scores.len();
                part_scores.push(scores[group])
            });
        Ok(Merged {
            elements_parts: owner.iter().map(|&group| parts[group]).collect(),
            scores: part_scores,
            unresolved: unresolved.iter().map(|&group| parts[group]).collect(),
        })
    }
}

fn unresolved(
    criterion: &impl Criterion,
    groups: &[Vec<usize>],
    alive: &[bool],
) -> Result<Vec<usize>, String> {
    (0..groups.len())
        .filter(|&group| alive[group])
        .filter_map(|group| {
            criterion
                .certified(&groups[group])
                .map(|certified| (!certified).then_some(group))
                .transpose()
        })
        .collect()
}
