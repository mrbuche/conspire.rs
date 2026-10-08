#[cfg(test)]
mod test;

use super::Candidates;
use crate::{
    geometry::mesh::{ElementsFaces, Mesh, Partition},
    math::{Quantity, Scalar},
    units::Time,
};
use std::{collections::BTreeSet, mem::take};

/// The time scale a typical element is expected to have.
pub enum Reference {
    /// The median of the time scales of the elements, taken one by one.
    Median,
    /// A time scale given directly, such as from a mesh known to be well shaped.
    Value(Quantity<Time>),
}

/// An element is too fast when its time scale is not certified to exceed the reference over
/// `step_reduction`. Such an element is joined to the neighbor that gives the largest time scale,
/// if that is at least `minimum_improvement` times the smaller of the two, and the join can be
/// one virtual element, see [`Candidates::check`].
pub struct Agglomeration {
    pub reference: Reference,
    pub step_reduction: Scalar,
    pub minimum_improvement: Scalar,
    pub minimum_volume: Scalar,
    pub passes: usize,
}

pub struct Agglomerated {
    pub reference: Quantity<Time>,
    pub elements_parts: Vec<usize>,
    pub time_scales: Vec<Quantity<Time>>,
    pub unresolved: Vec<usize>,
}

impl Agglomerated {
    pub fn partition(&self, mesh: &Mesh<3>) -> Partition {
        Partition::new(mesh, self.elements_parts.clone())
    }
    pub fn mesh(&self, mesh: &Mesh<3>) -> Result<Mesh<3>, &'static str> {
        self.partition(mesh).agglomerate(mesh)
    }
}

impl<S: ElementsFaces> Candidates<S> {
    pub fn agglomerate(&self, agglomeration: &Agglomeration) -> Result<Agglomerated, String> {
        let number_of_elements = self.number_of_elements();
        let mut scales = self.time_scales()?;
        let reference = match &agglomeration.reference {
            Reference::Median => {
                let mut values = scales.iter().map(|scale| scale.value()).collect::<Vec<_>>();
                values.sort_by(|a, b| a.total_cmp(b));
                Time::seconds(values[values.len() / 2])
            }
            Reference::Value(value) => *value,
        };
        let threshold = Time::seconds(reference.value() / agglomeration.step_reduction);
        let adjacent = self.boundary.adjacent();
        let mut groups = (0..number_of_elements)
            .map(|element| vec![element])
            .collect::<Vec<_>>();
        let mut owner = (0..number_of_elements).collect::<Vec<_>>();
        let mut alive = vec![true; number_of_elements];
        for _ in 0..agglomeration.passes {
            let mut seeds = self.unresolved(&groups, &alive, threshold)?;
            seeds.sort_by(|&a, &b| scales[a].value().total_cmp(&scales[b].value()));
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
                        self.check(&union, agglomeration.minimum_volume)
                            .ok()
                            .map(|()| self.time_scale(&union).map(|scale| (neighbor, scale)))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if let Some((neighbor, scale)) = candidates
                    .into_iter()
                    .max_by(|a, b| a.1.value().total_cmp(&b.1.value()))
                    && scale.value()
                        > agglomeration.minimum_improvement
                            * scales[seed].value().min(scales[neighbor].value())
                {
                    let absorbed = take(&mut groups[neighbor]);
                    absorbed.iter().for_each(|&element| owner[element] = seed);
                    groups[seed].extend(absorbed);
                    alive[neighbor] = false;
                    scales[seed] = scale;
                    merged = true
                }
            }
            if !merged {
                break;
            }
        }
        let unresolved = self.unresolved(&groups, &alive, threshold)?;
        let mut parts = vec![usize::MAX; number_of_elements];
        let mut time_scales = Vec::new();
        (0..number_of_elements)
            .filter(|&group| alive[group])
            .for_each(|group| {
                parts[group] = time_scales.len();
                time_scales.push(scales[group])
            });
        Ok(Agglomerated {
            reference,
            elements_parts: owner.iter().map(|&group| parts[group]).collect(),
            time_scales,
            unresolved: unresolved.iter().map(|&group| parts[group]).collect(),
        })
    }
    fn unresolved(
        &self,
        groups: &[Vec<usize>],
        alive: &[bool],
        threshold: Quantity<Time>,
    ) -> Result<Vec<usize>, String> {
        (0..groups.len())
            .filter(|&group| alive[group])
            .filter_map(|group| {
                self.time_scale_exceeds(&groups[group], threshold)
                    .map(|certified| (!certified).then_some(group))
                    .transpose()
            })
            .collect()
    }
}
