#[cfg(test)]
mod test;

use super::Candidates;
use crate::{
    geometry::mesh::{Criterion, ElementsFaces, Merged, Merging},
    math::{Quantity, Scalar},
    units::Time,
};

/// The time scale a typical element is expected to have.
pub enum Reference {
    /// The median of the time scales of the elements, taken one by one.
    Median,
    /// A time scale given directly, such as from a mesh known to be well shaped.
    Value(Quantity<Time>),
}

/// An element is too fast when its time scale is not certified to exceed the reference over
/// `step_reduction`. Such an element is joined to the neighbor that gives the largest time scale,
/// as described by `merging`, if the join can be one virtual element, see [`Candidates::check`].
pub struct Agglomeration {
    pub reference: Reference,
    pub step_reduction: Scalar,
    pub minimum_volume: Scalar,
    pub merging: Merging,
}

/// The elements joined, where the score of each part is the time scale.
pub struct Agglomerated {
    pub reference: Quantity<Time>,
    pub merged: Merged,
}

impl Agglomerated {
    pub fn time_scales(&self) -> Vec<Quantity<Time>> {
        self.merged
            .scores
            .iter()
            .copied()
            .map(Time::seconds)
            .collect()
    }
}

struct TimeScale<'a, S> {
    candidates: &'a Candidates<S>,
    threshold: Quantity<Time>,
    minimum_volume: Scalar,
}

impl<S> Criterion for TimeScale<'_, S>
where
    S: ElementsFaces,
{
    fn score(&self, elements: &[usize]) -> Result<Scalar, String> {
        self.candidates
            .time_scale(elements)
            .map(|time_scale| time_scale.value())
    }
    fn certified(&self, elements: &[usize]) -> Result<bool, String> {
        self.candidates.time_scale_exceeds(elements, self.threshold)
    }
    fn valid(&self, elements: &[usize]) -> bool {
        self.candidates.check(elements, self.minimum_volume).is_ok()
    }
}

impl<S> Candidates<S>
where
    S: ElementsFaces,
{
    pub fn agglomerate(&self, agglomeration: &Agglomeration) -> Result<Agglomerated, String> {
        let scales = self.time_scales()?;
        let reference = match &agglomeration.reference {
            Reference::Median => {
                let mut values = scales.iter().map(|scale| scale.value()).collect::<Vec<_>>();
                values.sort_by(|a, b| a.total_cmp(b));
                Time::seconds(values[values.len() / 2])
            }
            Reference::Value(value) => *value,
        };
        let merged = agglomeration.merging.agglomerate(
            &self.boundary.adjacent(),
            scales.iter().map(|scale| scale.value()).collect(),
            &TimeScale {
                candidates: self,
                threshold: Time::seconds(reference.value() / agglomeration.step_reduction),
                minimum_volume: agglomeration.minimum_volume,
            },
        )?;
        Ok(Agglomerated { reference, merged })
    }
}
