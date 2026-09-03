//! Grouping the solids read from one CAD file into meshable bodies: a solid
//! enclosed by no other is a body, and any solid whose centre lies inside a
//! body is a void carved out of it.

#[cfg(test)]
mod test;

use super::brep::Brep;
use crate::geometry::{
    Coordinate,
    csg::{
        Primitive,
        ops::{Difference, UnionAll},
    },
    solid::{Solid, SolidOracle},
};
use std::array::from_fn;

const D: usize = 3;

/// One recognised body with its voids carved out.
pub type Body = Difference<Primitive, UnionAll<Primitive>>;

/// Groups recognised solids into [`Body`]s. Each solid enclosed by no other
/// becomes a body; every other solid is carved from the innermost body that
/// encloses its centre. Errors if a solid is not a recognised [`Primitive`] or
/// lies inside no body. One [`Body`] per body, in input order.
pub fn assemble(breps: &[Brep]) -> Result<Vec<Body>, &'static str> {
    let primitives = breps
        .iter()
        .map(|brep| {
            brep.primitive()
                .ok_or("solid is not a recognised primitive")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let count = primitives.len();

    let mut centre = Vec::with_capacity(count);
    let mut oracle = Vec::with_capacity(count);
    let mut volume = Vec::with_capacity(count);
    for primitive in &primitives {
        let (low, high) = primitive.bounding_box()?;
        let mid: [f64; D] = from_fn(|k| 0.5 * (low[k].value() + high[k].value()));
        centre.push(Coordinate::from(mid));
        volume.push(
            (0..D)
                .map(|k| high[k].value() - low[k].value())
                .product::<f64>(),
        );
        oracle.push(primitive.oracle()?);
    }

    let encloses = |i: usize, j: usize| i != j && oracle[i].signed_distance(&centre[j]) > 0.0;
    let is_root: Vec<bool> = (0..count)
        .map(|j| !(0..count).any(|i| encloses(i, j)))
        .collect();

    let mut voids: Vec<Vec<usize>> = vec![Vec::new(); count];
    for j in 0..count {
        if is_root[j] {
            continue;
        }
        let host = (0..count)
            .filter(|&i| is_root[i] && encloses(i, j))
            .min_by(|&a, &b| volume[a].total_cmp(&volume[b]))
            .ok_or("solid lies inside no body")?;
        voids[host].push(j);
    }

    let mut slots: Vec<Option<Primitive>> = primitives.into_iter().map(Some).collect();
    Ok((0..count)
        .filter(|&root| is_root[root])
        .map(|root| {
            let matrix = slots[root].take().expect("a root is taken once");
            let carved = voids[root]
                .iter()
                .map(|&void| slots[void].take().expect("a void is taken once"))
                .collect();
            Difference::new(matrix, UnionAll::new(carved))
        })
        .collect())
}
