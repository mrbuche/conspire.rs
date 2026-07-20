#[cfg(test)]
mod test;

use crate::{
    geometry::ntree::{
        Balancing, Orthotree, Pairing, Rescaling,
        node::{Kind, Node, split::Split},
    },
    io::vtk::{
        invalid,
        read::{attribute, bits, data_array, encoding, floats, tag},
        unsupported,
    },
    math::Scalar,
};
use std::{
    array::from_fn, collections::VecDeque, fs::read_to_string, io::Result, ops::Add, path::Path,
};

pub trait ReadHtg<P>
where
    P: AsRef<Path>,
    Self: Sized,
{
    fn read_htg(input: P) -> Result<Self>;
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, P> ReadHtg<P>
    for Orthotree<D, L, M, N, T, U>
where
    P: AsRef<Path>,
    T: Add<Output = T> + Copy + Split + Into<usize> + TryFrom<usize>,
    U: Copy + From<usize> + Into<usize>,
{
    fn read_htg(input: P) -> Result<Self> {
        let text = read_to_string(input)?;
        let header = tag(&text, "<VTKFile")?;
        if attribute(header, "type") != Some("HyperTreeGrid") {
            return Err(invalid("not a HyperTreeGrid".into()));
        }
        if matches!(attribute(header, "byte_order"), Some(order) if order != "LittleEndian") {
            return Err(unsupported("big-endian HTG is not supported"));
        }
        let encoding = encoding(header)?;
        let grid = tag(&text, "<HyperTreeGrid")?;
        if matches!(attribute(grid, "BranchFactor"), Some(factor) if factor != "2") {
            return Err(unsupported("only BranchFactor 2 is supported"));
        }
        let dimensions =
            attribute(grid, "Dimensions").ok_or_else(|| invalid("no Dimensions".into()))?;
        let split_axes = dimensions.split_whitespace().filter(|d| *d != "1").count();
        if split_axes != D {
            return Err(invalid(format!(
                "HTG has {split_axes} refined axes but Orthotree was asked for D={D}"
            )));
        }
        let axes = ["XCoordinates", "YCoordinates", "ZCoordinates"];
        let mut lo = [0.0; D];
        let mut hi = [0.0; D];
        for (axis, &name) in axes.iter().take(D).enumerate() {
            let coordinates = floats(&data_array(&text, Some(name))?, &encoding)?;
            lo[axis] = coordinates[0] as Scalar;
            hi[axis] = coordinates[1] as Scalar;
        }
        let levels: usize = attribute(tag(&text, "<Tree ")?, "NumberOfLevels")
            .and_then(|n| n.parse().ok())
            .ok_or_else(|| invalid("no NumberOfLevels".into()))?;
        let descriptor = bits(&data_array(&text, Some("Descriptor"))?, &encoding)?;
        let root_length = 1usize << (levels - 1);
        let cell = (hi[0] - lo[0]) / root_length as Scalar;
        let rescale = Rescaling {
            center: from_fn(|axis| 0.5 * (lo[axis] + hi[axis])),
            cell,
            half: 0.5 * root_length as Scalar,
        };
        let zero = number(0)?;
        let mut tree = Orthotree {
            balanced: Balancing::None,
            nodes: vec![Node {
                corner: from_fn(|_| zero),
                length: number(root_length)?,
                facets: [None; M],
                kind: Kind::Leaf,
                value: None,
            }],
            paired: Pairing::None,
            rescale,
        };
        let mut queue: VecDeque<U> = VecDeque::from([U::from(0)]);
        let mut bit = 0;
        while let Some(node) = queue.pop_front() {
            if bit >= descriptor.len() {
                break;
            }
            let refined = descriptor[bit] == 1;
            bit += 1;
            if refined {
                tree.subdivide(node).map_err(|e| invalid(e.into()))?;
                queue.extend(tree.nodes[node.into()].orthants().unwrap().iter().copied());
            }
        }
        Ok(tree)
    }
}

fn number<T: TryFrom<usize>>(value: usize) -> Result<T> {
    T::try_from(value).map_err(|_| invalid("tree coordinate does not fit in T".into()))
}
