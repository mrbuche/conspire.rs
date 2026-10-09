#[cfg(test)]
mod test;

use super::PrimitiveConnectivity;
use std::array::from_fn;

/// The facets of an element with a fixed topology, as local node indices, each oriented outward.
pub trait LocalFacets {
    const LOCAL_FACETS: &'static [&'static [usize]];
    type Facet: AsRef<[usize]>;
    fn facet(local: &[usize], element: &[usize]) -> Self::Facet;
}

/// A facet of an element that has both triangles and quadrilaterals.
pub enum MixedFacet {
    Triangle([usize; 3]),
    Quadrilateral([usize; 4]),
}

impl MixedFacet {
    fn new(local: &[usize], element: &[usize]) -> Self {
        match local.len() {
            3 => Self::Triangle(gather(local, element)),
            4 => Self::Quadrilateral(gather(local, element)),
            length => panic!("a facet has {length} nodes, not three or four"),
        }
    }
}

impl AsRef<[usize]> for MixedFacet {
    fn as_ref(&self) -> &[usize] {
        match self {
            Self::Triangle(nodes) => nodes,
            Self::Quadrilateral(nodes) => nodes,
        }
    }
}

fn gather<const K: usize>(local: &[usize], element: &[usize]) -> [usize; K] {
    assert_eq!(local.len(), K, "a facet does not have {K} nodes");
    from_fn(|index| element[local[index]])
}

impl LocalFacets for PrimitiveConnectivity<3, 8> {
    const LOCAL_FACETS: &'static [&'static [usize]] = &[
        &[0, 1, 5, 4],
        &[1, 2, 6, 5],
        &[2, 3, 7, 6],
        &[3, 0, 4, 7],
        &[0, 3, 2, 1],
        &[4, 5, 6, 7],
    ];
    type Facet = [usize; 4];
    fn facet(local: &[usize], element: &[usize]) -> [usize; 4] {
        gather(local, element)
    }
}

impl LocalFacets for PrimitiveConnectivity<3, 4> {
    const LOCAL_FACETS: &'static [&'static [usize]] =
        &[&[0, 1, 3], &[1, 2, 3], &[2, 0, 3], &[0, 2, 1]];
    type Facet = [usize; 3];
    fn facet(local: &[usize], element: &[usize]) -> [usize; 3] {
        gather(local, element)
    }
}

impl LocalFacets for PrimitiveConnectivity<3, 5> {
    const LOCAL_FACETS: &'static [&'static [usize]] = &[
        &[0, 1, 4],
        &[1, 2, 4],
        &[2, 3, 4],
        &[3, 0, 4],
        &[0, 3, 2, 1],
    ];
    type Facet = MixedFacet;
    fn facet(local: &[usize], element: &[usize]) -> MixedFacet {
        MixedFacet::new(local, element)
    }
}

impl LocalFacets for PrimitiveConnectivity<3, 6> {
    const LOCAL_FACETS: &'static [&'static [usize]] = &[
        &[0, 1, 4, 3],
        &[1, 2, 5, 4],
        &[2, 0, 3, 5],
        &[0, 2, 1],
        &[3, 4, 5],
    ];
    type Facet = MixedFacet;
    fn facet(local: &[usize], element: &[usize]) -> MixedFacet {
        MixedFacet::new(local, element)
    }
}

impl LocalFacets for PrimitiveConnectivity<2, 4> {
    const LOCAL_FACETS: &'static [&'static [usize]] = &[&[0, 1], &[1, 2], &[2, 3], &[3, 0]];
    type Facet = [usize; 2];
    fn facet(local: &[usize], element: &[usize]) -> [usize; 2] {
        gather(local, element)
    }
}

impl LocalFacets for PrimitiveConnectivity<2, 3> {
    const LOCAL_FACETS: &'static [&'static [usize]] = &[&[0, 1], &[1, 2], &[2, 0]];
    type Facet = [usize; 2];
    fn facet(local: &[usize], element: &[usize]) -> [usize; 2] {
        gather(local, element)
    }
}
