use crate::geometry::ntree::{
    error::OrthotreeError,
    node::{split::Split, Kind, Node},
};
use std::{array::from_fn, ops::AddAssign};

impl<const D: usize, const M: usize, const N: usize, T, U> Node<D, M, N, T, U>
where
    T: AddAssign + Copy + Split,
    U: Copy,
{
    pub fn subdivide(&self, indices: [U; N]) -> Result<[Self; N], OrthotreeError> {
        debug_assert_eq!(M, 2 * D);
        debug_assert_eq!(N, 1usize << D);

        match self.kind {
            Kind::Leaf => {
                let length = self.length.split();
                let corner = self.corner;

                Ok(from_fn(|i| Node {
                    corner: from_fn(|axis| {
                        if ((i >> axis) & 1) == 1 {
                            let mut c = corner[axis];
                            c += length;
                            c
                        } else {
                            corner[axis]
                        }
                    }),
                    length,
                    facets: from_fn(|f| {
                        let axis = f / 2;
                        let sign = f % 2; // 0 = -axis, 1 = +axis
                        let bit = (i >> axis) & 1;

                        if bit == sign {
                            None
                        } else {
                            Some(indices[i ^ (1 << axis)])
                        }
                    }),
                    kind: Kind::Leaf,
                }))
            }
            Kind::Tree(_) => Err(OrthotreeError::CannotSubdivideTree),
        }
    }
}

// use crate::geometry::ntree::{
//     error::OrthotreeError,
//     node::{Kind, Node, split::Split},
// };
// use std::{array::from_fn, ops::AddAssign};

// impl<T, U> Node<3, 6, 8, T, U>
// where
//     T: AddAssign + Copy + Split,
//     U: Copy,
// {
//     pub fn subdivide(&self, indices: [U; 8]) -> Result<[Self; 8], OrthotreeError> {
//         match self.kind {
//             Kind::Leaf => {
//                 let length = self.length.split();
//                 let min_x = self.corner[0];
//                 let min_y = self.corner[1];
//                 let min_z = self.corner[2];
//                 let mut val_x = min_x;
//                 val_x += length;
//                 let mut val_y = min_y;
//                 val_y += length;
//                 let mut val_z = min_z;
//                 val_z += length;
//                 let corner = self.corner;
//                 let facets = self.facets;
//                 Ok([
//                     Node {
//                         corner: [min_x, min_y, min_z],
//                         length,
//                         facets: [
//                             None,
//                             Some(indices[1]),
//                             None,
//                             Some(indices[2]),
//                             None,
//                             Some(indices[4]),
//                         ],
//                         kind: Kind::Leaf,
//                     },
//                     Node {
//                         corner: [val_x, min_y, min_z],
//                         length,
//                         facets: [
//                             Some(indices[0]),
//                             None,
//                             None,
//                             Some(indices[3]),
//                             None,
//                             Some(indices[5]),
//                         ],
//                         kind: Kind::Leaf,
//                     },
//                     Node {
//                         corner: [min_x, val_y, min_z],
//                         length,
//                         facets: [
//                             None,
//                             Some(indices[3]),
//                             Some(indices[0]),
//                             None,
//                             None,
//                             Some(indices[6]),
//                         ],
//                         kind: Kind::Leaf,
//                     },
//                     Node {
//                         corner: [val_x, val_y, min_z],
//                         length,
//                         facets: [
//                             Some(indices[2]),
//                             None,
//                             Some(indices[1]),
//                             None,
//                             None,
//                             Some(indices[7]),
//                         ],
//                         kind: Kind::Leaf,
//                     },
//                     Node {
//                         corner: [min_x, min_y, val_z],
//                         length,
//                         facets: [
//                             None,
//                             Some(indices[5]),
//                             None,
//                             Some(indices[6]),
//                             Some(indices[0]),
//                             None,
//                         ],
//                         kind: Kind::Leaf,
//                     },
//                     Node {
//                         corner: [val_x, min_y, val_z],
//                         length,
//                         facets: [
//                             Some(indices[4]),
//                             None,
//                             None,
//                             Some(indices[7]),
//                             Some(indices[1]),
//                             None,
//                         ],
//                         kind: Kind::Leaf,
//                     },
//                     Node {
//                         corner: [min_x, val_y, val_z],
//                         length,
//                         facets: [
//                             None,
//                             Some(indices[7]),
//                             Some(indices[4]),
//                             None,
//                             Some(indices[2]),
//                             None,
//                         ],
//                         kind: Kind::Leaf,
//                     },
//                     Node {
//                         corner: [val_x, val_y, val_z],
//                         length,
//                         facets: [
//                             Some(indices[6]),
//                             None,
//                             Some(indices[5]),
//                             None,
//                             Some(indices[3]),
//                             None,
//                         ],
//                         kind: Kind::Leaf,
//                     },
//                 ])
//             }
//             Kind::Tree(_) => Err(OrthotreeError::CannotSubdivideTree),
//         }
//     }
// }

// impl<const D: usize, const M: usize, const N: usize, T, U> Node<D, M, N, T, U>
// where
//     T: AddAssign + Copy + Split,
//     U: Copy,
// {
//     pub fn subdivide(&self, indices: [U; N]) -> Result<[Self; N], OrthotreeError> {
//         match self.kind {
//             Kind::Leaf => {
//                 let length = self.length.split();
//                 let corner = self.corner;
//                 let facets = self.facets;
//                 Ok(from_fn(|i| Node {
//                     corner: from_fn(|ax| {
//                         if (i >> ax) & 1 == 1 {
//                             let mut c = corner[ax];
//                             c += length;
//                             c
//                         } else {
//                             corner[ax]
//                         }
//                     }),
//                     length,
//                     facets: from_fn(|f| {
//                         let axis = f / 2;
//                         let sign = f % 2;
//                         if (i >> axis) & 1 == sign {
//                             facets[f]
//                         } else {
//                             indices[i ^ (1 << axis)]
//                         }
//                     }),
//                     kind: Kind::Leaf,
//                 }))
//             }
//             Kind::Tree(_) => Err(OrthotreeError::CannotSubdivideLeaf),
//         }
//     }
// }
