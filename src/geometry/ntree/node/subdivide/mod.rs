use crate::geometry::ntree::{
    error::OrthotreeError,
    node::{Kind, Node, split::Split},
};
use std::{array::from_fn, ops::AddAssign};

impl<T, U> Node<3, 6, 8, T, U>
where
    T: AddAssign + Copy + Split,
    U: Copy + std::default::Default,
{
    pub fn subdivide(&self, indices: [U; 8]) -> Result<[Self; 8], OrthotreeError> {
        match self.kind {
            Kind::Leaf => {
                let length = self.length.split();
                let min_x = self.corner[0];
                let min_y = self.corner[1];
                let min_z = self.corner[2];
                let mut val_x = min_x;
                val_x += length;
                let mut val_y = min_y;
                val_y += length;
                let mut val_z = min_z;
                val_z += length;
                let corner = self.corner;
                let facets = self.facets;
                Ok([
                    Node {
                        corner: [min_x, min_y, min_z],
                        length,
                        facets: [U::default(), indices[1], indices[2], U::default(), U::default(), indices[4]],
                        kind: Kind::Leaf,
                    },
                    Node {
                        corner: [val_x, min_y, min_z],
                        length,
                        facets: [U::default(), U::default(), indices[3], indices[0], U::default(), indices[5]],
                        kind: Kind::Leaf,
                    },
                    Node {
                        corner: [min_x, val_y, min_z],
                        length,
                        facets: [indices[0], indices[3], U::default(), U::default(), U::default(), indices[6]],
                        kind: Kind::Leaf,
                    },
                    Node {
                        corner: [val_x, val_y, min_z],
                        length,
                        facets: [indices[1], U::default(), U::default(), indices[2], U::default(), indices[7]],
                        kind: Kind::Leaf,
                    },
                    Node {
                        corner: [min_x, min_y, val_z],
                        length,
                        facets: [U::default(), indices[5], indices[6], U::default(), indices[0], U::default()],
                        kind: Kind::Leaf,
                    },
                    Node {
                        corner: [val_x, min_y, val_z],
                        length,
                        facets: [U::default(), U::default(), indices[7], indices[4], indices[1], U::default()],
                        kind: Kind::Leaf,
                    },
                    Node {
                        corner: [min_x, val_y, val_z],
                        length,
                        facets: [indices[4], indices[7], U::default(), U::default(), indices[2], U::default()],
                        kind: Kind::Leaf,
                    },
                    Node {
                        corner: [val_x, val_y, val_z],
                        length,
                        facets: [indices[5], U::default(), U::default(), indices[6], indices[3], U::default()],
                        kind: Kind::Leaf,
                    },
                ])
            }
            Kind::Tree(_) => Err(OrthotreeError::CannotSubdivideTree),
        }
    }
}

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
