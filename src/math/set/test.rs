#[test]
fn todo() {
    todo!()
}

// use crate::math::{InverseSets, Sets};

// const DATA: [[usize; 3]; 4] = [[0, 3, 5], [1, 5, 8], [2, 9, 1], [11, 0, 2]];

// fn data() -> Vec<[usize; 3]> {
//     DATA.to_vec()
// }

// fn sets() -> Sets<Vec<[usize; 3]>, [usize; 3]> {
//     Sets {
//         data_sets: DATA.to_vec(),
//     }
// }

// mod deserialization {
//     use super::*;
//     #[test]
//     fn to_vec() {
//         let sets = Sets::from(data());
//         assert_eq!(Vec::from(sets), data())
//     }
// }

// mod serialization {
//     use super::*;
//     #[test]
//     fn from_vec() {
//         assert_eq!(Sets::from(data()), sets())
//     }
// }

// #[test]
// fn inverse() {
//     assert_eq!(
//         Sets::from(data()).inverse(),
//         vec![
//             vec![0, 3],
//             vec![1, 2],
//             vec![2, 3],
//             vec![0],
//             vec![],
//             vec![0, 1],
//             vec![],
//             vec![],
//             vec![1],
//             vec![2],
//             vec![],
//             vec![3],
//         ]
//         .into(),
//     )
// }
