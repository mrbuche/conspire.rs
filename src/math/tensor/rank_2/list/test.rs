use super::{
    super::super::{Tensor, TensorArray},
    TensorRank0, TensorRank2, TensorRank2List,
};

fn get_array() -> [[[TensorRank0; 3]; 3]; 8] {
    [
        [[3.0, 0.0, 3.0], [4.0, 4.0, 1.0], [4.0, 0.0, 4.0]],
        [[3.0, 4.0, 0.0], [1.0, 3.0, 2.0], [4.0, 0.0, 2.0]],
        [[0.0, 1.0, 4.0], [3.0, 1.0, 4.0], [1.0, 0.0, 0.0]],
        [[4.0, 0.0, 1.0], [0.0, 4.0, 3.0], [1.0, 2.0, 2.0]],
        [[3.0, 3.0, 2.0], [4.0, 4.0, 4.0], [4.0, 1.0, 3.0]],
        [[1.0, 3.0, 1.0], [2.0, 2.0, 3.0], [0.0, 0.0, 3.0]],
        [[3.0, 3.0, 4.0], [4.0, 1.0, 2.0], [0.0, 3.0, 3.0]],
        [[3.0, 1.0, 0.0], [3.0, 0.0, 4.0], [0.0, 3.0, 3.0]],
    ]
}

fn get_tensor_rank_2_list() -> TensorRank2List<3, 1, 1, 8> {
    TensorRank2List::new(get_array())
}

#[test]
fn as_array() {
    get_tensor_rank_2_list()
        .as_array()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_2_entry, array_entry)| {
            tensor_rank_2_entry.iter().zip(array_entry.iter()).for_each(
                |(tensor_rank_2_entry_i, array_entry_i)| {
                    tensor_rank_2_entry_i
                        .iter()
                        .zip(array_entry_i.iter())
                        .for_each(|(tensor_rank_2_entry_ij, array_entry_ij)| {
                            assert_eq!(tensor_rank_2_entry_ij, array_entry_ij)
                        })
                },
            )
        });
}

#[test]
fn from_iter() {
    let into_iterator = get_tensor_rank_2_list().0.into_iter();
    let tensor_rank_2_list = TensorRank2List::<3, 1, 1, 8>::from_iter(get_tensor_rank_2_list().0);
    tensor_rank_2_list
        .iter()
        .zip(into_iterator)
        .for_each(|(tensor_rank_2_entry, array_entry)| {
            tensor_rank_2_entry.iter().zip(array_entry.iter()).for_each(
                |(tensor_rank_2_entry_i, array_entry_i)| {
                    tensor_rank_2_entry_i
                        .iter()
                        .zip(array_entry_i.iter())
                        .for_each(|(tensor_rank_2_entry_ij, array_entry_ij)| {
                            assert_eq!(tensor_rank_2_entry_ij, array_entry_ij)
                        })
                },
            )
        });
}

#[test]
fn iter() {
    get_tensor_rank_2_list()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_2_entry, array_entry)| {
            tensor_rank_2_entry.iter().zip(array_entry.iter()).for_each(
                |(tensor_rank_2_entry_i, array_entry_i)| {
                    tensor_rank_2_entry_i
                        .iter()
                        .zip(array_entry_i.iter())
                        .for_each(|(tensor_rank_2_entry_ij, array_entry_ij)| {
                            assert_eq!(tensor_rank_2_entry_ij, array_entry_ij)
                        })
                },
            )
        });
}

#[test]
fn iter_mut() {
    get_tensor_rank_2_list()
        .iter_mut()
        .zip(get_array().iter_mut())
        .for_each(|(tensor_rank_2_entry, array_entry)| {
            tensor_rank_2_entry
                .iter_mut()
                .zip(array_entry.iter_mut())
                .for_each(|(tensor_rank_2_entry_i, array_entry_i)| {
                    tensor_rank_2_entry_i
                        .iter_mut()
                        .zip(array_entry_i.iter_mut())
                        .for_each(|(tensor_rank_2_entry_ij, array_entry_ij)| {
                            assert_eq!(tensor_rank_2_entry_ij, array_entry_ij)
                        })
                })
        });
}

#[test]
fn new() {
    get_tensor_rank_2_list()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_2_entry, array_entry)| {
            tensor_rank_2_entry.iter().zip(array_entry.iter()).for_each(
                |(tensor_rank_2_entry_i, array_entry_i)| {
                    tensor_rank_2_entry_i
                        .iter()
                        .zip(array_entry_i.iter())
                        .for_each(|(tensor_rank_2_entry_ij, array_entry_ij)| {
                            assert_eq!(tensor_rank_2_entry_ij, array_entry_ij)
                        })
                },
            )
        });
}

#[test]
fn size() {
    assert_eq!(
        std::mem::size_of::<TensorRank2List::<3, 1, 1, 8>>(),
        std::mem::size_of::<[TensorRank2::<3, 1, 1>; 8]>()
    )
}

#[test]
fn zero() {
    TensorRank2List::<3, 1, 1, 8>::zero()
        .iter()
        .for_each(|tensor_rank_2_entry| {
            tensor_rank_2_entry
                .iter()
                .for_each(|tensor_rank_2_entry_i| {
                    tensor_rank_2_entry_i
                        .iter()
                        .for_each(|tensor_rank_2_entry_ij| assert_eq!(tensor_rank_2_entry_ij, &0.0))
                })
        });
}
