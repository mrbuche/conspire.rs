use super::{
    super::super::{Tensor, TensorArray},
    TensorRank0, TensorRank3, TensorRank3List,
};

fn get_array() -> [[[[TensorRank0; 3]; 3]; 3]; 2] {
    [
        [
            [[1.0, 2.0, 1.0], [0.0, 3.0, 1.0], [3.0, 1.0, 2.0]],
            [[2.0, 0.0, 3.0], [0.0, 2.0, 3.0], [3.0, 3.0, 2.0]],
            [[2.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 1.0]],
        ],
        [
            [[2.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 1.0]],
            [[2.0, 0.0, 3.0], [0.0, 2.0, 3.0], [3.0, 3.0, 2.0]],
            [[1.0, 2.0, 1.0], [0.0, 3.0, 1.0], [3.0, 1.0, 2.0]],
        ],
    ]
}

fn get_tensor_rank_3_list() -> TensorRank3List<3, 1, 1, 1, 2> {
    TensorRank3List::new(get_array())
}

fn get_other_tensor_rank_3_list() -> TensorRank3List<3, 1, 1, 1, 2> {
    TensorRank3List::new([
        [
            [[5.0, 0.0, 2.0], [3.0, 0.0, 6.0], [5.0, 1.0, 4.0]],
            [[0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [1.0, 3.0, 3.0]],
            [[4.0, 6.0, 6.0], [4.0, 3.0, 4.0], [2.0, 4.0, 0.0]],
        ],
        [
            [[2.0, 4.0, 0.0], [5.0, 1.0, 3.0], [4.0, 2.0, 6.0]],
            [[4.0, 5.0, 2.0], [4.0, 4.0, 0.0], [2.0, 6.0, 0.0]],
            [[6.0, 6.0, 6.0], [5.0, 4.0, 1.0], [4.0, 3.0, 0.0]],
        ],
    ])
}

fn get_other_tensor_rank_3_list_add_tensor_rank_3_list() -> TensorRank3List<3, 1, 1, 1, 2> {
    TensorRank3List::new([
        [
            [[6.0, 2.0, 3.0], [3.0, 3.0, 7.0], [8.0, 2.0, 6.0]],
            [[2.0, 2.0, 3.0], [0.0, 5.0, 3.0], [4.0, 6.0, 5.0]],
            [[6.0, 8.0, 6.0], [5.0, 3.0, 4.0], [2.0, 6.0, 1.0]],
        ],
        [
            [[4.0, 6.0, 0.0], [6.0, 1.0, 3.0], [4.0, 4.0, 7.0]],
            [[6.0, 5.0, 5.0], [4.0, 6.0, 3.0], [5.0, 9.0, 2.0]],
            [[7.0, 8.0, 7.0], [5.0, 7.0, 2.0], [7.0, 4.0, 2.0]],
        ],
    ])
}

#[test]
fn add_assign_tensor_rank_3_list() {
    let mut tensor_rank_3_list = get_tensor_rank_3_list();
    tensor_rank_3_list += get_other_tensor_rank_3_list();
    tensor_rank_3_list
        .iter()
        .zip(get_other_tensor_rank_3_list_add_tensor_rank_3_list().iter())
        .for_each(|(tensor_rank_3, res_tensor_rank_3)| {
            tensor_rank_3.iter().zip(res_tensor_rank_3.iter()).for_each(
                |(tensor_rank_3_i, res_tensor_rank_3_i)| {
                    tensor_rank_3_i
                        .iter()
                        .zip(res_tensor_rank_3_i.iter())
                        .for_each(|(tensor_rank_3_ij, res_tensor_rank_3_ij)| {
                            tensor_rank_3_ij
                                .iter()
                                .zip(res_tensor_rank_3_ij.iter())
                                .for_each(|(tensor_rank_3_ijk, res_tensor_rank_3_ijk)| {
                                    assert_eq!(tensor_rank_3_ijk, res_tensor_rank_3_ijk)
                                })
                        })
                },
            )
        })
}

#[test]
fn add_assign_tensor_rank_3_list_ref() {
    let mut tensor_rank_3_list = get_tensor_rank_3_list();
    tensor_rank_3_list += &get_other_tensor_rank_3_list();
    tensor_rank_3_list
        .iter()
        .zip(get_other_tensor_rank_3_list_add_tensor_rank_3_list().iter())
        .for_each(|(tensor_rank_3, res_tensor_rank_3)| {
            tensor_rank_3.iter().zip(res_tensor_rank_3.iter()).for_each(
                |(tensor_rank_3_i, res_tensor_rank_3_i)| {
                    tensor_rank_3_i
                        .iter()
                        .zip(res_tensor_rank_3_i.iter())
                        .for_each(|(tensor_rank_3_ij, res_tensor_rank_3_ij)| {
                            tensor_rank_3_ij
                                .iter()
                                .zip(res_tensor_rank_3_ij.iter())
                                .for_each(|(tensor_rank_3_ijk, res_tensor_rank_3_ijk)| {
                                    assert_eq!(tensor_rank_3_ijk, res_tensor_rank_3_ijk)
                                })
                        })
                },
            )
        })
}

#[test]
fn from_iter() {
    let into_iterator = get_tensor_rank_3_list().0.into_iter();
    let tensor_rank_3_list =
        TensorRank3List::<3, 1, 1, 1, 2>::from_iter(get_tensor_rank_3_list().0);
    tensor_rank_3_list
        .iter()
        .zip(into_iterator)
        .for_each(|(tensor_rank_3_entry, array_entry)| {
            tensor_rank_3_entry.iter().zip(array_entry.iter()).for_each(
                |(tensor_rank_3_entry_i, array_entry_i)| {
                    tensor_rank_3_entry_i
                        .iter()
                        .zip(array_entry_i.iter())
                        .for_each(|(tensor_rank_3_entry_ij, array_entry_ij)| {
                            tensor_rank_3_entry_ij
                                .iter()
                                .zip(array_entry_ij.iter())
                                .for_each(|(tensor_rank_3_entry_ijk, array_entry_ijk)| {
                                    assert_eq!(tensor_rank_3_entry_ijk, array_entry_ijk)
                                })
                        })
                },
            )
        })
}

#[test]
fn iter() {
    get_tensor_rank_3_list()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_3_entry, array_entry)| {
            tensor_rank_3_entry.iter().zip(array_entry.iter()).for_each(
                |(tensor_rank_3_entry_i, array_entry_i)| {
                    tensor_rank_3_entry_i
                        .iter()
                        .zip(array_entry_i.iter())
                        .for_each(|(tensor_rank_3_entry_ij, array_entry_ij)| {
                            tensor_rank_3_entry_ij
                                .iter()
                                .zip(array_entry_ij.iter())
                                .for_each(|(tensor_rank_3_entry_ijk, array_entry_ijk)| {
                                    assert_eq!(tensor_rank_3_entry_ijk, array_entry_ijk)
                                })
                        })
                },
            )
        })
}

#[test]
fn iter_mut() {
    get_tensor_rank_3_list()
        .iter_mut()
        .zip(get_array().iter_mut())
        .for_each(|(tensor_rank_3_entry, array_entry)| {
            tensor_rank_3_entry
                .iter_mut()
                .zip(array_entry.iter_mut())
                .for_each(|(tensor_rank_3_entry_i, array_entry_i)| {
                    tensor_rank_3_entry_i
                        .iter_mut()
                        .zip(array_entry_i.iter_mut())
                        .for_each(|(tensor_rank_3_entry_ij, array_entry_ij)| {
                            tensor_rank_3_entry_ij
                                .iter_mut()
                                .zip(array_entry_ij.iter_mut())
                                .for_each(|(tensor_rank_3_entry_ijk, array_entry_ijk)| {
                                    assert_eq!(tensor_rank_3_entry_ijk, array_entry_ijk)
                                })
                        })
                })
        })
}

#[test]
fn new() {
    get_tensor_rank_3_list()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_3_entry, array_entry)| {
            tensor_rank_3_entry.iter().zip(array_entry.iter()).for_each(
                |(tensor_rank_3_entry_i, array_entry_i)| {
                    tensor_rank_3_entry_i
                        .iter()
                        .zip(array_entry_i.iter())
                        .for_each(|(tensor_rank_3_entry_ij, array_entry_ij)| {
                            tensor_rank_3_entry_ij
                                .iter()
                                .zip(array_entry_ij.iter())
                                .for_each(|(tensor_rank_3_entry_ijk, array_entry_ijk)| {
                                    assert_eq!(tensor_rank_3_entry_ijk, array_entry_ijk)
                                })
                        })
                },
            )
        })
}

#[test]
fn size() {
    assert_eq!(
        std::mem::size_of::<TensorRank3List::<3, 1, 1, 1, 2>>(),
        std::mem::size_of::<[TensorRank3::<3, 1, 1, 1>; 2]>()
    )
}

#[test]
fn zero() {
    TensorRank3List::<3, 1, 1, 1, 2>::zero()
        .iter()
        .for_each(|tensor_rank_3_entry| {
            tensor_rank_3_entry
                .iter()
                .for_each(|tensor_rank_3_entry_i| {
                    tensor_rank_3_entry_i
                        .iter()
                        .for_each(|tensor_rank_3_entry_ij| {
                            tensor_rank_3_entry_ij
                                .iter()
                                .for_each(|tensor_rank_3_entry_ijk| {
                                    assert_eq!(tensor_rank_3_entry_ijk, &0.0)
                                })
                        })
                })
        })
}
