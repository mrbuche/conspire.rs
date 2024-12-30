use super::{
    super::{
        super::{Tensor, TensorArray},
        TensorRank3,
    },
    TensorRank0, TensorRank3List2D,
};

fn get_array() -> [[[[[TensorRank0; 3]; 3]; 3]; 2]; 2] {
    [
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
        ],
        [
            [
                [[2.0, 0.0, 3.0], [0.0, 2.0, 3.0], [3.0, 3.0, 2.0]],
                [[1.0, 2.0, 1.0], [0.0, 3.0, 1.0], [3.0, 1.0, 2.0]],
                [[2.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 1.0]],
            ],
            [
                [[2.0, 0.0, 3.0], [0.0, 2.0, 3.0], [3.0, 3.0, 2.0]],
                [[2.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 1.0]],
                [[1.0, 2.0, 1.0], [0.0, 3.0, 1.0], [3.0, 1.0, 2.0]],
            ],
        ],
    ]
}

fn get_tensor_rank_3_list_2d() -> TensorRank3List2D<3, 1, 1, 1, 2, 2> {
    TensorRank3List2D::new(get_array())
}

#[test]
fn from_iter() {
    let into_iterator = get_tensor_rank_3_list_2d().0.into_iter();
    let tensor_rank_3_list_2d =
        TensorRank3List2D::<3, 1, 1, 1, 2, 2>::from_iter(get_tensor_rank_3_list_2d().0);
    tensor_rank_3_list_2d.iter().zip(into_iterator).for_each(
        |(tensor_rank_3_list_2d_entry, array_entry)| {
            tensor_rank_3_list_2d_entry
                .iter()
                .zip(array_entry.iter())
                .for_each(|(tensor_rank_3, array)| {
                    tensor_rank_3
                        .iter()
                        .zip(array.iter())
                        .for_each(|(tensor_rank_3_i, array_i)| {
                            tensor_rank_3_i.iter().zip(array_i.iter()).for_each(
                                |(tensor_rank_3_ij, array_ij)| {
                                    tensor_rank_3_ij.iter().zip(array_ij.iter()).for_each(
                                        |(tensor_rank_3_ijk, array_ijk)| {
                                            assert_eq!(tensor_rank_3_ijk, array_ijk)
                                        },
                                    )
                                },
                            )
                        })
                })
        },
    )
}

#[test]
fn iter() {
    get_tensor_rank_3_list_2d()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_3_list_2d_entry, array_entry)| {
            tensor_rank_3_list_2d_entry
                .iter()
                .zip(array_entry.iter())
                .for_each(|(tensor_rank_3, array)| {
                    tensor_rank_3
                        .iter()
                        .zip(array.iter())
                        .for_each(|(tensor_rank_3_i, array_i)| {
                            tensor_rank_3_i.iter().zip(array_i.iter()).for_each(
                                |(tensor_rank_3_ij, array_ij)| {
                                    tensor_rank_3_ij.iter().zip(array_ij.iter()).for_each(
                                        |(tensor_rank_3_ijk, array_ijk)| {
                                            assert_eq!(tensor_rank_3_ijk, array_ijk)
                                        },
                                    )
                                },
                            )
                        })
                })
        })
}

#[test]
fn iter_mut() {
    get_tensor_rank_3_list_2d()
        .iter_mut()
        .zip(get_array().iter_mut())
        .for_each(|(tensor_rank_3_list_2d_entry, array_entry)| {
            tensor_rank_3_list_2d_entry
                .iter_mut()
                .zip(array_entry.iter_mut())
                .for_each(|(tensor_rank_3, array)| {
                    tensor_rank_3.iter_mut().zip(array.iter_mut()).for_each(
                        |(tensor_rank_3_i, array_i)| {
                            tensor_rank_3_i.iter_mut().zip(array_i.iter_mut()).for_each(
                                |(tensor_rank_3_ij, array_ij)| {
                                    tensor_rank_3_ij
                                        .iter_mut()
                                        .zip(array_ij.iter_mut())
                                        .for_each(|(tensor_rank_3_ijk, array_ijk)| {
                                            assert_eq!(tensor_rank_3_ijk, array_ijk)
                                        })
                                },
                            )
                        },
                    )
                })
        })
}

#[test]
fn new() {
    get_tensor_rank_3_list_2d()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_3_list_2d_entry, array_entry)| {
            tensor_rank_3_list_2d_entry
                .iter()
                .zip(array_entry.iter())
                .for_each(|(tensor_rank_3, array)| {
                    tensor_rank_3
                        .iter()
                        .zip(array.iter())
                        .for_each(|(tensor_rank_3_i, array_i)| {
                            tensor_rank_3_i.iter().zip(array_i.iter()).for_each(
                                |(tensor_rank_3_ij, array_ij)| {
                                    tensor_rank_3_ij.iter().zip(array_ij.iter()).for_each(
                                        |(tensor_rank_3_ijk, array_ijk)| {
                                            assert_eq!(tensor_rank_3_ijk, array_ijk)
                                        },
                                    )
                                },
                            )
                        })
                })
        })
}

#[test]
fn size() {
    assert_eq!(
        std::mem::size_of::<TensorRank3List2D::<3, 1, 1, 1, 8, 8>>(),
        std::mem::size_of::<[[TensorRank3::<3, 1, 1, 1>; 8]; 8]>()
    )
}

#[test]
fn zero() {
    TensorRank3List2D::<3, 1, 1, 1, 8, 8>::zero()
        .iter()
        .for_each(|tensor_rank_3_list_2d_entry| {
            tensor_rank_3_list_2d_entry
                .iter()
                .for_each(|tensor_rank_3| {
                    tensor_rank_3.iter().for_each(|tensor_rank_3_i| {
                        tensor_rank_3_i.iter().for_each(|tensor_rank_3_ij| {
                            tensor_rank_3_ij
                                .iter()
                                .for_each(|tensor_rank_3_ijk| assert_eq!(tensor_rank_3_ijk, &0.0))
                        })
                    })
                })
        })
}
