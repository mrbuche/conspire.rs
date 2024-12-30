use super::{
    super::{
        super::{Tensor, TensorArray},
        TensorRank0, TensorRank3,
    },
    TensorRank3List3D,
};

#[allow(clippy::type_complexity)]
fn get_array() -> [[[[[[TensorRank0; 3]; 3]; 3]; 2]; 2]; 2] {
    [
        [
            [
                [
                    [[0.0, 4.0, 4.0], [4.0, 4.0, 6.0], [2.0, 3.0, 5.0]],
                    [[2.0, 5.0, 2.0], [5.0, 1.0, 3.0], [3.0, 0.0, 5.0]],
                    [[2.0, 0.0, 1.0], [2.0, 3.0, 1.0], [0.0, 2.0, 5.0]],
                ],
                [
                    [[5.0, 4.0, 2.0], [3.0, 4.0, 2.0], [1.0, 1.0, 2.0]],
                    [[4.0, 4.0, 6.0], [2.0, 2.0, 1.0], [2.0, 5.0, 4.0]],
                    [[5.0, 5.0, 6.0], [0.0, 5.0, 2.0], [4.0, 4.0, 6.0]],
                ],
            ],
            [
                [
                    [[4.0, 3.0, 3.0], [0.0, 1.0, 2.0], [3.0, 0.0, 2.0]],
                    [[3.0, 4.0, 1.0], [3.0, 2.0, 1.0], [0.0, 3.0, 0.0]],
                    [[6.0, 0.0, 5.0], [0.0, 1.0, 0.0], [1.0, 6.0, 2.0]],
                ],
                [
                    [[5.0, 3.0, 3.0], [1.0, 1.0, 2.0], [4.0, 6.0, 0.0]],
                    [[0.0, 4.0, 0.0], [3.0, 0.0, 6.0], [1.0, 0.0, 5.0]],
                    [[5.0, 4.0, 6.0], [1.0, 4.0, 3.0], [1.0, 2.0, 1.0]],
                ],
            ],
        ],
        [
            [
                [
                    [[6.0, 2.0, 1.0], [0.0, 5.0, 2.0], [1.0, 0.0, 1.0]],
                    [[4.0, 2.0, 0.0], [3.0, 4.0, 5.0], [4.0, 0.0, 3.0]],
                    [[4.0, 5.0, 0.0], [6.0, 4.0, 2.0], [2.0, 5.0, 0.0]],
                ],
                [
                    [[4.0, 6.0, 3.0], [6.0, 1.0, 3.0], [3.0, 1.0, 6.0]],
                    [[1.0, 5.0, 3.0], [1.0, 3.0, 0.0], [6.0, 5.0, 2.0]],
                    [[0.0, 1.0, 6.0], [3.0, 3.0, 6.0], [2.0, 2.0, 2.0]],
                ],
            ],
            [
                [
                    [[2.0, 5.0, 3.0], [0.0, 1.0, 2.0], [6.0, 5.0, 0.0]],
                    [[5.0, 0.0, 1.0], [5.0, 6.0, 2.0], [2.0, 0.0, 4.0]],
                    [[5.0, 4.0, 6.0], [5.0, 3.0, 3.0], [3.0, 3.0, 2.0]],
                ],
                [
                    [[4.0, 6.0, 5.0], [0.0, 0.0, 5.0], [1.0, 3.0, 6.0]],
                    [[5.0, 1.0, 5.0], [0.0, 5.0, 5.0], [2.0, 3.0, 5.0]],
                    [[2.0, 2.0, 1.0], [2.0, 0.0, 1.0], [0.0, 2.0, 2.0]],
                ],
            ],
        ],
    ]
}

fn get_tensor_rank_3_list_3d() -> TensorRank3List3D<3, 1, 1, 1, 2, 2, 2> {
    TensorRank3List3D::new(get_array())
}

#[test]
fn from_iter() {
    let into_iterator = get_tensor_rank_3_list_3d().0.into_iter();
    let tensor_rank_3_list_3d =
        TensorRank3List3D::<3, 1, 1, 1, 2, 2, 2>::from_iter(get_tensor_rank_3_list_3d().0);
    tensor_rank_3_list_3d
        .iter()
        .zip(into_iterator)
        .for_each(|(tensor_rank_3_list_2d, array)| {
            tensor_rank_3_list_2d.iter().zip(array.iter()).for_each(
                |(tensor_rank_3_list_1d, array_entry)| {
                    tensor_rank_3_list_1d
                        .iter()
                        .zip(array_entry.iter())
                        .for_each(|(tensor_rank_3, array)| {
                            tensor_rank_3.iter().zip(array.iter()).for_each(
                                |(tensor_rank_3_i, array_i)| {
                                    tensor_rank_3_i.iter().zip(array_i.iter()).for_each(
                                        |(tensor_rank_3_ij, array_ij)| {
                                            tensor_rank_3_ij.iter().zip(array_ij.iter()).for_each(
                                                |(tensor_rank_3_ijk, array_ijk)| {
                                                    assert_eq!(tensor_rank_3_ijk, array_ijk)
                                                },
                                            )
                                        },
                                    )
                                },
                            )
                        })
                },
            )
        })
}

#[test]
fn iter() {
    get_tensor_rank_3_list_3d()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_3_list_2d, array)| {
            tensor_rank_3_list_2d.iter().zip(array.iter()).for_each(
                |(tensor_rank_3_list_1d, array_entry)| {
                    tensor_rank_3_list_1d
                        .iter()
                        .zip(array_entry.iter())
                        .for_each(|(tensor_rank_3, array)| {
                            tensor_rank_3.iter().zip(array.iter()).for_each(
                                |(tensor_rank_3_i, array_i)| {
                                    tensor_rank_3_i.iter().zip(array_i.iter()).for_each(
                                        |(tensor_rank_3_ij, array_ij)| {
                                            tensor_rank_3_ij.iter().zip(array_ij.iter()).for_each(
                                                |(tensor_rank_3_ijk, array_ijk)| {
                                                    assert_eq!(tensor_rank_3_ijk, array_ijk)
                                                },
                                            )
                                        },
                                    )
                                },
                            )
                        })
                },
            )
        })
}

#[test]
fn iter_mut() {
    get_tensor_rank_3_list_3d()
        .iter_mut()
        .zip(get_array().iter_mut())
        .for_each(|(tensor_rank_3_list_2d, array)| {
            tensor_rank_3_list_2d
                .iter_mut()
                .zip(array.iter_mut())
                .for_each(|(tensor_rank_3_list_1d, array_entry)| {
                    tensor_rank_3_list_1d
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
        })
}

#[test]
fn new() {
    get_tensor_rank_3_list_3d()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_3_list_2d, array)| {
            tensor_rank_3_list_2d.iter().zip(array.iter()).for_each(
                |(tensor_rank_3_list_1d, array_entry)| {
                    tensor_rank_3_list_1d
                        .iter()
                        .zip(array_entry.iter())
                        .for_each(|(tensor_rank_3, array)| {
                            tensor_rank_3.iter().zip(array.iter()).for_each(
                                |(tensor_rank_3_i, array_i)| {
                                    tensor_rank_3_i.iter().zip(array_i.iter()).for_each(
                                        |(tensor_rank_3_ij, array_ij)| {
                                            tensor_rank_3_ij.iter().zip(array_ij.iter()).for_each(
                                                |(tensor_rank_3_ijk, array_ijk)| {
                                                    assert_eq!(tensor_rank_3_ijk, array_ijk)
                                                },
                                            )
                                        },
                                    )
                                },
                            )
                        })
                },
            )
        })
}

#[test]
fn size() {
    assert_eq!(
        std::mem::size_of::<TensorRank3List3D::<3, 1, 1, 1, 3, 4, 5>>(),
        std::mem::size_of::<[[[TensorRank3::<3, 1, 1, 1>; 3]; 4]; 5]>()
    )
}

#[test]
fn zero() {
    TensorRank3List3D::<3, 1, 1, 1, 3, 4, 5>::zero()
        .iter()
        .for_each(|tensor_rank_3_list_2d| {
            tensor_rank_3_list_2d
                .iter()
                .for_each(|tensor_rank_3_list_1d| {
                    tensor_rank_3_list_1d.iter().for_each(|tensor_rank_3| {
                        tensor_rank_3.iter().for_each(|tensor_rank_3_i| {
                            tensor_rank_3_i.iter().for_each(|tensor_rank_3_ij| {
                                tensor_rank_3_ij.iter().for_each(|tensor_rank_3_ijk| {
                                    assert_eq!(tensor_rank_3_ijk, &0.0)
                                })
                            })
                        })
                    })
                })
        })
}
