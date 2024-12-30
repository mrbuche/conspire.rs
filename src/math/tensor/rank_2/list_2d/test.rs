use super::{
    super::{Tensor, TensorArray},
    TensorRank0, TensorRank2, TensorRank2List2D,
};

fn get_array() -> [[[[TensorRank0; 3]; 3]; 2]; 2] {
    [
        [
            [[1.0, 4.0, 6.0], [7.0, 2.0, 5.0], [9.0, 8.0, 3.0]],
            [[3.0, 2.0, 3.0], [6.0, 5.0, 2.0], [4.0, 5.0, 0.0]],
        ],
        [
            [[5.0, 2.0, 9.0], [2.0, 4.0, 5.0], [1.0, 3.0, 8.0]],
            [[4.0, 3.0, 2.0], [2.0, 5.0, 4.0], [1.0, 7.0, 1.0]],
        ],
    ]
}

fn get_tensor_rank_2_list_2d() -> TensorRank2List2D<3, 1, 1, 2, 2> {
    TensorRank2List2D::new(get_array())
}

fn get_tensor_rank_2() -> TensorRank2<3, 1, 1> {
    TensorRank2::new([[1.0, 4.0, 6.0], [7.0, 2.0, 5.0], [9.0, 8.0, 3.0]])
}

fn get_tensor_rank_2_list_2d_mul_tensor_rank_2() -> TensorRank2List2D<3, 1, 1, 2, 2> {
    TensorRank2List2D::new([
        [
            [[83.0, 60.0, 44.0], [66.0, 72.0, 67.0], [92.0, 76.0, 103.0]],
            [[44.0, 40.0, 37.0], [59.0, 50.0, 67.0], [39.0, 26.0, 49.0]],
        ],
        [
            [[100.0, 96.0, 67.0], [75.0, 56.0, 47.0], [94.0, 74.0, 45.0]],
            [[43.0, 38.0, 45.0], [73.0, 50.0, 49.0], [59.0, 26.0, 44.0]],
        ],
    ])
}

#[test]
fn as_array() {
    get_tensor_rank_2_list_2d()
        .as_array()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_2_list_2d_entry, array_entry)| {
            tensor_rank_2_list_2d_entry
                .iter()
                .zip(array_entry.iter())
                .for_each(|(tensor_rank_2, array)| {
                    tensor_rank_2
                        .iter()
                        .zip(array.iter())
                        .for_each(|(tensor_rank_2_i, array_i)| {
                            tensor_rank_2_i.iter().zip(array_i.iter()).for_each(
                                |(tensor_rank_2_ij, array_ij)| {
                                    assert_eq!(tensor_rank_2_ij, array_ij)
                                },
                            )
                        })
                })
        });
}

#[test]
fn from_iter() {
    let into_iterator = get_tensor_rank_2_list_2d().0.into_iter();
    let tensor_rank_2_list_2d =
        TensorRank2List2D::<3, 1, 1, 2, 2>::from_iter(get_tensor_rank_2_list_2d().0);
    tensor_rank_2_list_2d.iter().zip(into_iterator).for_each(
        |(tensor_rank_2_list_2d_i, array_i)| {
            tensor_rank_2_list_2d_i.iter().zip(array_i.iter()).for_each(
                |(tensor_rank_2_list_2d_ij, array_ij)| {
                    tensor_rank_2_list_2d_ij
                        .iter()
                        .zip(array_ij.iter())
                        .for_each(|(tensor_rank_2_list_2d_ij_w, array_ij_w)| {
                            tensor_rank_2_list_2d_ij_w
                                .iter()
                                .zip(array_ij_w.iter())
                                .for_each(|(tensor_rank_2_list_2d_ij_ww, array_ij_ww)| {
                                    assert_eq!(tensor_rank_2_list_2d_ij_ww, array_ij_ww)
                                })
                        })
                },
            )
        },
    );
}

#[test]
fn iter() {
    get_tensor_rank_2_list_2d()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_2_list_2d_entry, array_entry)| {
            tensor_rank_2_list_2d_entry
                .iter()
                .zip(array_entry.iter())
                .for_each(|(tensor_rank_2, array)| {
                    tensor_rank_2
                        .iter()
                        .zip(array.iter())
                        .for_each(|(tensor_rank_2_i, array_i)| {
                            tensor_rank_2_i.iter().zip(array_i.iter()).for_each(
                                |(tensor_rank_2_ij, array_ij)| {
                                    assert_eq!(tensor_rank_2_ij, array_ij)
                                },
                            )
                        })
                })
        });
}

#[test]
fn iter_mut() {
    get_tensor_rank_2_list_2d()
        .iter_mut()
        .zip(get_array().iter_mut())
        .for_each(|(tensor_rank_2_list_2d_entry, array_entry)| {
            tensor_rank_2_list_2d_entry
                .iter_mut()
                .zip(array_entry.iter_mut())
                .for_each(|(tensor_rank_2, array)| {
                    tensor_rank_2.iter_mut().zip(array.iter_mut()).for_each(
                        |(tensor_rank_2_i, array_i)| {
                            tensor_rank_2_i.iter_mut().zip(array_i.iter_mut()).for_each(
                                |(tensor_rank_2_ij, array_ij)| {
                                    assert_eq!(tensor_rank_2_ij, array_ij)
                                },
                            )
                        },
                    )
                })
        });
}

#[test]
fn mul_tensor_rank_2_to_self() {
    (get_tensor_rank_2_list_2d() * get_tensor_rank_2())
        .iter()
        .zip(get_tensor_rank_2_list_2d_mul_tensor_rank_2().iter())
        .for_each(|(tensor_rank_2_list_2d_entry, res_entry)| {
            tensor_rank_2_list_2d_entry
                .iter()
                .zip(res_entry.iter())
                .for_each(|(tensor_rank_2, res)| {
                    tensor_rank_2
                        .iter()
                        .zip(res.iter())
                        .for_each(|(tensor_rank_2_i, res_i)| {
                            tensor_rank_2_i.iter().zip(res_i.iter()).for_each(
                                |(tensor_rank_2_ij, res_ij)| assert_eq!(tensor_rank_2_ij, res_ij),
                            )
                        })
                })
        });
}

#[test]
fn mul_tensor_rank_2_ref_to_self() {
    (get_tensor_rank_2_list_2d() * &get_tensor_rank_2())
        .iter()
        .zip(get_tensor_rank_2_list_2d_mul_tensor_rank_2().iter())
        .for_each(|(tensor_rank_2_list_2d_entry, res_entry)| {
            tensor_rank_2_list_2d_entry
                .iter()
                .zip(res_entry.iter())
                .for_each(|(tensor_rank_2, res)| {
                    tensor_rank_2
                        .iter()
                        .zip(res.iter())
                        .for_each(|(tensor_rank_2_i, res_i)| {
                            tensor_rank_2_i.iter().zip(res_i.iter()).for_each(
                                |(tensor_rank_2_ij, res_ij)| assert_eq!(tensor_rank_2_ij, res_ij),
                            )
                        })
                })
        });
}

#[test]
fn new() {
    get_tensor_rank_2_list_2d()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_2_list_2d_entry, array_entry)| {
            tensor_rank_2_list_2d_entry
                .iter()
                .zip(array_entry.iter())
                .for_each(|(tensor_rank_2, array)| {
                    tensor_rank_2
                        .iter()
                        .zip(array.iter())
                        .for_each(|(tensor_rank_2_i, array_i)| {
                            tensor_rank_2_i.iter().zip(array_i.iter()).for_each(
                                |(tensor_rank_2_ij, array_ij)| {
                                    assert_eq!(tensor_rank_2_ij, array_ij)
                                },
                            )
                        })
                })
        });
}

#[test]
fn size() {
    assert_eq!(
        std::mem::size_of::<TensorRank2List2D::<3, 1, 1, 8, 8>>(),
        std::mem::size_of::<[[TensorRank2::<3, 1, 1>; 8]; 8]>()
    )
}

#[test]
fn zero() {
    TensorRank2List2D::<3, 1, 1, 8, 8>::zero()
        .iter()
        .for_each(|tensor_rank_2_list_2d_entry| {
            tensor_rank_2_list_2d_entry
                .iter()
                .for_each(|tensor_rank_2| {
                    tensor_rank_2.iter().for_each(|tensor_rank_2_i| {
                        tensor_rank_2_i
                            .iter()
                            .for_each(|tensor_rank_2_ij| assert_eq!(tensor_rank_2_ij, &0.0))
                    })
                })
        });
}
