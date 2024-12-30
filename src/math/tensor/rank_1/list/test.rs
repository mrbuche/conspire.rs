use super::{Tensor, TensorArray, TensorRank0, TensorRank1, TensorRank1List, TensorRank2};

fn get_array() -> [[TensorRank0; 3]; 8] {
    [
        [5.0, 0.0, 0.0],
        [5.0, 5.0, 6.0],
        [3.0, 1.0, 4.0],
        [3.0, 4.0, 2.0],
        [1.0, 0.0, 3.0],
        [1.0, 3.0, 1.0],
        [1.0, 6.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
}

fn get_tensor_rank_1_list() -> TensorRank1List<3, 1, 8> {
    TensorRank1List::new(get_array())
}

fn get_other_tensor_rank_1_list() -> TensorRank1List<3, 1, 8> {
    TensorRank1List::new([
        [3.0, 3.0, 6.0],
        [2.0, 4.0, 3.0],
        [6.0, 2.0, 5.0],
        [5.0, 2.0, 5.0],
        [4.0, 7.0, 2.0],
        [2.0, 6.0, 6.0],
        [2.0, 3.0, 2.0],
        [3.0, 7.0, 5.0],
    ])
}

fn get_tensor_rank_1_list_add_other_tensor_rank_1_list() -> TensorRank1List<3, 1, 8> {
    TensorRank1List::new([
        [8.0, 3.0, 6.0],
        [7.0, 9.0, 9.0],
        [9.0, 3.0, 9.0],
        [8.0, 6.0, 7.0],
        [5.0, 7.0, 5.0],
        [3.0, 9.0, 7.0],
        [3.0, 9.0, 2.0],
        [4.0, 8.0, 6.0],
    ])
}

fn get_tensor_rank_1_list_sub_other_tensor_rank_1_list() -> TensorRank1List<3, 1, 8> {
    TensorRank1List::new([
        [2.0, -3.0, -6.0],
        [3.0, 1.0, 3.0],
        [-3.0, -1.0, -1.0],
        [-2.0, 2.0, -3.0],
        [-3.0, -7.0, 1.0],
        [-1.0, -3.0, -5.0],
        [-1.0, 3.0, -2.0],
        [-2.0, -6.0, -4.0],
    ])
}

fn get_tensor_rank_1_list_mul_other_tensor_rank_1_list() -> TensorRank2<3, 1, 1> {
    TensorRank2::new([[69.0, 70.0, 90.0], [57.0, 73.0, 75.0], [63.0, 70.0, 65.0]])
}

#[test]
fn add_tensor_rank_1_list_to_self() {
    (get_tensor_rank_1_list() + get_other_tensor_rank_1_list())
        .iter()
        .zip(get_tensor_rank_1_list_add_other_tensor_rank_1_list().iter())
        .for_each(|(tensor_rank_1_list_entry, add_tensor_rank_1_list_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(add_tensor_rank_1_list_entry.iter())
                .for_each(
                    |(tensor_rank_1_list_entry_i, add_tensor_rank_1_list_entry_i)| {
                        assert_eq!(tensor_rank_1_list_entry_i, add_tensor_rank_1_list_entry_i)
                    },
                )
        });
}

#[test]
fn add_tensor_rank_1_list_ref_to_self() {
    (get_tensor_rank_1_list() + &get_other_tensor_rank_1_list())
        .iter()
        .zip(get_tensor_rank_1_list_add_other_tensor_rank_1_list().iter())
        .for_each(|(tensor_rank_1_list_entry, add_tensor_rank_1_list_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(add_tensor_rank_1_list_entry.iter())
                .for_each(
                    |(tensor_rank_1_list_entry_i, add_tensor_rank_1_list_entry_i)| {
                        assert_eq!(tensor_rank_1_list_entry_i, add_tensor_rank_1_list_entry_i)
                    },
                )
        });
}

#[test]
fn add_assign_tensor_rank_1_list() {
    let mut tensor_rank_1_list = get_tensor_rank_1_list();
    tensor_rank_1_list += get_other_tensor_rank_1_list();
    tensor_rank_1_list
        .iter()
        .zip(get_tensor_rank_1_list_add_other_tensor_rank_1_list().iter())
        .for_each(|(tensor_rank_1_list_entry, add_tensor_rank_1_list_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(add_tensor_rank_1_list_entry.iter())
                .for_each(
                    |(tensor_rank_1_list_entry_i, add_tensor_rank_1_list_entry_i)| {
                        assert_eq!(tensor_rank_1_list_entry_i, add_tensor_rank_1_list_entry_i)
                    },
                )
        });
}

#[test]
fn add_assign_tensor_rank_1_list_ref() {
    let mut tensor_rank_1_list = get_tensor_rank_1_list();
    tensor_rank_1_list += &get_other_tensor_rank_1_list();
    tensor_rank_1_list
        .iter()
        .zip(get_tensor_rank_1_list_add_other_tensor_rank_1_list().iter())
        .for_each(|(tensor_rank_1_list_entry, add_tensor_rank_1_list_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(add_tensor_rank_1_list_entry.iter())
                .for_each(
                    |(tensor_rank_1_list_entry_i, add_tensor_rank_1_list_entry_i)| {
                        assert_eq!(tensor_rank_1_list_entry_i, add_tensor_rank_1_list_entry_i)
                    },
                )
        });
}

#[test]
fn as_array() {
    get_tensor_rank_1_list()
        .as_array()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_as_array_entry, array_entry)| {
            tensor_rank_1_as_array_entry
                .iter()
                .zip(array_entry.iter())
                .for_each(|(tensor_rank_1_as_array_entry_i, array_entry_i)| {
                    assert_eq!(tensor_rank_1_as_array_entry_i, array_entry_i)
                })
        });
}

#[test]
fn div_tensor_rank_0_to_self() {
    (get_tensor_rank_1_list() / 3.3)
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_list_entry, array_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(array_entry.iter())
                .for_each(|(tensor_rank_1_list_entry_i, array_entry_i)| {
                    assert_eq!(tensor_rank_1_list_entry_i, &(array_entry_i / 3.3))
                })
        });
}

#[test]
#[allow(clippy::op_ref)]
fn div_tensor_rank_0_ref_to_self() {
    (get_tensor_rank_1_list() / &3.3)
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_list_entry, array_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(array_entry.iter())
                .for_each(|(tensor_rank_1_list_entry_i, array_entry_i)| {
                    assert_eq!(tensor_rank_1_list_entry_i, &(array_entry_i / 3.3))
                })
        });
}

#[test]
fn div_assign_tensor_rank_0() {
    let mut tensor_rank_1_list = get_tensor_rank_1_list();
    tensor_rank_1_list /= 3.3;
    tensor_rank_1_list.iter().zip(get_array().iter()).for_each(
        |(tensor_rank_1_list_entry, array_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(array_entry.iter())
                .for_each(|(tensor_rank_1_list_entry_i, array_entry_i)| {
                    assert_eq!(tensor_rank_1_list_entry_i, &(array_entry_i / 3.3))
                })
        },
    );
}

#[test]
fn div_assign_tensor_rank_0_ref() {
    let mut tensor_rank_1_list = get_tensor_rank_1_list();
    tensor_rank_1_list /= &3.3;
    tensor_rank_1_list.iter().zip(get_array().iter()).for_each(
        |(tensor_rank_1_list_entry, array_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(array_entry.iter())
                .for_each(|(tensor_rank_1_list_entry_i, array_entry_i)| {
                    assert_eq!(tensor_rank_1_list_entry_i, &(array_entry_i / 3.3))
                })
        },
    );
}

#[test]
fn from_iter() {
    let into_iterator = get_tensor_rank_1_list().0;
    let tensor_rank_1_list = TensorRank1List::<3, 1, 8>::from_iter(get_tensor_rank_1_list().0);
    tensor_rank_1_list
        .iter()
        .zip(into_iterator)
        .for_each(|(tensor_rank_1_list_entry, entry)| {
            tensor_rank_1_list_entry.iter().zip(entry.iter()).for_each(
                |(tensor_rank_1_list_entry_i, entry_i)| {
                    assert_eq!(tensor_rank_1_list_entry_i, entry_i)
                },
            )
        });
}

#[test]
fn iter() {
    get_tensor_rank_1_list()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_entry, array_entry)| {
            tensor_rank_1_entry.iter().zip(array_entry.iter()).for_each(
                |(tensor_rank_1_entry_i, array_entry_i)| {
                    assert_eq!(tensor_rank_1_entry_i, array_entry_i)
                },
            )
        });
}

#[test]
fn iter_mut() {
    get_tensor_rank_1_list()
        .iter_mut()
        .zip(get_array().iter_mut())
        .for_each(|(tensor_rank_1_entry, array_entry)| {
            tensor_rank_1_entry
                .iter_mut()
                .zip(array_entry.iter_mut())
                .for_each(|(tensor_rank_1_entry_i, array_entry_i)| {
                    assert_eq!(tensor_rank_1_entry_i, array_entry_i)
                })
        });
}

#[test]
fn mul_tensor_rank_0_to_self() {
    (get_tensor_rank_1_list() * 3.3)
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_list_entry, array_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(array_entry.iter())
                .for_each(|(tensor_rank_1_list_entry_i, array_entry_i)| {
                    assert_eq!(tensor_rank_1_list_entry_i, &(array_entry_i * 3.3))
                })
        });
}

#[test]
#[allow(clippy::op_ref)]
fn mul_tensor_rank_0_ref_to_self() {
    (get_tensor_rank_1_list() * &3.3)
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_list_entry, array_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(array_entry.iter())
                .for_each(|(tensor_rank_1_list_entry_i, array_entry_i)| {
                    assert_eq!(tensor_rank_1_list_entry_i, &(array_entry_i * 3.3))
                })
        });
}

#[test]
#[allow(clippy::op_ref)]
fn mul_tensor_rank_0_ref_to_self_ref() {
    (&get_tensor_rank_1_list() * &3.3)
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_list_entry, array_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(array_entry.iter())
                .for_each(|(tensor_rank_1_list_entry_i, array_entry_i)| {
                    assert_eq!(tensor_rank_1_list_entry_i, &(array_entry_i * 3.3))
                })
        });
}

#[test]
fn mul_assign_tensor_rank_0() {
    let mut tensor_rank_1_list = get_tensor_rank_1_list();
    tensor_rank_1_list *= 3.3;
    tensor_rank_1_list.iter().zip(get_array().iter()).for_each(
        |(tensor_rank_1_list_entry, array_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(array_entry.iter())
                .for_each(|(tensor_rank_1_list_entry_i, array_entry_i)| {
                    assert_eq!(tensor_rank_1_list_entry_i, &(array_entry_i * 3.3))
                })
        },
    );
}

#[test]
fn mul_assign_tensor_rank_0_ref() {
    let mut tensor_rank_1_list = get_tensor_rank_1_list();
    tensor_rank_1_list *= &3.3;
    tensor_rank_1_list.iter().zip(get_array().iter()).for_each(
        |(tensor_rank_1_list_entry, array_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(array_entry.iter())
                .for_each(|(tensor_rank_1_list_entry_i, array_entry_i)| {
                    assert_eq!(tensor_rank_1_list_entry_i, &(array_entry_i * 3.3))
                })
        },
    );
}

#[test]
fn mul_tensor_rank_1_list_to_self() {
    (get_tensor_rank_1_list() * get_other_tensor_rank_1_list())
        .iter()
        .zip(get_tensor_rank_1_list_mul_other_tensor_rank_1_list().iter())
        .for_each(|(tensor_rank_1_list_entry, mul_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(mul_entry.iter())
                .for_each(|(tensor_rank_1_list_entry_i, mul_entry_i)| {
                    assert_eq!(tensor_rank_1_list_entry_i, mul_entry_i)
                })
        });
}

#[test]
fn mul_tensor_rank_1_list_ref_to_self() {
    (get_tensor_rank_1_list() * &get_other_tensor_rank_1_list())
        .iter()
        .zip(get_tensor_rank_1_list_mul_other_tensor_rank_1_list().iter())
        .for_each(|(tensor_rank_1_list_entry, mul_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(mul_entry.iter())
                .for_each(|(tensor_rank_1_list_entry_i, mul_entry_i)| {
                    assert_eq!(tensor_rank_1_list_entry_i, mul_entry_i)
                })
        });
}

#[test]
fn mul_tensor_rank_1_list_to_self_ref() {
    (&get_tensor_rank_1_list() * get_other_tensor_rank_1_list())
        .iter()
        .zip(get_tensor_rank_1_list_mul_other_tensor_rank_1_list().iter())
        .for_each(|(tensor_rank_1_list_entry, mul_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(mul_entry.iter())
                .for_each(|(tensor_rank_1_list_entry_i, mul_entry_i)| {
                    assert_eq!(tensor_rank_1_list_entry_i, mul_entry_i)
                })
        });
}

#[test]
fn mul_tensor_rank_1_list_ref_to_self_ref() {
    (&get_tensor_rank_1_list() * &get_other_tensor_rank_1_list())
        .iter()
        .zip(get_tensor_rank_1_list_mul_other_tensor_rank_1_list().iter())
        .for_each(|(tensor_rank_1_list_entry, mul_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(mul_entry.iter())
                .for_each(|(tensor_rank_1_list_entry_i, mul_entry_i)| {
                    assert_eq!(tensor_rank_1_list_entry_i, mul_entry_i)
                })
        });
}

#[test]
fn new() {
    get_tensor_rank_1_list()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_entry, array_entry)| {
            tensor_rank_1_entry.iter().zip(array_entry.iter()).for_each(
                |(tensor_rank_1_entry_i, array_entry_i)| {
                    assert_eq!(tensor_rank_1_entry_i, array_entry_i)
                },
            )
        });
}

#[test]
fn size() {
    assert_eq!(
        std::mem::size_of::<TensorRank1List::<3, 1, 8>>(),
        std::mem::size_of::<[TensorRank1::<3, 1>; 8]>()
    )
}

#[test]
fn sub_tensor_rank_1_list_to_self() {
    (get_tensor_rank_1_list() - get_other_tensor_rank_1_list())
        .iter()
        .zip(get_tensor_rank_1_list_sub_other_tensor_rank_1_list().iter())
        .for_each(|(tensor_rank_1_list_entry, add_tensor_rank_1_list_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(add_tensor_rank_1_list_entry.iter())
                .for_each(
                    |(tensor_rank_1_list_entry_i, add_tensor_rank_1_list_entry_i)| {
                        assert_eq!(tensor_rank_1_list_entry_i, add_tensor_rank_1_list_entry_i)
                    },
                )
        });
}

#[test]
fn sub_tensor_rank_1_list_ref_to_self() {
    (get_tensor_rank_1_list() - &get_other_tensor_rank_1_list())
        .iter()
        .zip(get_tensor_rank_1_list_sub_other_tensor_rank_1_list().iter())
        .for_each(|(tensor_rank_1_list_entry, add_tensor_rank_1_list_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(add_tensor_rank_1_list_entry.iter())
                .for_each(
                    |(tensor_rank_1_list_entry_i, add_tensor_rank_1_list_entry_i)| {
                        assert_eq!(tensor_rank_1_list_entry_i, add_tensor_rank_1_list_entry_i)
                    },
                )
        });
}

#[test]
fn sub_assign_tensor_rank_1_list() {
    let mut tensor_rank_1_list = get_tensor_rank_1_list();
    tensor_rank_1_list -= get_other_tensor_rank_1_list();
    tensor_rank_1_list
        .iter()
        .zip(get_tensor_rank_1_list_sub_other_tensor_rank_1_list().iter())
        .for_each(|(tensor_rank_1_list_entry, add_tensor_rank_1_list_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(add_tensor_rank_1_list_entry.iter())
                .for_each(
                    |(tensor_rank_1_list_entry_i, add_tensor_rank_1_list_entry_i)| {
                        assert_eq!(tensor_rank_1_list_entry_i, add_tensor_rank_1_list_entry_i)
                    },
                )
        });
}

#[test]
fn sub_assign_tensor_rank_1_list_ref() {
    let mut tensor_rank_1_list = get_tensor_rank_1_list();
    tensor_rank_1_list -= &get_other_tensor_rank_1_list();
    tensor_rank_1_list
        .iter()
        .zip(get_tensor_rank_1_list_sub_other_tensor_rank_1_list().iter())
        .for_each(|(tensor_rank_1_list_entry, add_tensor_rank_1_list_entry)| {
            tensor_rank_1_list_entry
                .iter()
                .zip(add_tensor_rank_1_list_entry.iter())
                .for_each(
                    |(tensor_rank_1_list_entry_i, add_tensor_rank_1_list_entry_i)| {
                        assert_eq!(tensor_rank_1_list_entry_i, add_tensor_rank_1_list_entry_i)
                    },
                )
        });
}

#[test]
fn zero() {
    TensorRank1List::<3, 1, 8>::zero()
        .iter()
        .for_each(|tensor_rank_1_entry| {
            tensor_rank_1_entry
                .iter()
                .for_each(|tensor_rank_1_entry_i| assert_eq!(tensor_rank_1_entry_i, &0.0))
        });
}
