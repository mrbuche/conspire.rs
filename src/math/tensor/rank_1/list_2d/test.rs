use super::{
    super::{Tensor, TensorArray},
    TensorRank0, TensorRank1List, TensorRank1List2D,
};

fn get_array() -> [[[TensorRank0; 3]; 8]; 2] {
    [
        [
            [5.0, 0.0, 0.0],
            [5.0, 5.0, 6.0],
            [3.0, 1.0, 4.0],
            [3.0, 4.0, 2.0],
            [1.0, 0.0, 3.0],
            [1.0, 3.0, 1.0],
            [1.0, 6.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        [
            [3.0, 4.0, 2.0],
            [1.0, 0.0, 3.0],
            [5.0, 5.0, 6.0],
            [3.0, 1.0, 4.0],
            [1.0, 1.0, 1.0],
            [5.0, 0.0, 0.0],
            [1.0, 3.0, 1.0],
            [1.0, 6.0, 0.0],
        ],
    ]
}

fn get_tensor_rank_1_list_2d() -> TensorRank1List2D<3, 1, 8, 2> {
    TensorRank1List2D::new(get_array())
}

#[test]
fn from_iter() {
    let into_iterator = get_tensor_rank_1_list_2d().0.into_iter();
    let tensor_rank_1_list_2d =
        TensorRank1List2D::<3, 1, 8, 2>::from_iter(get_tensor_rank_1_list_2d().0);
    tensor_rank_1_list_2d
        .iter()
        .zip(into_iterator)
        .for_each(|(tensor_rank_1_list, array_list)| {
            tensor_rank_1_list
                .iter()
                .zip(array_list.iter())
                .for_each(|(tensor_rank_1, array)| {
                    tensor_rank_1
                        .iter()
                        .zip(array.iter())
                        .for_each(|(tensor_rank_1_i, array_i)| assert_eq!(tensor_rank_1_i, array_i))
                })
        });
}

#[test]
fn iter() {
    get_tensor_rank_1_list_2d()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_list, array_list)| {
            tensor_rank_1_list
                .iter()
                .zip(array_list.iter())
                .for_each(|(tensor_rank_1, array)| {
                    tensor_rank_1
                        .iter()
                        .zip(array.iter())
                        .for_each(|(tensor_rank_1_i, array_i)| assert_eq!(tensor_rank_1_i, array_i))
                })
        });
}

#[test]
fn iter_mut() {
    get_tensor_rank_1_list_2d()
        .iter_mut()
        .zip(get_array().iter_mut())
        .for_each(|(tensor_rank_1_list, array_list)| {
            tensor_rank_1_list
                .iter_mut()
                .zip(array_list.iter_mut())
                .for_each(|(tensor_rank_1, array)| {
                    tensor_rank_1
                        .iter_mut()
                        .zip(array.iter_mut())
                        .for_each(|(tensor_rank_1_i, array_i)| assert_eq!(tensor_rank_1_i, array_i))
                })
        });
}

#[test]
fn new() {
    get_tensor_rank_1_list_2d()
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_1_list, array_list)| {
            tensor_rank_1_list
                .iter()
                .zip(array_list.iter())
                .for_each(|(tensor_rank_1, array)| {
                    tensor_rank_1
                        .iter()
                        .zip(array.iter())
                        .for_each(|(tensor_rank_1_i, array_i)| assert_eq!(tensor_rank_1_i, array_i))
                })
        });
}

#[test]
fn size() {
    assert_eq!(
        std::mem::size_of::<TensorRank1List2D::<3, 1, 7, 8>>(),
        std::mem::size_of::<[TensorRank1List::<3, 1, 7>; 8]>()
    )
}

#[test]
fn zero() {
    TensorRank1List2D::<3, 1, 7, 8>::zero()
        .iter()
        .for_each(|tensor_rank_1_list| {
            tensor_rank_1_list.iter().for_each(|tensor_rank_1| {
                tensor_rank_1
                    .iter()
                    .for_each(|tensor_rank_1_i| assert_eq!(tensor_rank_1_i, &0.0))
            })
        });
}
