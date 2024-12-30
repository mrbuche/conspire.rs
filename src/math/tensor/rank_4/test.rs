use super::{
    super::{test::ErrorTensor, Tensor, TensorArray},
    ContractAllIndicesWithFirstIndicesOf, ContractFirstSecondIndicesWithSecondIndicesOf,
    ContractFirstThirdFourthIndicesWithFirstIndicesOf,
    ContractSecondFourthIndicesWithFirstIndicesOf, ContractSecondIndexWithFirstIndexOf,
    ContractThirdFourthIndicesWithFirstSecondIndicesOf, Rank2, TensorRank0, TensorRank1,
    TensorRank2, TensorRank3, TensorRank4,
};
use crate::{ABS_TOL, REL_TOL};

fn get_array() -> [[[[TensorRank0; 3]; 3]; 3]; 3] {
    [
        [
            [[4.0, 2.0, 4.0], [1.0, 4.0, 3.0], [2.0, 4.0, 4.0]],
            [[2.0, 2.0, 2.0], [3.0, 1.0, 1.0], [1.0, 4.0, 2.0]],
            [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [1.0, 1.0, 0.0]],
        ],
        [
            [[2.0, 4.0, 2.0], [1.0, 2.0, 3.0], [3.0, 3.0, 2.0]],
            [[1.0, 1.0, 1.0], [4.0, 2.0, 1.0], [1.0, 4.0, 1.0]],
            [[2.0, 2.0, 4.0], [3.0, 3.0, 1.0], [0.0, 3.0, 3.0]],
        ],
        [
            [[0.0, 1.0, 4.0], [3.0, 3.0, 3.0], [4.0, 4.0, 0.0]],
            [[2.0, 3.0, 1.0], [1.0, 2.0, 0.0], [2.0, 2.0, 4.0]],
            [[3.0, 4.0, 1.0], [2.0, 1.0, 2.0], [4.0, 4.0, 1.0]],
        ],
    ]
}

fn get_tensor_rank_4() -> TensorRank4<3, 1, 1, 1, 1> {
    TensorRank4::new(get_array())
}

fn get_other_tensor_rank_4() -> TensorRank4<3, 1, 1, 1, 1> {
    TensorRank4::new([
        [
            [[2.0, 2.0, 4.0], [0.0, 0.0, 1.0], [1.0, 3.0, 3.0]],
            [[0.0, 3.0, 1.0], [0.0, 0.0, 1.0], [4.0, 2.0, 1.0]],
            [[3.0, 0.0, 1.0], [2.0, 0.0, 3.0], [4.0, 4.0, 2.0]],
        ],
        [
            [[4.0, 4.0, 0.0], [2.0, 1.0, 1.0], [0.0, 0.0, 4.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 3.0], [0.0, 1.0, 1.0]],
            [[4.0, 2.0, 3.0], [4.0, 2.0, 4.0], [3.0, 0.0, 4.0]],
        ],
        [
            [[1.0, 3.0, 2.0], [0.0, 0.0, 0.0], [2.0, 4.0, 2.0]],
            [[2.0, 2.0, 2.0], [4.0, 1.0, 2.0], [4.0, 2.0, 2.0]],
            [[1.0, 2.0, 3.0], [4.0, 0.0, 1.0], [4.0, 2.0, 1.0]],
        ],
    ])
}

fn get_other_tensor_rank_4_add_tensor_rank_4() -> TensorRank4<3, 1, 1, 1, 1> {
    TensorRank4::new([
        [
            [[6.0, 4.0, 8.0], [1.0, 4.0, 4.0], [3.0, 7.0, 7.0]],
            [[2.0, 5.0, 3.0], [3.0, 1.0, 2.0], [5.0, 6.0, 3.0]],
            [[4.0, 2.0, 4.0], [4.0, 2.0, 6.0], [5.0, 5.0, 2.0]],
        ],
        [
            [[6.0, 8.0, 2.0], [3.0, 3.0, 4.0], [3.0, 3.0, 6.0]],
            [[1.0, 2.0, 1.0], [5.0, 3.0, 4.0], [1.0, 5.0, 2.0]],
            [[6.0, 4.0, 7.0], [7.0, 5.0, 5.0], [3.0, 3.0, 7.0]],
        ],
        [
            [[1.0, 4.0, 6.0], [3.0, 3.0, 3.0], [6.0, 8.0, 2.0]],
            [[4.0, 5.0, 3.0], [5.0, 3.0, 2.0], [6.0, 4.0, 6.0]],
            [[4.0, 6.0, 4.0], [6.0, 1.0, 3.0], [8.0, 6.0, 2.0]],
        ],
    ])
}

fn get_other_tensor_rank_4_sub_tensor_rank_4() -> TensorRank4<3, 1, 1, 1, 1> {
    TensorRank4::new([
        [
            [[2.0, 0.0, 0.0], [1.0, 4.0, 2.0], [1.0, 1.0, 1.0]],
            [[2.0, -1.0, 1.0], [3.0, 1.0, 0.0], [-3.0, 2.0, 1.0]],
            [[-2.0, 2.0, 2.0], [0.0, 2.0, 0.0], [-3.0, -3.0, -2.0]],
        ],
        [
            [[-2.0, 0.0, 2.0], [-1.0, 1.0, 2.0], [3.0, 3.0, -2.0]],
            [[1.0, 0.0, 1.0], [3.0, 1.0, -2.0], [1.0, 3.0, 0.0]],
            [[-2.0, 0.0, 1.0], [-1.0, 1.0, -3.0], [-3.0, 3.0, -1.0]],
        ],
        [
            [[-1.0, -2.0, 2.0], [3.0, 3.0, 3.0], [2.0, 0.0, -2.0]],
            [[0.0, 1.0, -1.0], [-3.0, 1.0, -2.0], [-2.0, 0.0, 2.0]],
            [[2.0, 2.0, -2.0], [-2.0, 1.0, 1.0], [0.0, 2.0, 0.0]],
        ],
    ])
}

fn get_tensor_rank_4_mul_tensor_rank_2() -> TensorRank4<3, 1, 1, 1, 1> {
    TensorRank4::new([
        [
            [[54.0, 52.0, 46.0], [56.0, 36.0, 35.0], [66.0, 48.0, 44.0]],
            [[34.0, 28.0, 28.0], [19.0, 22.0, 26.0], [47.0, 28.0, 32.0]],
            [[42.0, 32.0, 25.0], [43.0, 36.0, 31.0], [8.0, 6.0, 11.0]],
        ],
        [
            [[48.0, 32.0, 38.0], [42.0, 32.0, 25.0], [42.0, 34.0, 39.0]],
            [[17.0, 14.0, 14.0], [27.0, 28.0, 37.0], [38.0, 20.0, 29.0]],
            [[52.0, 44.0, 34.0], [33.0, 26.0, 36.0], [48.0, 30.0, 24.0]],
        ],
        [
            [[43.0, 34.0, 17.0], [51.0, 42.0, 42.0], [32.0, 24.0, 44.0]],
            [[32.0, 22.0, 30.0], [15.0, 8.0, 16.0], [52.0, 44.0, 34.0]],
            [[40.0, 28.0, 41.0], [27.0, 26.0, 23.0], [41.0, 32.0, 47.0]],
        ],
    ])
}

fn get_tensor_rank_4_contract_all_indices_with_first_indices_of_tensor_rank_2(
) -> TensorRank4<3, 1, 1, 1, 1> {
    TensorRank4::new([
        [
            [
                [19307.0, 10466.0, 16156.0],
                [19363.0, 10450.0, 15890.0],
                [15929.0, 8584.0, 12812.0],
            ],
            [
                [17809.0, 9650.0, 14910.0],
                [17978.0, 9694.0, 14766.0],
                [14953.0, 8050.0, 12068.0],
            ],
            [
                [7839.0, 4266.0, 6596.0],
                [7683.0, 4170.0, 6346.0],
                [6125.0, 3328.0, 4892.0],
            ],
        ],
        [
            [
                [16048.0, 8772.0, 13456.0],
                [16316.0, 8892.0, 13508.0],
                [13668.0, 7448.0, 11168.0],
            ],
            [
                [14622.0, 7996.0, 12348.0],
                [14892.0, 8096.0, 12408.0],
                [12538.0, 6792.0, 10300.0],
            ],
            [
                [6776.0, 3700.0, 5600.0],
                [6796.0, 3740.0, 5572.0],
                [5596.0, 3128.0, 4528.0],
            ],
        ],
        [
            [
                [15427.0, 8342.0, 12520.0],
                [15491.0, 8374.0, 12356.0],
                [13053.0, 7128.0, 10332.0],
            ],
            [
                [13896.0, 7522.0, 11296.0],
                [13953.0, 7544.0, 11114.0],
                [11825.0, 6454.0, 9314.0],
            ],
            [
                [6779.0, 3678.0, 5448.0],
                [6871.0, 3750.0, 5532.0],
                [5777.0, 3200.0, 4692.0],
            ],
        ],
    ])
}

fn get_tensor_rank_4_contract_first_second_indices_with_second_indices_of_tensor_rank_2(
) -> TensorRank4<3, 1, 1, 1, 1> {
    TensorRank4::new([
        [
            [
                [153.0, 222.0, 207.0],
                [197.0, 192.0, 166.0],
                [223.0, 295.0, 150.0],
            ],
            [
                [216.0, 332.0, 326.0],
                [315.0, 325.0, 261.0],
                [363.0, 474.0, 258.0],
            ],
            [
                [138.0, 216.0, 204.0],
                [217.0, 225.0, 157.0],
                [237.0, 320.0, 198.0],
            ],
        ],
        [
            [
                [226.0, 257.0, 300.0],
                [230.0, 258.0, 243.0],
                [239.0, 353.0, 201.0],
            ],
            [
                [374.0, 393.0, 485.0],
                [374.0, 437.0, 363.0],
                [389.0, 620.0, 394.0],
            ],
            [
                [258.0, 263.0, 313.0],
                [266.0, 293.0, 213.0],
                [255.0, 446.0, 308.0],
            ],
        ],
        [
            [
                [322.0, 367.0, 436.0],
                [346.0, 380.0, 337.0],
                [271.0, 499.0, 313.0],
            ],
            [
                [540.0, 585.0, 653.0],
                [562.0, 611.0, 527.0],
                [481.0, 892.0, 556.0],
            ],
            [
                [368.0, 387.0, 401.0],
                [414.0, 399.0, 325.0],
                [331.0, 658.0, 398.0],
            ],
        ],
    ])
}

fn get_tensor_rank_4_contract_first_third_fourth_indices_with_first_indices_of_tensor_rank_2(
) -> TensorRank4<3, 1, 1, 1, 1> {
    TensorRank4::new([
        [
            [
                [3382.0, 3251.0, 2580.0],
                [3081.0, 2979.0, 2424.0],
                [1330.0, 1247.0, 900.0],
            ],
            [
                [2283.0, 2202.0, 1839.0],
                [2261.0, 2158.0, 1754.0],
                [719.0, 706.0, 623.0],
            ],
            [
                [2904.0, 2844.0, 2376.0],
                [2692.0, 2612.0, 2162.0],
                [1176.0, 1164.0, 976.0],
            ],
        ],
        [
            [
                [2992.0, 2864.0, 2248.0],
                [2738.0, 2620.0, 2088.0],
                [1160.0, 1120.0, 840.0],
            ],
            [
                [1920.0, 1848.0, 1518.0],
                [1906.0, 1820.0, 1458.0],
                [632.0, 616.0, 534.0],
            ],
            [
                [2244.0, 2232.0, 1900.0],
                [2038.0, 2024.0, 1734.0],
                [940.0, 920.0, 756.0],
            ],
        ],
        [
            [
                [3090.0, 2901.0, 2186.0],
                [2859.0, 2682.0, 2033.0],
                [1190.0, 1141.0, 870.0],
            ],
            [
                [1929.0, 1902.0, 1633.0],
                [1880.0, 1804.0, 1488.0],
                [641.0, 670.0, 617.0],
            ],
            [
                [2280.0, 2211.0, 1773.0],
                [2022.0, 1944.0, 1549.0],
                [1008.0, 987.0, 785.0],
            ],
        ],
    ])
}

fn get_tensor_rank_4_contract_second_index_with_first_index_of_tensor_rank_2(
) -> TensorRank4<3, 1, 1, 1, 1> {
    TensorRank4::new([
        [
            [[27.0, 34.0, 45.0], [40.0, 29.0, 37.0], [18.0, 41.0, 18.0]],
            [[28.0, 28.0, 44.0], [26.0, 34.0, 38.0], [18.0, 32.0, 20.0]],
            [[37.0, 28.0, 43.0], [27.0, 35.0, 32.0], [20.0, 47.0, 34.0]],
        ],
        [
            [[27.0, 29.0, 45.0], [56.0, 43.0, 19.0], [10.0, 58.0, 36.0]],
            [[26.0, 34.0, 42.0], [36.0, 36.0, 22.0], [14.0, 44.0, 34.0]],
            [[23.0, 35.0, 29.0], [35.0, 31.0, 26.0], [23.0, 47.0, 26.0]],
        ],
        [
            [[41.0, 58.0, 20.0], [28.0, 26.0, 21.0], [54.0, 54.0, 37.0]],
            [[28.0, 42.0, 26.0], [30.0, 24.0, 28.0], [52.0, 52.0, 16.0]],
            [[19.0, 33.0, 32.0], [29.0, 31.0, 24.0], [46.0, 46.0, 23.0]],
        ],
    ])
}

fn get_tensor_rank_4_contract_third_fourth_indices_with_first_second_indices_of_tensor_rank_2(
) -> TensorRank2<3, 1, 1> {
    TensorRank2::new([
        [128.0, 97.0, 77.0],
        [113.0, 92.0, 99.0],
        [138.0, 77.0, 122.0],
    ])
}

fn get_tensor_rank_4_contract_second_fourth_indices_with_first_indices_of_tensors_rank_1(
) -> TensorRank2<3, 1, 1> {
    TensorRank2::new([
        [206.0, 196.0, 151.0],
        [196.0, 195.0, 198.0],
        [201.0, 148.0, 246.0],
    ])
}

fn get_tensor_rank_1() -> TensorRank1<3, 1> {
    TensorRank1::new([1.0, 2.0, 3.0])
}

fn get_other_tensor_rank_1() -> TensorRank1<3, 1> {
    TensorRank1::new([4.0, 5.0, 6.0])
}

fn get_tensor_rank_2() -> TensorRank2<3, 1, 1> {
    TensorRank2::new([[1.0, 4.0, 6.0], [7.0, 2.0, 5.0], [9.0, 8.0, 3.0]])
}

fn get_other_tensor_rank_2() -> TensorRank2<3, 1, 1> {
    TensorRank2::new([[3.0, 2.0, 3.0], [6.0, 5.0, 2.0], [4.0, 5.0, 0.0]])
}

fn get_other_other_tensor_rank_2() -> TensorRank2<3, 1, 1> {
    TensorRank2::new([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0], [3.0, 3.0, 2.0]])
}

fn get_other_other_other_tensor_rank_2() -> TensorRank2<3, 1, 1> {
    TensorRank2::new([[3.0, 2.0, 4.0], [1.0, 0.0, 0.0], [3.0, 2.0, 2.0]])
}

#[test]
fn add_tensor_rank_4_to_self() {
    (get_tensor_rank_4() + get_other_tensor_rank_4())
        .iter()
        .zip(get_other_tensor_rank_4_add_tensor_rank_4().iter())
        .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
            tensor_rank_4_i
                .iter()
                .zip(res_tensor_rank_4_i.iter())
                .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                    tensor_rank_4_ij
                        .iter()
                        .zip(res_tensor_rank_4_ij.iter())
                        .for_each(|(tensor_rank_4_ijk, res_tensor_rank_4_ijk)| {
                            tensor_rank_4_ijk
                                .iter()
                                .zip(res_tensor_rank_4_ijk.iter())
                                .for_each(|(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)
                                })
                        })
                })
        });
}

#[test]
fn add_tensor_rank_4_ref_to_self() {
    (get_tensor_rank_4() + &get_other_tensor_rank_4())
        .iter()
        .zip(get_other_tensor_rank_4_add_tensor_rank_4().iter())
        .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
            tensor_rank_4_i
                .iter()
                .zip(res_tensor_rank_4_i.iter())
                .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                    tensor_rank_4_ij
                        .iter()
                        .zip(res_tensor_rank_4_ij.iter())
                        .for_each(|(tensor_rank_4_ijk, res_tensor_rank_4_ijk)| {
                            tensor_rank_4_ijk
                                .iter()
                                .zip(res_tensor_rank_4_ijk.iter())
                                .for_each(|(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)
                                })
                        })
                })
        });
}

#[test]
fn add_tensor_rank_4_to_self_ref() {
    (&get_tensor_rank_4() + get_other_tensor_rank_4())
        .iter()
        .zip(get_other_tensor_rank_4_add_tensor_rank_4().iter())
        .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
            tensor_rank_4_i
                .iter()
                .zip(res_tensor_rank_4_i.iter())
                .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                    tensor_rank_4_ij
                        .iter()
                        .zip(res_tensor_rank_4_ij.iter())
                        .for_each(|(tensor_rank_4_ijk, res_tensor_rank_4_ijk)| {
                            tensor_rank_4_ijk
                                .iter()
                                .zip(res_tensor_rank_4_ijk.iter())
                                .for_each(|(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)
                                })
                        })
                })
        });
}

#[test]
fn add_assign_tensor_rank_4() {
    let mut tensor_rank_4 = get_tensor_rank_4();
    tensor_rank_4 += get_other_tensor_rank_4();
    tensor_rank_4
        .iter()
        .zip(get_other_tensor_rank_4_add_tensor_rank_4().iter())
        .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
            tensor_rank_4_i
                .iter()
                .zip(res_tensor_rank_4_i.iter())
                .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                    tensor_rank_4_ij
                        .iter()
                        .zip(res_tensor_rank_4_ij.iter())
                        .for_each(|(tensor_rank_4_ijk, res_tensor_rank_4_ijk)| {
                            tensor_rank_4_ijk
                                .iter()
                                .zip(res_tensor_rank_4_ijk.iter())
                                .for_each(|(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)
                                })
                        })
                })
        });
}

#[test]
fn add_assign_tensor_rank_4_ref() {
    let mut tensor_rank_4 = get_tensor_rank_4();
    tensor_rank_4 += &get_other_tensor_rank_4();
    tensor_rank_4
        .iter()
        .zip(get_other_tensor_rank_4_add_tensor_rank_4().iter())
        .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
            tensor_rank_4_i
                .iter()
                .zip(res_tensor_rank_4_i.iter())
                .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                    tensor_rank_4_ij
                        .iter()
                        .zip(res_tensor_rank_4_ij.iter())
                        .for_each(|(tensor_rank_4_ijk, res_tensor_rank_4_ijk)| {
                            tensor_rank_4_ijk
                                .iter()
                                .zip(res_tensor_rank_4_ijk.iter())
                                .for_each(|(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)
                                })
                        })
                })
        });
}

#[test]
fn as_array() {
    get_tensor_rank_4()
        .as_array()
        .iter()
        .zip(get_array().iter())
        .for_each(|(get_tensor_rank_4_as_array_i, array_i)| {
            get_tensor_rank_4_as_array_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(get_tensor_rank_4_as_array_ij, array_ij)| {
                    get_tensor_rank_4_as_array_ij
                        .iter()
                        .zip(array_ij.iter())
                        .for_each(|(get_tensor_rank_4_as_array_ijk, array_ijk)| {
                            get_tensor_rank_4_as_array_ijk
                                .iter()
                                .zip(array_ijk.iter())
                                .for_each(|(get_tensor_rank_4_as_array_ijkl, array_ijkl)| {
                                    assert_eq!(get_tensor_rank_4_as_array_ijkl, array_ijkl)
                                })
                        })
                })
        });
}

#[test]
fn contract_all_indices_with_first_indices_of() {
    (get_tensor_rank_4().contract_all_indices_with_first_indices_of(
        &get_tensor_rank_2(),
        &get_other_tensor_rank_2(),
        &get_other_other_tensor_rank_2(),
        &get_other_other_other_tensor_rank_2(),
    ))
    .iter()
    .zip(get_tensor_rank_4_contract_all_indices_with_first_indices_of_tensor_rank_2().iter())
    .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
        tensor_rank_4_i
            .iter()
            .zip(res_tensor_rank_4_i.iter())
            .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                tensor_rank_4_ij
                    .iter()
                    .zip(res_tensor_rank_4_ij.iter())
                    .for_each(|(tensor_rank_4_ijk, res_tensor_rank_4_ijk)| {
                        tensor_rank_4_ijk
                            .iter()
                            .zip(res_tensor_rank_4_ijk.iter())
                            .for_each(|(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)| {
                                assert_eq!(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)
                            })
                    })
            })
    });
}

#[test]
fn contract_first_second_indices_with_second_indices_of() {
    (get_tensor_rank_4().contract_first_second_indices_with_second_indices_of(
        &get_tensor_rank_2(),
        &get_other_tensor_rank_2(),
    ))
    .iter()
    .zip(
        get_tensor_rank_4_contract_first_second_indices_with_second_indices_of_tensor_rank_2()
            .iter(),
    )
    .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
        tensor_rank_4_i
            .iter()
            .zip(res_tensor_rank_4_i.iter())
            .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                tensor_rank_4_ij
                    .iter()
                    .zip(res_tensor_rank_4_ij.iter())
                    .for_each(|(tensor_rank_4_ijk, res_tensor_rank_4_ijk)| {
                        tensor_rank_4_ijk
                            .iter()
                            .zip(res_tensor_rank_4_ijk.iter())
                            .for_each(|(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)| {
                                assert_eq!(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)
                            })
                    })
            })
    });
}

#[test]
fn contract_first_third_fourth_indices_with_first_indices_of() {
    (get_tensor_rank_4().contract_first_third_fourth_indices_with_first_indices_of(
        &get_tensor_rank_2(),
        &get_other_tensor_rank_2(),
        &get_other_other_tensor_rank_2(),
    ))
    .iter()
    .zip(
        get_tensor_rank_4_contract_first_third_fourth_indices_with_first_indices_of_tensor_rank_2()
            .iter(),
    )
    .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
        tensor_rank_4_i
            .iter()
            .zip(res_tensor_rank_4_i.iter())
            .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                tensor_rank_4_ij
                    .iter()
                    .zip(res_tensor_rank_4_ij.iter())
                    .for_each(|(tensor_rank_4_ijk, res_tensor_rank_4_ijk)| {
                        tensor_rank_4_ijk
                            .iter()
                            .zip(res_tensor_rank_4_ijk.iter())
                            .for_each(|(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)| {
                                assert_eq!(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)
                            })
                    })
            })
    });
}

#[test]
fn contract_second_index_with_first_index_of() {
    (get_tensor_rank_4().contract_second_index_with_first_index_of(&get_tensor_rank_2()))
        .iter()
        .zip(get_tensor_rank_4_contract_second_index_with_first_index_of_tensor_rank_2().iter())
        .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
            tensor_rank_4_i
                .iter()
                .zip(res_tensor_rank_4_i.iter())
                .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                    tensor_rank_4_ij
                        .iter()
                        .zip(res_tensor_rank_4_ij.iter())
                        .for_each(|(tensor_rank_4_ijk, res_tensor_rank_4_ijk)| {
                            tensor_rank_4_ijk
                                .iter()
                                .zip(res_tensor_rank_4_ijk.iter())
                                .for_each(|(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)
                                })
                        })
                })
        });
}

#[test]
fn contract_second_fourth_indices_with_first_indices_of() {
    (get_tensor_rank_4().contract_second_fourth_indices_with_first_indices_of(
        &get_tensor_rank_1(),
        &get_other_tensor_rank_1(),
    ))
    .iter()
    .zip(
        get_tensor_rank_4_contract_second_fourth_indices_with_first_indices_of_tensors_rank_1()
            .iter(),
    )
    .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
        tensor_rank_4_i
            .iter()
            .zip(res_tensor_rank_4_i.iter())
            .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                assert_eq!(tensor_rank_4_ij, res_tensor_rank_4_ij)
            })
    })
}

#[test]
fn contract_third_fourth_indices_with_first_second_indices_of() {
    (get_tensor_rank_4()
        .contract_third_fourth_indices_with_first_second_indices_of(&get_tensor_rank_2()))
    .iter()
    .zip(
        get_tensor_rank_4_contract_third_fourth_indices_with_first_second_indices_of_tensor_rank_2(
        )
        .iter(),
    )
    .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
        tensor_rank_4_i
            .iter()
            .zip(res_tensor_rank_4_i.iter())
            .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                assert_eq!(tensor_rank_4_ij, res_tensor_rank_4_ij)
            })
    })
}

#[test]
fn div_tensor_rank_0_to_self() {
    (get_tensor_rank_4() / 3.3)
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_4_i, array_i)| {
            tensor_rank_4_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(tensor_rank_4_ij, array_ij)| {
                    tensor_rank_4_ij.iter().zip(array_ij.iter()).for_each(
                        |(tensor_rank_4_ijk, array_ijk)| {
                            tensor_rank_4_ijk.iter().zip(array_ijk.iter()).for_each(
                                |(tensor_rank_4_ijkl, array_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, &(array_ijkl / 3.3))
                                },
                            )
                        },
                    )
                })
        });
}

#[test]
#[allow(clippy::op_ref)]
fn div_tensor_rank_0_ref_to_self() {
    (get_tensor_rank_4() / &3.3)
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_4_i, array_i)| {
            tensor_rank_4_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(tensor_rank_4_ij, array_ij)| {
                    tensor_rank_4_ij.iter().zip(array_ij.iter()).for_each(
                        |(tensor_rank_4_ijk, array_ijk)| {
                            tensor_rank_4_ijk.iter().zip(array_ijk.iter()).for_each(
                                |(tensor_rank_4_ijkl, array_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, &(array_ijkl / 3.3))
                                },
                            )
                        },
                    )
                })
        });
}

#[test]
fn div_assign_tensor_rank_0() {
    let mut tensor_rank_4 = get_tensor_rank_4();
    tensor_rank_4 /= 3.3;
    tensor_rank_4
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_4_i, array_i)| {
            tensor_rank_4_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(tensor_rank_4_ij, array_ij)| {
                    tensor_rank_4_ij.iter().zip(array_ij.iter()).for_each(
                        |(tensor_rank_4_ijk, array_ijk)| {
                            tensor_rank_4_ijk.iter().zip(array_ijk.iter()).for_each(
                                |(tensor_rank_4_ijkl, array_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, &(array_ijkl / 3.3))
                                },
                            )
                        },
                    )
                })
        });
}

#[test]
fn div_assign_tensor_rank_0_ref() {
    let mut tensor_rank_4 = get_tensor_rank_4();
    tensor_rank_4 /= &3.3;
    tensor_rank_4
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_4_i, array_i)| {
            tensor_rank_4_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(tensor_rank_4_ij, array_ij)| {
                    tensor_rank_4_ij.iter().zip(array_ij.iter()).for_each(
                        |(tensor_rank_4_ijk, array_ijk)| {
                            tensor_rank_4_ijk.iter().zip(array_ijk.iter()).for_each(
                                |(tensor_rank_4_ijkl, array_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, &(array_ijkl / 3.3))
                                },
                            )
                        },
                    )
                })
        });
}

#[test]
fn dyad_ij_kl() {
    let tensor_a = get_tensor_rank_2();
    let tensor_b = get_other_tensor_rank_2();
    TensorRank4::<3, 1, 1, 1, 1>::dyad_ij_kl(&tensor_a, &tensor_b)
        .iter()
        .zip(tensor_a.iter())
        .for_each(|(tensor_rank_4_i, tensor_a_i)| {
            tensor_rank_4_i.iter().zip(tensor_a_i.iter()).for_each(
                |(tensor_rank_4_ij, tensor_a_ij)| {
                    tensor_rank_4_ij.iter().zip(tensor_b.iter()).for_each(
                        |(tensor_rank_4_ijk, tensor_b_k)| {
                            tensor_rank_4_ijk.iter().zip(tensor_b_k.iter()).for_each(
                                |(tensor_rank_4_ijkl, tensor_b_kl)| {
                                    assert_eq!(tensor_rank_4_ijkl, &(tensor_a_ij * tensor_b_kl))
                                },
                            )
                        },
                    )
                },
            )
        });
}

#[test]
fn dyad_ik_jl() {
    let tensor_a = get_tensor_rank_2();
    let tensor_b = get_other_tensor_rank_2();
    TensorRank4::<3, 1, 1, 1, 1>::dyad_ik_jl(&tensor_a, &tensor_b)
        .iter()
        .zip(tensor_a.iter())
        .for_each(|(tensor_rank_4_i, tensor_a_i)| {
            tensor_rank_4_i.iter().zip(tensor_b.iter()).for_each(
                |(tensor_rank_4_ij, tensor_b_j)| {
                    tensor_rank_4_ij.iter().zip(tensor_a_i.iter()).for_each(
                        |(tensor_rank_4_ijk, tensor_a_ik)| {
                            tensor_rank_4_ijk.iter().zip(tensor_b_j.iter()).for_each(
                                |(tensor_rank_4_ijkl, tensor_b_jl)| {
                                    assert_eq!(tensor_rank_4_ijkl, &(tensor_a_ik * tensor_b_jl))
                                },
                            )
                        },
                    )
                },
            )
        });
}

#[test]
fn dyad_il_jk() {
    let tensor_a = get_tensor_rank_2();
    let tensor_b = get_other_tensor_rank_2();
    TensorRank4::<3, 1, 1, 1, 1>::dyad_il_jk(&tensor_a, &tensor_b)
        .iter()
        .zip(tensor_a.iter())
        .for_each(|(tensor_rank_4_i, tensor_a_i)| {
            tensor_rank_4_i.iter().zip(tensor_b.iter()).for_each(
                |(tensor_rank_4_ij, tensor_b_j)| {
                    tensor_rank_4_ij.iter().zip(tensor_b_j.iter()).for_each(
                        |(tensor_rank_4_ijk, tensor_b_jk)| {
                            tensor_rank_4_ijk.iter().zip(tensor_a_i.iter()).for_each(
                                |(tensor_rank_4_ijkl, tensor_a_il)| {
                                    assert_eq!(tensor_rank_4_ijkl, &(tensor_a_il * tensor_b_jk))
                                },
                            )
                        },
                    )
                },
            )
        });
}

#[test]
fn dyad_il_kj() {
    let tensor_a = get_tensor_rank_2();
    let tensor_b = get_other_tensor_rank_2();
    TensorRank4::<3, 1, 1, 1, 1>::dyad_il_kj(&tensor_a, &tensor_b)
        .iter()
        .zip(tensor_a.iter())
        .for_each(|(tensor_rank_4_i, tensor_a_i)| {
            tensor_rank_4_i
                .iter()
                .zip(tensor_b.transpose().iter())
                .for_each(|(tensor_rank_4_ij, tensor_b_j)| {
                    tensor_rank_4_ij.iter().zip(tensor_b_j.iter()).for_each(
                        |(tensor_rank_4_ijk, tensor_b_jk)| {
                            tensor_rank_4_ijk.iter().zip(tensor_a_i.iter()).for_each(
                                |(tensor_rank_4_ijkl, tensor_a_il)| {
                                    assert_eq!(tensor_rank_4_ijkl, &(tensor_a_il * tensor_b_jk))
                                },
                            )
                        },
                    )
                })
        });
}

#[test]
fn error() {
    let a = get_tensor_rank_4();
    let b = get_other_tensor_rank_4();
    assert_eq!(a.error(&a, &ABS_TOL, &REL_TOL), None);
    assert_eq!(a.error(&b, &ABS_TOL, &REL_TOL), Some(67));
}

#[test]
fn from_iter() {
    let into_iterator = get_tensor_rank_4().0.into_iter();
    let tensor_rank_4 = TensorRank4::<3, 1, 1, 1, 1>::from_iter(get_tensor_rank_4().0);
    tensor_rank_4
        .iter()
        .zip(into_iterator)
        .for_each(|(tensor_rank_4_i, value_i)| {
            tensor_rank_4_i
                .iter()
                .zip(value_i.iter())
                .for_each(|(tensor_rank_4_ij, value_ij)| {
                    tensor_rank_4_ij.iter().zip(value_ij.iter()).for_each(
                        |(tensor_rank_4_ijk, value_ijk)| {
                            tensor_rank_4_ijk.iter().zip(value_ijk.iter()).for_each(
                                |(tensor_rank_4_ijkl, value_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, value_ijkl)
                                },
                            )
                        },
                    )
                })
        });
}

#[test]
fn iter() {
    get_tensor_rank_4().iter().zip(get_array().iter()).for_each(
        |(get_tensor_rank_4_i, array_i)| {
            get_tensor_rank_4_i.iter().zip(array_i.iter()).for_each(
                |(get_tensor_rank_4_ij, array_ij)| {
                    get_tensor_rank_4_ij.iter().zip(array_ij.iter()).for_each(
                        |(get_tensor_rank_4_ijk, array_ijk)| {
                            get_tensor_rank_4_ijk.iter().zip(array_ijk.iter()).for_each(
                                |(get_tensor_rank_4_ijkl, array_ijkl)| {
                                    assert_eq!(get_tensor_rank_4_ijkl, array_ijkl)
                                },
                            )
                        },
                    )
                },
            )
        },
    );
}

#[test]
fn iter_mut() {
    get_tensor_rank_4()
        .iter_mut()
        .zip(get_array().iter_mut())
        .for_each(|(get_tensor_rank_4_i, array_i)| {
            get_tensor_rank_4_i
                .iter_mut()
                .zip(array_i.iter_mut())
                .for_each(|(get_tensor_rank_4_ij, array_ij)| {
                    get_tensor_rank_4_ij
                        .iter_mut()
                        .zip(array_ij.iter_mut())
                        .for_each(|(get_tensor_rank_4_ijk, array_ijk)| {
                            get_tensor_rank_4_ijk
                                .iter_mut()
                                .zip(array_ijk.iter_mut())
                                .for_each(|(get_tensor_rank_4_ijkl, array_ijkl)| {
                                    assert_eq!(get_tensor_rank_4_ijkl, array_ijkl)
                                })
                        })
                })
        });
}

#[test]
fn mul_tensor_rank_0_to_self() {
    (get_tensor_rank_4() * 3.3)
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_4_i, array_i)| {
            tensor_rank_4_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(tensor_rank_4_ij, array_ij)| {
                    tensor_rank_4_ij.iter().zip(array_ij.iter()).for_each(
                        |(tensor_rank_4_ijk, array_ijk)| {
                            tensor_rank_4_ijk.iter().zip(array_ijk.iter()).for_each(
                                |(tensor_rank_4_ijkl, array_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, &(array_ijkl * 3.3))
                                },
                            )
                        },
                    )
                })
        });
}

#[test]
#[allow(clippy::op_ref)]
fn mul_tensor_rank_0_ref_to_self() {
    (get_tensor_rank_4() * &3.3)
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_4_i, array_i)| {
            tensor_rank_4_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(tensor_rank_4_ij, array_ij)| {
                    tensor_rank_4_ij.iter().zip(array_ij.iter()).for_each(
                        |(tensor_rank_4_ijk, array_ijk)| {
                            tensor_rank_4_ijk.iter().zip(array_ijk.iter()).for_each(
                                |(tensor_rank_4_ijkl, array_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, &(array_ijkl * 3.3))
                                },
                            )
                        },
                    )
                })
        });
}

#[test]
fn mul_assign_tensor_rank_0() {
    let mut tensor_rank_4 = get_tensor_rank_4();
    tensor_rank_4 *= 3.3;
    tensor_rank_4
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_4_i, array_i)| {
            tensor_rank_4_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(tensor_rank_4_ij, array_ij)| {
                    tensor_rank_4_ij.iter().zip(array_ij.iter()).for_each(
                        |(tensor_rank_4_ijk, array_ijk)| {
                            tensor_rank_4_ijk.iter().zip(array_ijk.iter()).for_each(
                                |(tensor_rank_4_ijkl, array_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, &(array_ijkl * 3.3))
                                },
                            )
                        },
                    )
                })
        });
}

#[test]
fn mul_assign_tensor_rank_0_ref() {
    let mut tensor_rank_4 = get_tensor_rank_4();
    tensor_rank_4 *= &3.3;
    tensor_rank_4
        .iter()
        .zip(get_array().iter())
        .for_each(|(tensor_rank_4_i, array_i)| {
            tensor_rank_4_i
                .iter()
                .zip(array_i.iter())
                .for_each(|(tensor_rank_4_ij, array_ij)| {
                    tensor_rank_4_ij.iter().zip(array_ij.iter()).for_each(
                        |(tensor_rank_4_ijk, array_ijk)| {
                            tensor_rank_4_ijk.iter().zip(array_ijk.iter()).for_each(
                                |(tensor_rank_4_ijkl, array_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, &(array_ijkl * 3.3))
                                },
                            )
                        },
                    )
                })
        });
}

#[test]
fn mul_tensor_rank_2_to_self() {
    (get_tensor_rank_4() * get_tensor_rank_2())
        .iter()
        .zip(get_tensor_rank_4_mul_tensor_rank_2().iter())
        .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
            tensor_rank_4_i
                .iter()
                .zip(res_tensor_rank_4_i.iter())
                .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                    tensor_rank_4_ij
                        .iter()
                        .zip(res_tensor_rank_4_ij.iter())
                        .for_each(|(tensor_rank_4_ijk, res_tensor_rank_4_ijk)| {
                            tensor_rank_4_ijk
                                .iter()
                                .zip(res_tensor_rank_4_ijk.iter())
                                .for_each(|(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)
                                })
                        })
                })
        });
}

#[test]
fn mul_tensor_rank_2_ref_to_self() {
    (get_tensor_rank_4() * &get_tensor_rank_2())
        .iter()
        .zip(get_tensor_rank_4_mul_tensor_rank_2().iter())
        .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
            tensor_rank_4_i
                .iter()
                .zip(res_tensor_rank_4_i.iter())
                .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                    tensor_rank_4_ij
                        .iter()
                        .zip(res_tensor_rank_4_ij.iter())
                        .for_each(|(tensor_rank_4_ijk, res_tensor_rank_4_ijk)| {
                            tensor_rank_4_ijk
                                .iter()
                                .zip(res_tensor_rank_4_ijk.iter())
                                .for_each(|(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)
                                })
                        })
                })
        });
}

#[test]
fn new() {
    get_tensor_rank_4().iter().zip(get_array().iter()).for_each(
        |(get_tensor_rank_4_i, array_i)| {
            get_tensor_rank_4_i.iter().zip(array_i.iter()).for_each(
                |(get_tensor_rank_4_ij, array_ij)| {
                    get_tensor_rank_4_ij.iter().zip(array_ij.iter()).for_each(
                        |(get_tensor_rank_4_ijk, array_ijk)| {
                            get_tensor_rank_4_ijk.iter().zip(array_ijk.iter()).for_each(
                                |(get_tensor_rank_4_ijkl, array_ijkl)| {
                                    assert_eq!(get_tensor_rank_4_ijkl, array_ijkl)
                                },
                            )
                        },
                    )
                },
            )
        },
    );
}

#[test]
fn size() {
    assert_eq!(
        std::mem::size_of::<TensorRank4::<3, 1, 1, 1, 1>>(),
        std::mem::size_of::<[TensorRank3::<3, 1, 1, 1>; 3]>()
    )
}

#[test]
fn sub_tensor_rank_4_to_self() {
    (get_tensor_rank_4() - get_other_tensor_rank_4())
        .iter()
        .zip(get_other_tensor_rank_4_sub_tensor_rank_4().iter())
        .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
            tensor_rank_4_i
                .iter()
                .zip(res_tensor_rank_4_i.iter())
                .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                    tensor_rank_4_ij
                        .iter()
                        .zip(res_tensor_rank_4_ij.iter())
                        .for_each(|(tensor_rank_4_ijk, res_tensor_rank_4_ijk)| {
                            tensor_rank_4_ijk
                                .iter()
                                .zip(res_tensor_rank_4_ijk.iter())
                                .for_each(|(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)
                                })
                        })
                })
        });
}

#[test]
fn sub_tensor_rank_4_ref_to_self() {
    (get_tensor_rank_4() - &get_other_tensor_rank_4())
        .iter()
        .zip(get_other_tensor_rank_4_sub_tensor_rank_4().iter())
        .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
            tensor_rank_4_i
                .iter()
                .zip(res_tensor_rank_4_i.iter())
                .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                    tensor_rank_4_ij
                        .iter()
                        .zip(res_tensor_rank_4_ij.iter())
                        .for_each(|(tensor_rank_4_ijk, res_tensor_rank_4_ijk)| {
                            tensor_rank_4_ijk
                                .iter()
                                .zip(res_tensor_rank_4_ijk.iter())
                                .for_each(|(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)
                                })
                        })
                })
        });
}

#[test]
fn sub_assign_tensor_rank_4() {
    let mut tensor_rank_4 = get_tensor_rank_4();
    tensor_rank_4 -= get_other_tensor_rank_4();
    tensor_rank_4
        .iter()
        .zip(get_other_tensor_rank_4_sub_tensor_rank_4().iter())
        .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
            tensor_rank_4_i
                .iter()
                .zip(res_tensor_rank_4_i.iter())
                .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                    tensor_rank_4_ij
                        .iter()
                        .zip(res_tensor_rank_4_ij.iter())
                        .for_each(|(tensor_rank_4_ijk, res_tensor_rank_4_ijk)| {
                            tensor_rank_4_ijk
                                .iter()
                                .zip(res_tensor_rank_4_ijk.iter())
                                .for_each(|(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)
                                })
                        })
                })
        });
}

#[test]
fn sub_assign_tensor_rank_4_ref() {
    let mut tensor_rank_4 = get_tensor_rank_4();
    tensor_rank_4 -= &get_other_tensor_rank_4();
    tensor_rank_4
        .iter()
        .zip(get_other_tensor_rank_4_sub_tensor_rank_4().iter())
        .for_each(|(tensor_rank_4_i, res_tensor_rank_4_i)| {
            tensor_rank_4_i
                .iter()
                .zip(res_tensor_rank_4_i.iter())
                .for_each(|(tensor_rank_4_ij, res_tensor_rank_4_ij)| {
                    tensor_rank_4_ij
                        .iter()
                        .zip(res_tensor_rank_4_ij.iter())
                        .for_each(|(tensor_rank_4_ijk, res_tensor_rank_4_ijk)| {
                            tensor_rank_4_ijk
                                .iter()
                                .zip(res_tensor_rank_4_ijk.iter())
                                .for_each(|(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)| {
                                    assert_eq!(tensor_rank_4_ijkl, res_tensor_rank_4_ijkl)
                                })
                        })
                })
        });
}

#[test]
fn zero() {
    TensorRank4::<3, 1, 1, 1, 1>::zero()
        .iter()
        .for_each(|tensor_rank_4_i| {
            tensor_rank_4_i.iter().for_each(|tensor_rank_4_ij| {
                tensor_rank_4_ij.iter().for_each(|tensor_rank_4_ijk| {
                    tensor_rank_4_ijk
                        .iter()
                        .for_each(|tensor_rank_4_ijkl| assert_eq!(tensor_rank_4_ijkl, &0.0))
                })
            })
        });
}
