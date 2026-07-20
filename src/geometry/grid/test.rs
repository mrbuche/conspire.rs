use crate::{
    geometry::grid::{Input, Output, Pixels, Voxels},
    io::{Npy, Write, write::data_array_compressed},
};

#[test]
fn round_trip_npy() {
    let data: Vec<u8> = (0..24).collect();
    let path = "target/voxels.npy";
    Voxels::new(data.clone(), [2, 3, 4])
        .write(Output::Npy(path))
        .unwrap();
    let read = Voxels::<u8>::try_from(Input::Npy(path)).unwrap();
    assert_eq!(read.data(), data);
    assert_eq!(read.nel(), &[2, 3, 4]);
}

#[test]
fn reads_c_order_with_transpose() {
    let path = "target/voxels_c.npy";
    Npy {
        data: vec![0u8, 1, 10, 11],
        shape: vec![2, 2],
        fortran_order: false,
    }
    .write(path)
    .unwrap();
    let pixels = Pixels::<u8>::try_from(Input::Npy(path)).unwrap();
    assert_eq!(pixels[[0, 0]], 0);
    assert_eq!(pixels[[1, 0]], 10);
    assert_eq!(pixels[[0, 1]], 1);
    assert_eq!(pixels[[1, 1]], 11);
}

fn assert_same_logical(a: &Voxels<u8>, b: &Voxels<u8>) {
    assert_eq!(a.nel(), b.nel());
    let [nx, ny, nz] = *a.nel();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                assert_eq!(a[[i, j, k]], b[[i, j, k]]);
            }
        }
    }
}

#[test]
fn c_order_npy_indexes_like_numpy() {
    let nel = [2usize, 3, 4];
    let [nx, ny, nz] = nel;
    let c: Vec<u32> = (0..(nx * ny * nz) as u32).collect();
    let path = "target/orient_c.npy";
    Npy {
        data: c.clone(),
        shape: vec![nx, ny, nz],
        fortran_order: false,
    }
    .write(path)
    .unwrap();
    let voxels = Voxels::<u32>::try_from(Input::Npy(path)).unwrap();
    assert_eq!(voxels.nel(), &nel);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                assert_eq!(voxels[[i, j, k]], c[i * ny * nz + j * nz + k]);
            }
        }
    }
}

#[test]
fn npy_round_trip_preserves_orientation() {
    let nel = [3usize, 2, 4];
    let source = Voxels::new((1..=24).collect(), nel);
    let path = "target/orient_rt.npy";
    source.write(Output::Npy(path)).unwrap();
    assert_same_logical(&source, &Voxels::try_from(Input::Npy(path)).unwrap());
}

#[test]
fn spn_round_trip_preserves_orientation() {
    let nel = [3usize, 2, 4];
    let source = Voxels::new((1..=24).collect(), nel);
    let path = "target/orient_rt.spn";
    source.write(Output::Spn(path)).unwrap();
    let read = Voxels::try_from(Input::Spn(path, nel.to_vec())).unwrap();
    assert_same_logical(&source, &read);
}

#[test]
fn vti_round_trip_preserves_orientation() {
    let nel = [3usize, 2, 4];
    let source = Voxels::new((1..=24).collect(), nel);
    let path = "target/orient_rt.vti";
    source.write(Output::Vti(path)).unwrap();
    assert_same_logical(&source, &Voxels::try_from(Input::Vti(path)).unwrap());
}

#[test]
fn transpose_matches_reference_on_nonuniform_grid() {
    let nel = [37usize, 5, 20];
    let [nx, ny, nz] = nel;
    let total = nx * ny * nz;
    let c: Vec<u32> = (0..total as u32).collect();
    let path = "target/voxels_nonuniform_c.npy";
    Npy {
        data: c.clone(),
        shape: vec![nx, ny, nz],
        fortran_order: false,
    }
    .write(path)
    .unwrap();
    let read = Voxels::<u32>::try_from(Input::Npy(path)).unwrap();
    let mut want = vec![0u32; total];
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                want[i + j * nx + k * nx * ny] = c[i * ny * nz + j * nz + k];
            }
        }
    }
    assert_eq!(read.data(), c.as_slice());
    assert_eq!(read.data_col_major().as_ref(), want.as_slice());
    assert_eq!(read.nel(), &nel);
}

#[test]
fn round_trip_vti() {
    let data: Vec<u8> = (0..24).collect();
    let path = "target/voxels.vti";
    Voxels::new(data.clone(), [2, 3, 4])
        .write(Output::Vti(path))
        .unwrap();
    let contents = std::fs::read_to_string(path).unwrap();
    assert!(contents.contains("type=\"ImageData\""));
    assert!(contents.contains("WholeExtent=\"0 2 0 3 0 4\""));
    let read = Voxels::<u8>::try_from(Input::Vti(path)).unwrap();
    assert_eq!(read.data(), data);
    assert_eq!(read.nel(), &[2, 3, 4]);
}

#[test]
fn round_trip_vti_2d() {
    let data: Vec<u16> = (0..6).collect();
    let path = "target/pixels.vti";
    Pixels::new(data.clone(), [2, 3])
        .write(Output::Vti(path))
        .unwrap();
    let contents = std::fs::read_to_string(path).unwrap();
    assert!(contents.contains("WholeExtent=\"0 2 0 3 0 0\""));
    let read = Pixels::<u16>::try_from(Input::Vti(path)).unwrap();
    assert_eq!(read.data(), data);
    assert_eq!(read.nel(), &[2, 3]);
}

#[test]
fn round_trip_vti_compressed() {
    let data: Vec<u8> = (0..24).collect();
    let voxels = Voxels::new(data.clone(), [2, 3, 4]);
    let bytes: Vec<u8> = voxels.data_col_major().iter().copied().collect();
    let path = "target/voxels_compressed.vti";
    std::fs::write(
        path,
        format!(
            "<?xml version=\"1.0\"?>\n\
             <VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" \
             header_type=\"UInt64\" compressor=\"vtkZLibDataCompressor\">\n\
             <ImageData WholeExtent=\"0 2 0 3 0 4\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n\
             <Piece Extent=\"0 2 0 3 0 4\">\n\
             <CellData Scalars=\"data\">\n\
             <DataArray type=\"UInt8\" Name=\"data\" NumberOfComponents=\"1\" format=\"binary\">{}</DataArray>\n\
             </CellData>\n\
             </Piece>\n\
             </ImageData>\n\
             </VTKFile>\n",
            data_array_compressed(&bytes)
        ),
    )
    .unwrap();
    let read = Voxels::<u8>::try_from(Input::Vti(path)).unwrap();
    assert_eq!(read.data(), data);
    assert_eq!(read.nel(), &[2, 3, 4]);
}

#[test]
fn dimension_mismatch_errors() {
    let path = "target/pixels.npy";
    Pixels::new(vec![0u8; 6], [2, 3])
        .write(Output::Npy(path))
        .unwrap();
    assert!(Voxels::<u8>::try_from(Input::Npy(path)).is_err());
}

#[test]
fn round_trip_spn() {
    let data: Vec<u8> = (0..24).collect();
    let path = "target/voxels.spn";
    Voxels::new(data.clone(), [2, 3, 4])
        .write(Output::Spn(path))
        .unwrap();
    let read = Voxels::<u8>::try_from(Input::Spn(path, vec![2, 3, 4])).unwrap();
    assert_eq!(read.data(), data);
    assert_eq!(read.nel(), &[2, 3, 4]);
}

#[test]
fn spn_wrong_count_errors() {
    let data: Vec<u8> = (0..24).collect();
    let path = "target/voxels_bad.spn";
    Voxels::new(data, [2, 3, 4])
        .write(Output::Spn(path))
        .unwrap();
    assert!(Voxels::<u8>::try_from(Input::Spn(path, vec![2, 3, 3])).is_err());
}

#[test]
fn extract_sub_grid() {
    let data: Vec<u8> = (0..24).collect();
    let voxels = Voxels::new(data, [2, 3, 4]);
    let sub = voxels.extract([0..1, 1..3, 0..2]);
    assert_eq!(sub.nel(), &[1, 2, 2]);
    assert_eq!(sub[[0, 0, 0]], voxels[[0, 1, 0]]);
    assert_eq!(sub[[0, 1, 0]], voxels[[0, 2, 0]]);
    assert_eq!(sub[[0, 0, 1]], voxels[[0, 1, 1]]);
    assert_eq!(sub[[0, 1, 1]], voxels[[0, 2, 1]]);
}

#[test]
fn diff_marks_differences() {
    let a = Voxels::new(vec![0u8, 1, 2, 3], [2, 2, 1]);
    let b = Voxels::new(vec![0u8, 9, 2, 9], [2, 2, 1]);
    let d = a.diff(&b);
    assert_eq!(d.data(), vec![0u8, 1, 0, 1]);
    assert_eq!(d.nel(), &[2, 2, 1]);
}
