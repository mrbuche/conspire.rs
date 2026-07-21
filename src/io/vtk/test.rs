use super::{
    invalid,
    read::{
        DataArray, Encoding, attribute, bits, data_array, data_arrays, decode, encoding,
        find_data_array, floats, integers, parse, region, tag, unbase64,
    },
    unsupported,
    write::{self, Compression, base64, data_array_compressed},
};
use crate::io::deflate::zlib_encode;

#[test]
fn invalid_sets_invalid_data_kind() {
    let error = invalid("boom".into());
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert_eq!(error.to_string(), "boom");
}

#[test]
fn unsupported_sets_unsupported_kind() {
    let error = unsupported("nope");
    assert_eq!(error.kind(), std::io::ErrorKind::Unsupported);
    assert_eq!(error.to_string(), "nope");
}

#[test]
fn encoding_defaults_to_uncompressed_four_byte_header() {
    let result = encoding("<VTKFile type=\"UnstructuredGrid\">").unwrap();
    assert_eq!(result.header_bytes, 4);
    assert!(!result.compressed);
}

#[test]
fn encoding_reads_compressed_eight_byte_header() {
    let result =
        encoding("<VTKFile compressor=\"vtkZLibDataCompressor\" header_type=\"UInt64\">").unwrap();
    assert_eq!(result.header_bytes, 8);
    assert!(result.compressed);
}

#[test]
fn encoding_reads_uncompressed_four_byte_header_explicitly() {
    let result = encoding("<VTKFile header_type=\"UInt32\">").unwrap();
    assert_eq!(result.header_bytes, 4);
    assert!(!result.compressed);
}

#[test]
fn encoding_rejects_unsupported_compressor() {
    assert!(encoding("<VTKFile compressor=\"SomeOtherCompressor\">").is_err());
}

#[test]
fn encoding_rejects_unsupported_header_type() {
    assert!(encoding("<VTKFile header_type=\"UInt16\">").is_err());
}

#[test]
fn tag_finds_open_tag_through_closing_bracket() {
    let result = tag("prefix <VTKFile type=\"foo\"> rest", "<VTKFile").unwrap();
    assert_eq!(result, "<VTKFile type=\"foo\"");
}

#[test]
fn tag_errors_when_open_missing() {
    assert!(tag("no tags here", "<VTKFile").is_err());
}

#[test]
fn tag_errors_when_unterminated() {
    assert!(tag("<VTKFile type=\"foo\"", "<VTKFile").is_err());
}

#[test]
fn region_finds_bounds_between_open_and_close() {
    let result = region("before <Foo>middle</Foo> after", "Foo").unwrap();
    assert_eq!(result, "<Foo>middle");
}

#[test]
fn region_errors_when_open_missing() {
    assert!(region("<Bar></Bar>", "Foo").is_err());
}

#[test]
fn region_errors_when_close_missing() {
    assert!(region("<Foo>middle", "Foo").is_err());
}

#[test]
fn attribute_extracts_quoted_value() {
    let result = attribute("<Foo bar=\"baz\" qux=\"1\">", "bar").unwrap();
    assert_eq!(result, "baz");
}

#[test]
fn attribute_returns_none_when_missing() {
    assert!(attribute("<Foo bar=\"baz\">", "missing").is_none());
}

#[test]
fn attribute_returns_none_when_unterminated() {
    assert!(attribute("<Foo bar=\"baz", "bar").is_none());
}

#[test]
fn data_arrays_parses_multiple_entries() {
    let region = "<DataArray type=\"Float64\" Name=\"points\" format=\"ascii\">1 2 3</DataArray><DataArray type=\"Int64\" NumberOfTuples=\"2\">4 5</DataArray>";
    let arrays = data_arrays(region).unwrap();
    assert_eq!(arrays.len(), 2);
    assert_eq!(arrays[0].name, Some("points"));
    assert_eq!(arrays[0].data_type, "Float64");
    assert_eq!(arrays[0].format, "ascii");
    assert_eq!(arrays[0].text, "1 2 3");
    assert_eq!(arrays[1].name, None);
    assert_eq!(arrays[1].tuples, 2);
}

#[test]
fn data_arrays_errors_on_unterminated_tag() {
    assert!(data_arrays("<DataArray type=\"Float64\"").is_err());
}

#[test]
fn data_arrays_errors_on_unclosed_element() {
    assert!(data_arrays("<DataArray type=\"Float64\">1 2 3").is_err());
}

#[test]
fn data_arrays_errors_when_type_missing() {
    assert!(data_arrays("<DataArray Name=\"points\">1 2 3</DataArray>").is_err());
}

#[test]
fn data_array_finds_first_when_name_is_none() {
    let region = "<DataArray type=\"Float64\" Name=\"a\">1</DataArray>";
    let result = data_array(region, None).unwrap();
    assert_eq!(result.name, Some("a"));
}

#[test]
fn data_array_finds_by_name() {
    let region = "<DataArray type=\"Float64\" Name=\"a\">1</DataArray><DataArray type=\"Float64\" Name=\"b\">2</DataArray>";
    let result = data_array(region, Some("b")).unwrap();
    assert_eq!(result.text, "2");
}

#[test]
fn find_data_array_errors_when_missing() {
    let arrays: Vec<DataArray> = Vec::new();
    assert!(find_data_array(&arrays, Some("missing")).is_err());
}

#[test]
fn find_data_array_errors_when_none_and_empty() {
    let arrays: Vec<DataArray> = Vec::new();
    assert!(find_data_array(&arrays, None).is_err());
}

fn ascii_array<'a>(data_type: &'a str, text: &'a str) -> DataArray<'a> {
    DataArray {
        name: None,
        data_type,
        format: "ascii",
        tuples: 0,
        text,
    }
}

fn binary_array<'a>(data_type: &'a str, text: &'a str, tuples: usize) -> DataArray<'a> {
    DataArray {
        name: None,
        data_type,
        format: "binary",
        tuples,
        text,
    }
}

fn uncompressed_encoding(header_bytes: usize) -> Encoding {
    Encoding {
        header_bytes,
        compressed: false,
    }
}

fn compressed_encoding(header_bytes: usize) -> Encoding {
    Encoding {
        header_bytes,
        compressed: true,
    }
}

#[test]
fn floats_parses_ascii() {
    let array = ascii_array("Float64", "1.5 2.5 3.5");
    assert_eq!(
        floats(&array, &uncompressed_encoding(4)).unwrap(),
        vec![1.5, 2.5, 3.5]
    );
}

#[test]
fn floats_ascii_errors_on_malformed_token() {
    let array = ascii_array("Float64", "not-a-number");
    assert!(floats(&array, &uncompressed_encoding(4)).is_err());
}

#[test]
fn floats_decodes_binary_float64() {
    let data: Vec<u8> = [1.0f64, -2.0f64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let encoded = write::data_array(&data);
    let array = binary_array("Float64", &encoded, 2);
    assert_eq!(
        floats(&array, &uncompressed_encoding(8)).unwrap(),
        vec![1.0, -2.0]
    );
}

#[test]
fn floats_decodes_binary_float32() {
    let data: Vec<u8> = [1.5f32, -2.5f32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let encoded = write::data_array(&data);
    let array = binary_array("Float32", &encoded, 2);
    assert_eq!(
        floats(&array, &uncompressed_encoding(8)).unwrap(),
        vec![1.5, -2.5]
    );
}

#[test]
fn floats_errors_on_unsupported_binary_type() {
    let encoded = write::data_array(&[0u8; 8]);
    let array = binary_array("Float16", &encoded, 1);
    assert!(floats(&array, &uncompressed_encoding(8)).is_err());
}

#[test]
fn integers_parses_ascii() {
    let array = ascii_array("Int64", "1 -2 3");
    assert_eq!(
        integers(&array, &uncompressed_encoding(4)).unwrap(),
        vec![1, -2, 3]
    );
}

#[test]
fn integers_ascii_errors_on_malformed_token() {
    let array = ascii_array("Int64", "nope");
    assert!(integers(&array, &uncompressed_encoding(4)).is_err());
}

#[test]
fn integers_decodes_binary_int64() {
    let data: Vec<u8> = [1i64, -2i64].iter().flat_map(|v| v.to_le_bytes()).collect();
    let encoded = write::data_array(&data);
    let array = binary_array("Int64", &encoded, 2);
    assert_eq!(
        integers(&array, &uncompressed_encoding(8)).unwrap(),
        vec![1, -2]
    );
}

#[test]
fn integers_decodes_binary_uint64() {
    let data: Vec<u8> = [7u64].iter().flat_map(|v| v.to_le_bytes()).collect();
    let encoded = write::data_array(&data);
    let array = binary_array("UInt64", &encoded, 1);
    assert_eq!(
        integers(&array, &uncompressed_encoding(8)).unwrap(),
        vec![7]
    );
}

#[test]
fn integers_decodes_binary_int32() {
    let data: Vec<u8> = [1i32, -2i32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let encoded = write::data_array(&data);
    let array = binary_array("Int32", &encoded, 2);
    assert_eq!(
        integers(&array, &uncompressed_encoding(8)).unwrap(),
        vec![1, -2]
    );
}

#[test]
fn integers_decodes_binary_uint32() {
    let data: Vec<u8> = [9u32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let encoded = write::data_array(&data);
    let array = binary_array("UInt32", &encoded, 1);
    assert_eq!(
        integers(&array, &uncompressed_encoding(8)).unwrap(),
        vec![9]
    );
}

#[test]
fn integers_decodes_binary_int8_and_uint8() {
    let data = vec![1u8, 250u8];
    let encoded = write::data_array(&data);
    let int8 = binary_array("Int8", &encoded, 2);
    assert_eq!(
        integers(&int8, &uncompressed_encoding(8)).unwrap(),
        vec![1, 250]
    );
    let uint8 = binary_array("UInt8", &encoded, 2);
    assert_eq!(
        integers(&uint8, &uncompressed_encoding(8)).unwrap(),
        vec![1, 250]
    );
}

#[test]
fn integers_errors_on_unsupported_binary_type() {
    let encoded = write::data_array(&[0u8; 8]);
    let array = binary_array("Float64", &encoded, 1);
    assert!(integers(&array, &uncompressed_encoding(8)).is_err());
}

#[test]
fn bits_parses_ascii() {
    let array = ascii_array("Bit", "1 0 1");
    assert_eq!(
        bits(&array, &uncompressed_encoding(4)).unwrap(),
        vec![1, 0, 1]
    );
}

#[test]
fn bits_ascii_errors_on_malformed_token() {
    let array = ascii_array("Bit", "nope");
    assert!(bits(&array, &uncompressed_encoding(4)).is_err());
}

#[test]
fn bits_decodes_binary() {
    let data = vec![0b1010_0000u8];
    let encoded = write::data_array(&data);
    let array = binary_array("Bit", &encoded, 4);
    assert_eq!(
        bits(&array, &uncompressed_encoding(8)).unwrap(),
        vec![1, 0, 1, 0]
    );
}

#[test]
fn decode_errors_on_non_binary_format() {
    let array = ascii_array("Float64", "1 2 3");
    assert!(decode(&array, &uncompressed_encoding(4)).is_err());
}

#[test]
fn decode_errors_when_shorter_than_header() {
    let encoded = base64(&[1, 2, 3]);
    let array = binary_array("Float64", &encoded, 0);
    assert!(decode(&array, &uncompressed_encoding(8)).is_err());
}

#[test]
fn decode_round_trips_uncompressed_with_four_byte_header() {
    let data = vec![1u8, 2, 3, 4, 5];
    let mut buffer = (data.len() as u32).to_le_bytes().to_vec();
    buffer.extend_from_slice(&data);
    let encoded = base64(&buffer);
    let array = binary_array("Int8", &encoded, data.len());
    assert_eq!(decode(&array, &uncompressed_encoding(4)).unwrap(), data);
}

#[test]
fn decode_round_trips_compressed_single_block() {
    let data = vec![42u8; 100];
    let encoded = data_array_compressed(&data);
    let array = binary_array("Int8", &encoded, data.len());
    assert_eq!(decode(&array, &compressed_encoding(8)).unwrap(), data);
}

#[test]
fn decode_round_trips_compressed_multiple_blocks() {
    let data: Vec<u8> = (0..100_000).map(|i| (i % 251) as u8).collect();
    let encoded = data_array_compressed(&data);
    let array = binary_array("Int8", &encoded, data.len());
    assert_eq!(decode(&array, &compressed_encoding(8)).unwrap(), data);
}

#[test]
fn decode_round_trips_compressed_empty() {
    let data: Vec<u8> = Vec::new();
    let encoded = data_array_compressed(&data);
    let array = binary_array("Int8", &encoded, 0);
    assert_eq!(decode(&array, &compressed_encoding(8)).unwrap(), data);
}

#[test]
fn decode_round_trips_compressed_with_four_byte_header() {
    let data = vec![7u8; 50];
    let compressed = zlib_encode(&data);
    let mut buffer = Vec::new();
    buffer.extend_from_slice(&1u32.to_le_bytes());
    buffer.extend_from_slice(&(32768u32).to_le_bytes());
    buffer.extend_from_slice(&(data.len() as u32).to_le_bytes());
    buffer.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
    buffer.extend_from_slice(&compressed);
    let encoded = base64(&buffer);
    let array = binary_array("Int8", &encoded, data.len());
    assert_eq!(decode(&array, &compressed_encoding(4)).unwrap(), data);
}

#[test]
fn decode_errors_on_truncated_compressed_header() {
    let data = vec![3u8; 50];
    let encoded = data_array_compressed(&data);
    let mut bytes = unbase64(&encoded);
    bytes.truncate(3);
    let corrupted = base64(&bytes);
    let array = binary_array("Int8", &corrupted, 0);
    assert!(decode(&array, &compressed_encoding(8)).is_err());
}

#[test]
fn decode_errors_on_truncated_compressed_block() {
    let data: Vec<u8> = (0..40_000).map(|i| (i % 253) as u8).collect();
    let encoded = data_array_compressed(&data);
    let mut bytes = unbase64(&encoded);
    bytes.pop();
    let corrupted = base64(&bytes);
    let array = binary_array("Int8", &corrupted, data.len());
    assert!(decode(&array, &compressed_encoding(8)).is_err());
}

#[test]
#[should_panic]
fn decode_with_unsupported_header_size_panics() {
    let encoded = base64(&[0, 0, 0, 0]);
    let array = binary_array("Int8", &encoded, 0);
    let _ = decode(
        &array,
        &Encoding {
            header_bytes: 2,
            compressed: true,
        },
    );
}

#[test]
fn parse_succeeds_for_valid_token() {
    assert_eq!(parse::<i32>("42").unwrap(), 42);
}

#[test]
fn parse_errors_for_invalid_token() {
    assert!(parse::<i32>("nope").is_err());
}

#[test]
fn unbase64_decodes_known_vector() {
    assert_eq!(unbase64("TWFu"), b"Man");
}

#[test]
fn unbase64_ignores_invalid_characters() {
    assert_eq!(unbase64("TW\nFu"), b"Man");
}

#[test]
fn unbase64_handles_empty_input() {
    assert_eq!(unbase64(""), Vec::<u8>::new());
}

#[test]
fn compression_as_ref_on_returns_inner_path() {
    let compression = Compression::On("foo.vtu");
    assert_eq!(compression.as_ref(), std::path::Path::new("foo.vtu"));
}

#[test]
fn compression_as_ref_off_returns_inner_path() {
    let compression = Compression::Off("bar.vtu");
    assert_eq!(compression.as_ref(), std::path::Path::new("bar.vtu"));
}

#[test]
fn base64_round_trips_all_padding_cases() {
    for len in 0..12 {
        let data: Vec<u8> = (0..len).collect();
        assert_eq!(unbase64(&base64(&data)), data);
    }
}
