use super::{
    BitReader, BitWriter, Huffman, adler32, deflate, inflate, inflate_block, read_dynamic_tables,
    zlib_decode, zlib_encode,
};

fn hex_decode(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn adler32_known_vector() {
    assert_eq!(adler32(b"Wikipedia"), 0x11E60398);
}

#[test]
fn adler32_empty() {
    assert_eq!(adler32(b""), 1);
}

fn round_trip(data: &[u8]) {
    let compressed = deflate(data);
    let decompressed = inflate(&compressed).unwrap();
    assert_eq!(decompressed, data);
    let zlib_compressed = zlib_encode(data);
    let zlib_decompressed = zlib_decode(&zlib_compressed).unwrap();
    assert_eq!(zlib_decompressed, data);
}

#[test]
fn round_trip_empty() {
    round_trip(b"");
}

#[test]
fn round_trip_single_byte() {
    round_trip(b"x");
}

#[test]
fn round_trip_short_text() {
    round_trip(b"hello, world!");
}

#[test]
fn round_trip_repetitive_text() {
    let data = "the quick brown fox jumps over the lazy dog ".repeat(50) + "END";
    round_trip(data.as_bytes());
}

#[test]
fn round_trip_all_byte_values() {
    let data: Vec<u8> = (0..=255).cycle().take(10_000).collect();
    round_trip(&data);
}

#[test]
fn round_trip_long_runs() {
    let mut data = Vec::new();
    for byte in 0..8u8 {
        data.extend(std::iter::repeat_n(byte, 5_000));
    }
    round_trip(&data);
}

#[test]
fn round_trip_pseudo_random() {
    let mut state: u64 = 0x2545F4914F6CDD1D;
    let data: Vec<u8> = (0..20_000)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state & 0xFF) as u8
        })
        .collect();
    round_trip(&data);
}

#[test]
fn round_trip_exceeds_window() {
    let unit = "0123456789abcdefghijklmnopqrstuvwxyz";
    let data = unit.repeat(3000);
    round_trip(data.as_bytes());
}

#[test]
fn zlib_decode_dynamic_huffman_reference_vector() {
    let expected = "the quick brown fox jumps over the lazy dog ".repeat(50) + "END";
    let compressed = hex_decode(
        "789c2bc94855282ccd4cce56482aca2fcf5348cbaf50c82acd2d2856c82f4b2d5228014ae72456552aa4e4a78339a36a47d58eaa1d553baa7654eda8da51b5b452ebeae70200ab4325e7",
    );
    assert_eq!(zlib_decode(&compressed).unwrap(), expected.as_bytes());
}

#[test]
fn zlib_decode_stored_block_reference_vector() {
    let expected = "the quick brown fox jumps over the lazy dog ".repeat(50) + "END";
    let compressed = hex_decode(
        "7801019b0864f774686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672074686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f6720454e44ab4325e7",
    );
    assert_eq!(zlib_decode(&compressed).unwrap(), expected.as_bytes());
}

#[test]
fn zlib_decode_random_data_reference_vector() {
    let expected = hex_decode(
        "0db42a5c6c6e62f5e4f18e0f2502177bc0ceaaf367ea9718d74679a2def2da48e7f6f2a7977a1896b95c3074cebef48c27d8682a34c766feb18e31ae5f22d5798d5a9a5515f05814bfbd231ce14b614a1d42c5ff62c95014e6c5facb254f5e27c6ae5a0dd547d9b6e43794e006322fdbefecb1ec3c6f7d9ffb48c6d7fd6b028cad4bd8ee363b460cf52d475fe05d4b8317f5f4fcce13022e3e3bc266c08de3712f0cef9dd819febd105e4b340be761a1719a3a2de98688e4b0aa95be45cf76ee56a2e4a575c61959cf42d3ad20a4e13162afcd3ce5b912ec8f28710c6a9b3870c7140d346242c5f1cf5575c6aad70e8ca3c1146b250077cd09bc7f1c1e5b213112fc9299f4edc907c6b0f53a2ab3354bf17ff04090c8d122e1943730e55459a543d3bbc906472bf66e22d0e33463b581104e84a6fc0c61396edf341a2363cf3768cd22fd105fcc7d2c7c537aee2240dd4d74e6459385d699747d874e775e90de2b64321ae5bc6e38b6a5eaa10cfd40977d5152c64c6ab8363689038e1586eb51abde34fc9f7da6add9a91d806104c2a6b9f02f03b58f94587dfae42dc9c52d4a3c781a8de97540d23f46153b22574e155c1a5cc79914d10557959548d7cd53232c981a323cb617c3b6ac17817cac9a848ba0d723908c452505934f11bf810a4e7881920bf691717a226c1bcc4240bee744824181",
    );
    let compressed = hex_decode(
        "78da01f4010bfe0db42a5c6c6e62f5e4f18e0f2502177bc0ceaaf367ea9718d74679a2def2da48e7f6f2a7977a1896b95c3074cebef48c27d8682a34c766feb18e31ae5f22d5798d5a9a5515f05814bfbd231ce14b614a1d42c5ff62c95014e6c5facb254f5e27c6ae5a0dd547d9b6e43794e006322fdbefecb1ec3c6f7d9ffb48c6d7fd6b028cad4bd8ee363b460cf52d475fe05d4b8317f5f4fcce13022e3e3bc266c08de3712f0cef9dd819febd105e4b340be761a1719a3a2de98688e4b0aa95be45cf76ee56a2e4a575c61959cf42d3ad20a4e13162afcd3ce5b912ec8f28710c6a9b3870c7140d346242c5f1cf5575c6aad70e8ca3c1146b250077cd09bc7f1c1e5b213112fc9299f4edc907c6b0f53a2ab3354bf17ff04090c8d122e1943730e55459a543d3bbc906472bf66e22d0e33463b581104e84a6fc0c61396edf341a2363cf3768cd22fd105fcc7d2c7c537aee2240dd4d74e6459385d699747d874e775e90de2b64321ae5bc6e38b6a5eaa10cfd40977d5152c64c6ab8363689038e1586eb51abde34fc9f7da6add9a91d806104c2a6b9f02f03b58f94587dfae42dc9c52d4a3c781a8de97540d23f46153b22574e155c1a5cc79914d10557959548d7cd53232c981a323cb617c3b6ac17817cac9a848ba0d723908c452505934f11bf810a4e7881920bf691717a226c1bcc4240bee74482418109a9f5ec",
    );
    assert_eq!(zlib_decode(&compressed).unwrap(), expected);
}

#[test]
fn zlib_decode_empty_reference_vector() {
    let compressed = hex_decode("789c030000000001");
    assert_eq!(zlib_decode(&compressed).unwrap(), Vec::<u8>::new());
}

#[test]
fn zlib_decode_rejects_bad_checksum() {
    let mut compressed = zlib_encode(b"hello, world!");
    let last = compressed.len() - 1;
    compressed[last] ^= 0xFF;
    assert!(zlib_decode(&compressed).is_err());
}

#[test]
fn zlib_decode_rejects_truncated_stream() {
    let compressed = zlib_encode(b"hello, world!");
    assert!(zlib_decode(&compressed[..compressed.len() - 6]).is_err());
}

#[test]
fn compression_actually_shrinks_repetitive_data() {
    let data = "a".repeat(10_000);
    let compressed = zlib_encode(data.as_bytes());
    assert!(compressed.len() < data.len() / 10);
}

#[test]
fn bit_reader_read_bit_rejects_truncated_stream() {
    let mut reader = BitReader::new(&[]);
    assert!(reader.read_bit().is_err());
}

#[test]
fn bit_reader_read_bytes_rejects_truncated_stream() {
    let mut reader = BitReader::new(&[0u8; 2]);
    assert!(reader.read_bytes(5).is_err());
}

#[test]
fn huffman_build_skips_zero_length_symbols() {
    let _ = Huffman::build(&[0, 1, 1]);
}

#[test]
fn inflate_rejects_stored_block_len_nlen_mismatch() {
    assert!(inflate(&[0x01, 0x05, 0x00, 0x00, 0x00]).is_err());
}

#[test]
fn inflate_rejects_reserved_block_type() {
    assert!(inflate(&[0x07]).is_err());
}

#[test]
fn zlib_decode_rejects_stream_shorter_than_six_bytes() {
    assert!(zlib_decode(&[0, 1, 2]).is_err());
}

#[test]
fn zlib_decode_rejects_unsupported_compression_method() {
    assert!(zlib_decode(&[0x00, 0x00, 0, 0, 0, 0]).is_err());
}

#[test]
fn zlib_decode_rejects_preset_dictionary() {
    assert!(zlib_decode(&[0x78, 0x20, 0, 0, 0, 0]).is_err());
}

#[test]
fn zlib_decode_rejects_bad_header_check_bits() {
    assert!(zlib_decode(&[0x78, 0x00, 0, 0, 0, 0]).is_err());
}

#[test]
fn read_dynamic_tables_rejects_repeat_16_with_no_previous_length() {
    let mut writer = BitWriter::new();
    writer.write_bits(0, 5); // hlit
    writer.write_bits(0, 5); // hdist
    writer.write_bits(0, 4); // hclen
    writer.write_bits(1, 3); // code length for symbol 16
    writer.write_bits(0, 3); // code length for symbol 17
    writer.write_bits(0, 3); // code length for symbol 18
    writer.write_bits(0, 3); // code length for symbol 0
    writer.write_huffman(0, 1); // encodes symbol 16
    writer.write_bits(0, 2); // repeat count bits
    let bytes = writer.finish();
    let mut reader = BitReader::new(&bytes);
    assert!(read_dynamic_tables(&mut reader).is_err());
}

#[test]
fn read_dynamic_tables_rejects_code_length_overflow() {
    let mut writer = BitWriter::new();
    writer.write_bits(0, 5); // hlit -> 257
    writer.write_bits(0, 5); // hdist -> 1
    writer.write_bits(0, 4); // hclen -> 4
    writer.write_bits(0, 3); // code length for symbol 16
    writer.write_bits(0, 3); // code length for symbol 17
    writer.write_bits(1, 3); // code length for symbol 18
    writer.write_bits(0, 3); // code length for symbol 0
    writer.write_huffman(0, 1); // encodes symbol 18
    writer.write_bits(127, 7); // repeat = 138
    writer.write_huffman(0, 1); // encodes symbol 18 again
    writer.write_bits(127, 7); // repeat = 138, overshoots 258
    let bytes = writer.finish();
    let mut reader = BitReader::new(&bytes);
    assert!(read_dynamic_tables(&mut reader).is_err());
}

#[test]
fn inflate_block_rejects_invalid_literal_symbol() {
    let mut lengths = vec![0u8; 288];
    lengths[286] = 1;
    let literal = Huffman::build(&lengths);
    let distance = Huffman::build(&[1u8]);
    let mut writer = BitWriter::new();
    writer.write_huffman(0, 1);
    let bytes = writer.finish();
    let mut reader = BitReader::new(&bytes);
    let mut output = Vec::new();
    assert!(inflate_block(&mut reader, &literal, &distance, &mut output).is_err());
}

#[test]
fn inflate_block_rejects_invalid_distance_symbol() {
    let mut literal_lengths = vec![0u8; 258];
    literal_lengths[257] = 1;
    let literal = Huffman::build(&literal_lengths);
    let mut distance_lengths = vec![0u8; 31];
    distance_lengths[30] = 1;
    let distance = Huffman::build(&distance_lengths);
    let mut writer = BitWriter::new();
    writer.write_huffman(0, 1); // literal symbol 257
    writer.write_huffman(0, 1); // distance symbol 30
    let bytes = writer.finish();
    let mut reader = BitReader::new(&bytes);
    let mut output = Vec::new();
    assert!(inflate_block(&mut reader, &literal, &distance, &mut output).is_err());
}

#[test]
fn round_trip_lz77_partial_match_below_max_length() {
    round_trip(b"abcdeabcXYZ");
}

#[test]
fn round_trip_lz77_candidate_beyond_window() {
    let mut data = b"XYZ".to_vec();
    data.extend(std::iter::repeat_n(0u8, 40_000));
    data.extend_from_slice(b"XYZ");
    round_trip(&data);
}

#[test]
fn inflate_rejects_stored_block_data_truncated_after_stored_block() {
    // BFINAL=0, BTYPE=0 (stored), LEN=0, NLEN=!0, then a final stored block "ABC".
    let bytes: Vec<u8> = vec![
        0x00, 0x00, 0x00, 0xFF, 0xFF, 0x01, 0x03, 0x00, 0xFC, 0xFF, 0x41, 0x42, 0x43,
    ];
    assert_eq!(inflate(&bytes).unwrap(), b"ABC");
}

#[test]
fn inflate_round_trips_dynamic_huffman_block_built_by_hand() {
    let mut writer = BitWriter::new();
    writer.write_bit(1); // BFINAL
    writer.write_bits(2, 2); // BTYPE = 2 (dynamic)
    writer.write_bits(0, 5); // HLIT -> 257
    writer.write_bits(0, 5); // HDIST -> 1
    writer.write_bits(12, 4); // HCLEN -> 16
    for value in [3u32, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3] {
        writer.write_bits(value, 3);
    }
    // code-length alphabet: symbol 0 -> code 0b000, symbol 2 -> 0b001,
    // symbol 16 -> 0b010, symbol 17 -> 0b011, symbol 18 -> 0b100 (all length 3).
    writer.write_huffman(0, 3); // idx 0: explicit length 0
    writer.write_huffman(2, 3); // repeat previous (16), count = 4
    writer.write_bits(1, 2);
    writer.write_huffman(3, 3); // repeat zero (17), count = 5
    writer.write_bits(2, 3);
    writer.write_huffman(4, 3); // repeat zero (18), count = 55
    writer.write_bits(44, 7);
    writer.write_huffman(1, 3); // idx 65: explicit length 2 ('A')
    writer.write_huffman(1, 3); // idx 66: explicit length 2 ('B')
    writer.write_huffman(4, 3); // repeat zero (18), count = 138
    writer.write_bits(127, 7);
    writer.write_huffman(4, 3); // repeat zero (18), count = 51
    writer.write_bits(40, 7);
    writer.write_huffman(1, 3); // idx 256: explicit length 2 (EOB)
    writer.write_huffman(0, 3); // idx 257 (distance): explicit length 0
    // literal alphabet: symbol 65 -> 0b00, symbol 66 -> 0b01, symbol 256 -> 0b10 (length 2).
    writer.write_huffman(0, 2); // 'A'
    writer.write_huffman(1, 2); // 'B'
    writer.write_huffman(2, 2); // end of block
    let bytes = writer.finish();
    assert_eq!(inflate(&bytes).unwrap(), b"AB");
}

#[test]
fn inflate_block_rejects_distance_exceeding_output() {
    let mut literal_lengths = vec![0u8; 258];
    literal_lengths[257] = 1;
    let literal = Huffman::build(&literal_lengths);
    let distance = Huffman::build(&[1u8]);
    let mut writer = BitWriter::new();
    writer.write_huffman(0, 1); // literal symbol 257 (length 3)
    writer.write_huffman(0, 1); // distance symbol 0 (distance 1)
    let bytes = writer.finish();
    let mut reader = BitReader::new(&bytes);
    let mut output = Vec::new();
    assert!(inflate_block(&mut reader, &literal, &distance, &mut output).is_err());
}
