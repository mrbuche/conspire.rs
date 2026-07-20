#[cfg(test)]
mod test;

use std::io::{Error, ErrorKind, Result};

const MAX_BITS: usize = 15;
const WINDOW: usize = 32768;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn read_bit(&mut self) -> Result<u32> {
        let byte = self.pos / 8;
        let bit = self.pos % 8;
        let value = *self
            .data
            .get(byte)
            .ok_or_else(|| invalid("truncated deflate stream"))?;
        self.pos += 1;
        Ok(((value >> bit) & 1) as u32)
    }
    fn read_bits(&mut self, n: u32) -> Result<u32> {
        let mut value = 0;
        for i in 0..n {
            value |= self.read_bit()? << i;
        }
        Ok(value)
    }
    fn align(&mut self) {
        self.pos = self.pos.div_ceil(8) * 8;
    }
    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let start = self.pos / 8;
        let end = start + n;
        if end > self.data.len() {
            return Err(invalid("truncated deflate stream"));
        }
        self.pos = end * 8;
        Ok(&self.data[start..end])
    }
    fn peek_bits(&self, n: u32) -> u32 {
        let mut code = 0;
        for pos in self.pos..self.pos + n as usize {
            let byte = self.data.get(pos / 8).copied().unwrap_or(0);
            let bit = (byte >> (pos % 8)) & 1;
            code = (code << 1) | bit as u32;
        }
        code
    }
    fn consume(&mut self, n: u32) {
        self.pos += n as usize;
    }
}

struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    count: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current: 0,
            count: 0,
        }
    }
    fn write_bit(&mut self, bit: u32) {
        self.current |= ((bit & 1) as u8) << self.count;
        self.count += 1;
        if self.count == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.count = 0;
        }
    }
    fn write_bits(&mut self, value: u32, n: u32) {
        for i in 0..n {
            self.write_bit((value >> i) & 1);
        }
    }
    fn write_huffman(&mut self, code: u32, len: u32) {
        for i in (0..len).rev() {
            self.write_bit((code >> i) & 1);
        }
    }
    fn align(&mut self) {
        if self.count > 0 {
            self.bytes.push(self.current);
            self.current = 0;
            self.count = 0;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        self.align();
        self.bytes
    }
}

struct Huffman {
    table: Vec<(u16, u8)>,
}

impl Huffman {
    fn build(lengths: &[u8]) -> Self {
        let mut bl_count = [0u32; MAX_BITS + 1];
        for &length in lengths {
            bl_count[length as usize] += 1;
        }
        bl_count[0] = 0;
        let mut next_code = [0u32; MAX_BITS + 1];
        let mut code = 0u32;
        for bits in 1..=MAX_BITS {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }
        let mut table = vec![(0u16, 0u8); 1 << MAX_BITS];
        for (symbol, &length) in lengths.iter().enumerate() {
            if length == 0 {
                continue;
            }
            let length = length as usize;
            let code = next_code[length];
            next_code[length] += 1;
            let shift = MAX_BITS - length;
            let base = (code as usize) << shift;
            table[base..base + (1 << shift)].fill((symbol as u16, length as u8));
        }
        Self { table }
    }
    fn decode(&self, reader: &mut BitReader) -> Result<u16> {
        let window = reader.peek_bits(MAX_BITS as u32) as usize;
        let (symbol, length) = self.table[window];
        if length == 0 || reader.pos + length as usize > reader.data.len() * 8 {
            return Err(invalid("invalid Huffman code in deflate stream"));
        }
        reader.consume(length as u32);
        Ok(symbol)
    }
}

fn fixed_literal_lengths() -> Vec<u8> {
    let mut lengths = vec![0u8; 288];
    lengths[0..144].fill(8);
    lengths[144..256].fill(9);
    lengths[256..280].fill(7);
    lengths[280..288].fill(8);
    lengths
}

fn fixed_distance_lengths() -> Vec<u8> {
    vec![5u8; 30]
}

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u32; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u32; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u32; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

fn length_code(length: usize) -> (u16, u32, u32) {
    let index = LENGTH_BASE
        .iter()
        .rposition(|&base| base as usize <= length)
        .unwrap();
    (
        257 + index as u16,
        LENGTH_EXTRA[index],
        (length - LENGTH_BASE[index] as usize) as u32,
    )
}

fn distance_code(distance: usize) -> (u16, u32, u32) {
    let index = DIST_BASE
        .iter()
        .rposition(|&base| base as usize <= distance)
        .unwrap();
    (
        index as u16,
        DIST_EXTRA[index],
        (distance - DIST_BASE[index] as usize) as u32,
    )
}

pub fn inflate(data: &[u8]) -> Result<Vec<u8>> {
    let mut reader = BitReader::new(data);
    let mut output = Vec::new();
    loop {
        let is_final = reader.read_bit()? == 1;
        match reader.read_bits(2)? {
            0 => {
                reader.align();
                let len_bytes = reader.read_bytes(4)?;
                let len = u16::from_le_bytes([len_bytes[0], len_bytes[1]]) as usize;
                let nlen = u16::from_le_bytes([len_bytes[2], len_bytes[3]]);
                if nlen != !(len as u16) {
                    return Err(invalid("stored block LEN/NLEN mismatch"));
                }
                output.extend_from_slice(reader.read_bytes(len)?);
            }
            1 => {
                let literal = Huffman::build(&fixed_literal_lengths());
                let distance = Huffman::build(&fixed_distance_lengths());
                inflate_block(&mut reader, &literal, &distance, &mut output)?;
            }
            2 => {
                let (literal, distance) = read_dynamic_tables(&mut reader)?;
                inflate_block(&mut reader, &literal, &distance, &mut output)?;
            }
            _ => return Err(invalid("reserved block type in deflate stream")),
        }
        if is_final {
            break;
        }
    }
    Ok(output)
}

fn read_dynamic_tables(reader: &mut BitReader) -> Result<(Huffman, Huffman)> {
    let hlit = reader.read_bits(5)? as usize + 257;
    let hdist = reader.read_bits(5)? as usize + 1;
    let hclen = reader.read_bits(4)? as usize + 4;
    let mut code_length_lengths = [0u8; 19];
    for &order in CODE_LENGTH_ORDER.iter().take(hclen) {
        code_length_lengths[order] = reader.read_bits(3)? as u8;
    }
    let code_length_huffman = Huffman::build(&code_length_lengths);
    let mut lengths = Vec::with_capacity(hlit + hdist);
    while lengths.len() < hlit + hdist {
        match code_length_huffman.decode(reader)? {
            symbol @ 0..=15 => lengths.push(symbol as u8),
            16 => {
                let repeat = reader.read_bits(2)? + 3;
                let previous = *lengths
                    .last()
                    .ok_or_else(|| invalid("repeat code 16 with no previous length"))?;
                lengths.extend(std::iter::repeat_n(previous, repeat as usize));
            }
            17 => {
                let repeat = reader.read_bits(3)? + 3;
                lengths.extend(std::iter::repeat_n(0, repeat as usize));
            }
            18 => {
                let repeat = reader.read_bits(7)? + 11;
                lengths.extend(std::iter::repeat_n(0, repeat as usize));
            }
            other => return Err(invalid(format!("invalid code-length symbol {other}"))),
        }
    }
    if lengths.len() != hlit + hdist {
        return Err(invalid("dynamic Huffman code-length overflow"));
    }
    let literal = Huffman::build(&lengths[..hlit]);
    let distance = Huffman::build(&lengths[hlit..]);
    Ok((literal, distance))
}

fn inflate_block(
    reader: &mut BitReader,
    literal: &Huffman,
    distance: &Huffman,
    output: &mut Vec<u8>,
) -> Result<()> {
    loop {
        let symbol = literal.decode(reader)?;
        match symbol {
            0..=255 => output.push(symbol as u8),
            256 => return Ok(()),
            257..=285 => {
                let index = (symbol - 257) as usize;
                let length =
                    LENGTH_BASE[index] as usize + reader.read_bits(LENGTH_EXTRA[index])? as usize;
                let dist_symbol = distance.decode(reader)? as usize;
                if dist_symbol >= DIST_BASE.len() {
                    return Err(invalid("invalid distance symbol"));
                }
                let dist = DIST_BASE[dist_symbol] as usize
                    + reader.read_bits(DIST_EXTRA[dist_symbol])? as usize;
                if dist > output.len() {
                    return Err(invalid("distance exceeds output so far"));
                }
                let start = output.len() - dist;
                for i in 0..length {
                    let byte = output[start + i];
                    output.push(byte);
                }
            }
            other => return Err(invalid(format!("invalid literal/length symbol {other}"))),
        }
    }
}

enum Token {
    Literal(u8),
    Match { length: u16, distance: u16 },
}

fn lz77(data: &[u8]) -> Vec<Token> {
    const HASH_BITS: usize = 15;
    const HASH_SIZE: usize = 1 << HASH_BITS;
    const MAX_CHAIN: usize = 128;
    let n = data.len();
    let mut head = vec![u32::MAX; HASH_SIZE];
    let mut prev = vec![u32::MAX; n];
    let hash = |i: usize| -> usize {
        ((data[i] as u32) ^ ((data[i + 1] as u32) << 5) ^ ((data[i + 2] as u32) << 10)) as usize
            & (HASH_SIZE - 1)
    };
    let insert = |i: usize, head: &mut [u32], prev: &mut [u32]| {
        if i + MIN_MATCH <= n {
            let h = hash(i);
            prev[i] = head[h];
            head[h] = i as u32;
        }
    };
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < n {
        let mut best_len = 0;
        let mut best_dist = 0;
        if i + MIN_MATCH <= n {
            let h = hash(i);
            let mut candidate = head[h];
            let mut tries = 0;
            let max_len = (n - i).min(MAX_MATCH);
            while candidate != u32::MAX && tries < MAX_CHAIN {
                let c = candidate as usize;
                if i - c > WINDOW {
                    break;
                }
                if best_len < max_len && data[c + best_len] == data[i + best_len] {
                    let mut len = 0;
                    while len < max_len && data[c + len] == data[i + len] {
                        len += 1;
                    }
                    if len > best_len {
                        best_len = len;
                        best_dist = i - c;
                        if len >= max_len {
                            break;
                        }
                    }
                }
                candidate = prev[c];
                tries += 1;
            }
        }
        if best_len >= MIN_MATCH {
            tokens.push(Token::Match {
                length: best_len as u16,
                distance: best_dist as u16,
            });
            let end = i + best_len;
            while i < end {
                insert(i, &mut head, &mut prev);
                i += 1;
            }
        } else {
            tokens.push(Token::Literal(data[i]));
            insert(i, &mut head, &mut prev);
            i += 1;
        }
    }
    tokens
}

struct FixedCodes {
    literal: [(u32, u32); 288],
    distance: [(u32, u32); 30],
}

fn fixed_codes() -> FixedCodes {
    let mut literal = [(0u32, 0u32); 288];
    for (symbol, code) in literal.iter_mut().enumerate().take(144) {
        *code = (0b0011_0000 + symbol as u32, 8);
    }
    for (symbol, code) in literal.iter_mut().enumerate().take(256).skip(144) {
        *code = (0b1_1001_0000 + (symbol as u32 - 144), 9);
    }
    for (symbol, code) in literal.iter_mut().enumerate().take(280).skip(256) {
        *code = (symbol as u32 - 256, 7);
    }
    for (symbol, code) in literal.iter_mut().enumerate().take(288).skip(280) {
        *code = (0b1100_0000 + (symbol as u32 - 280), 8);
    }
    let mut distance = [(0u32, 0u32); 30];
    for (symbol, code) in distance.iter_mut().enumerate() {
        *code = (symbol as u32, 5);
    }
    FixedCodes { literal, distance }
}

pub fn deflate(data: &[u8]) -> Vec<u8> {
    let mut writer = BitWriter::new();
    if data.is_empty() {
        writer.write_bit(1);
        writer.write_bits(1, 2);
        let (code, len) = fixed_codes().literal[256];
        writer.write_huffman(code, len);
        return writer.finish();
    }
    let tokens = lz77(data);
    let codes = fixed_codes();
    writer.write_bit(1);
    writer.write_bits(1, 2);
    for token in &tokens {
        match token {
            Token::Literal(byte) => {
                let (code, len) = codes.literal[*byte as usize];
                writer.write_huffman(code, len);
            }
            Token::Match { length, distance } => {
                let (symbol, extra_bits, extra_val) = length_code(*length as usize);
                let (code, len) = codes.literal[symbol as usize];
                writer.write_huffman(code, len);
                writer.write_bits(extra_val, extra_bits);
                let (dsymbol, dextra_bits, dextra_val) = distance_code(*distance as usize);
                let (dcode, dlen) = codes.distance[dsymbol as usize];
                writer.write_huffman(dcode, dlen);
                writer.write_bits(dextra_val, dextra_bits);
            }
        }
    }
    let (code, len) = codes.literal[256];
    writer.write_huffman(code, len);
    writer.finish()
}

pub fn adler32(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    (b << 16) | a
}

pub fn zlib_encode(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    out.extend(deflate(data));
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

pub fn zlib_decode(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 6 {
        return Err(invalid("zlib stream too short"));
    }
    let cmf = data[0];
    let flg = data[1];
    if cmf & 0x0F != 8 {
        return Err(unsupported(
            "only the DEFLATE zlib compression method is supported",
        ));
    }
    if (flg & 0x20) != 0 {
        return Err(unsupported("zlib preset dictionaries are not supported"));
    }
    if !((cmf as u32) * 256 + flg as u32).is_multiple_of(31) {
        return Err(invalid("zlib header check bits are invalid"));
    }
    let payload = &data[2..data.len() - 4];
    let output = inflate(payload)?;
    let trailer = &data[data.len() - 4..];
    let checksum = u32::from_be_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
    if adler32(&output) != checksum {
        return Err(invalid("zlib Adler-32 checksum mismatch"));
    }
    Ok(output)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, message.into())
}

fn unsupported(message: &str) -> Error {
    Error::new(ErrorKind::Unsupported, message.to_string())
}
