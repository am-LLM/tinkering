from shannon_huffman_codec import ShannonHuffmanCodec

def test_huffman():
    ent = ShannonHuffmanCodec.shannon_entropy([0.5, 0.5])
    assert ent == 1.0
    codes = ShannonHuffmanCodec.huffman_codes({"a": 5, "b": 1})
    assert len(codes) == 2
