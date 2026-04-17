from tokenizer import PositionalCharTokenizer, build_embedding_initializer, build_transition_statistics


def main() -> None:
    tokenizer = PositionalCharTokenizer()

    assert [piece.token for piece in tokenizer.word_to_pieces("A")] == ["A1"]
    assert [piece.token for piece in tokenizer.word_to_pieces("AB")] == ["A1", "B-1"]
    assert [piece.token for piece in tokenizer.word_to_pieces("CAT")] == ["C1", "A2", "T-1"]
    assert [piece.token for piece in tokenizer.word_to_pieces("Thing")] == [
        "T1",
        "h2",
        "i3",
        "n-2",
        "g-1",
    ]
    assert [piece.token for piece in tokenizer.word_to_pieces("ABCDEFG")] == [
        "A1",
        "B2",
        "C3",
        "Dmiddle",
        "E-3",
        "F-2",
        "G-1",
    ]
    assert [piece.token for piece in tokenizer.word_to_pieces("Position")] == [
        "P1",
        "o2",
        "s3",
        "imiddle",
        "tmiddle",
        "i-3",
        "o-2",
        "n-1",
    ]
    assert tokenizer.text_to_tokens("Hi!\n") == ["h1", "i-1", "!", "\n"]
    stream = tokenizer.text_to_stream("Hi there")
    assert [item.token for item in stream] == ["h1", "i-1", " ", "t1", "h2", "e3", "r-2", "e-1"]
    assert stream[1].is_word_final is True
    assert stream[3].is_word_final is False

    tokenizer.fit("Hi there")
    counts, probs = build_transition_statistics(tokenizer, "Hi there")
    assert counts[tokenizer.token_to_id("i-1"), tokenizer.token_to_id("t1")] == 1
    assert probs[tokenizer.token_to_id("i-1"), tokenizer.token_to_id("t1")] == 1.0

    embedding = build_embedding_initializer(probs, embedding_dim=256)
    assert embedding.shape == (tokenizer.vocab_size(), 256)
    assert embedding[tokenizer.token_to_id("i-1"), tokenizer.token_to_id("t1")] == 1.0

    lower_stream = tokenizer.text_to_stream("Thing THING")
    assert [item.token for item in lower_stream] == [
        "t1",
        "h2",
        "i3",
        "n-2",
        "g-1",
        " ",
        "t1",
        "h2",
        "i3",
        "n-2",
        "g-1",
    ]

    print("tokenizer checks passed")


if __name__ == "__main__":
    main()
