from json import load as load_file, dump as dump_file
from pathlib import Path
import truecase

from tqdm import tqdm

from with_argparse import with_argparse


@with_argparse
def load_ace05(
    path: Path,
    update_truecase: bool = False,
):
    tc = truecase.get_truecaser()
    for filename in ["train", "dev", "test"]:
        with open(path / f"ace05_{filename}.json") as f:
            json_blob = load_file(f)
        out_json_blob = []
        for entry in tqdm(json_blob):
            tokens = entry["tokens"]
            extended_tokens = entry["extended"]
            if update_truecase and "".join(tokens).lower() == "".join(tokens):
                if "<extra_id_23>" in tokens:
                    sent_start = tokens.index("<extra_id_22>")
                    sent_end = tokens.index("<extra_id_23>")
                    tokens[sent_start+1:sent_end] = tc.get_true_case_from_tokens(
                        tokens[sent_start+1:sent_end],
                        out_of_vocabulary_token_option="as-is"
                    )
                else:
                    tokens = tc.get_true_case_from_tokens(tokens, out_of_vocabulary_token_option="as-is")

                if "<extra_id_23>" in extended_tokens:
                    sent_start = extended_tokens.index("<extra_id_22>")
                    sent_end = extended_tokens.index("<extra_id_23>")

                    extended_tokens[0:sent_start] = tc.get_true_case_from_tokens(
                        extended_tokens[0:sent_start], out_of_vocabulary_token_option="as-is"
                    )
                    extended_tokens[sent_start+1:sent_end] = tc.get_true_case_from_tokens(
                        extended_tokens[sent_start+1:sent_end], out_of_vocabulary_token_option="as-is",
                    )
                    extended_tokens[sent_end+1:] = tc.get_true_case_from_tokens(
                        extended_tokens[sent_end+1:], out_of_vocabulary_token_option="as-is"
                    )
                else:
                    extended_tokens = tc.get_true_case_from_tokens(
                        extended_tokens, out_of_vocabulary_token_option="as-is"
                    )
                entry["tokens"] = tokens
                entry["extended"] = extended_tokens
            out_json_blob.append(entry)
        with open(path / f"ace05_{filename}.json", "w") as f:
            dump_file(out_json_blob, f)

if __name__ == "__main__":
    load_ace05()
