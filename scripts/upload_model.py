from iter import ITER
from with_argparse import with_argparse


@with_argparse
def upload(
        model_name: str,
        upload_name: str,
        public: bool = False,
):
    model = ITER.from_pretrained(model_name)
    model.save_pretrained("models/" + upload_name)
    model = ITER.from_pretrained("models/" + upload_name)
    model.push_to_hub(
        repo_id=upload_name,
        private=not public,
        use_temp_dir=False,
    )


if __name__ == "__main__":
    upload()
