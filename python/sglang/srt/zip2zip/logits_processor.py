from sglang.srt.layers.logits_processor import LogitsProcessor


class Zip2ZipLogitsProcessor:
    def __init__(self, logits_processor: LogitsProcessor) -> None:
        self.logits_processor = logits_processor
