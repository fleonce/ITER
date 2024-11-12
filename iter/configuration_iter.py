from typing import Dict, Any, Optional

import transformers
from transformers import PretrainedConfig, AutoConfig, T5Tokenizer, BertTokenizerFast, T5TokenizerFast, BertTokenizer, \
    BertModel, LongT5EncoderModel, T5EncoderModel, AlbertTokenizerFast, AlbertTokenizer, AlbertModel, \
    RobertaTokenizerFast, RobertaTokenizer, RobertaModel, AutoTokenizer

from iter.modeling_features import FeaturesMixin


class ITERConfig(PretrainedConfig, FeaturesMixin):
    d_ff: int
    d_model: int
    num_types: int
    num_links: int
    features: int
    max_nest_depth: int
    dropout: float
    threshold: float
    transformer_config: PretrainedConfig
    model_type = "iter"

    def __init__(
            self,
            transformer_name="t5-small",
            transformer_config=None,
            num_types=4,
            num_links=5,
            features=0,
            dataset: Optional[str] = None,
            max_nest_depth: int = 1,
            dropout: float = 0.3,
            activation_fn: str = "gelu",
            use_gate: bool = True,
            use_bias: bool = False,
            use_mlp: bool = True,
            d_ff: int = 0,
            threshold: float = 0.5,
            entity_types: Optional[list[str]] = None,
            link_types: Optional[list[str]] = None,
            **kwargs
    ):
        if isinstance(features, list):
            features = sum(2** feat for feat in features)
        self.features = features

        transformer_name = (transformer_config or {}).get("_name_or_path", transformer_name)
        self.transformer_config = AutoConfig.from_pretrained(transformer_name, **(transformer_config or {}))
        self.num_types = num_types
        self.num_links = num_links
        self.dataset = dataset
        self.max_nest_depth = max_nest_depth
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.use_bias = use_bias
        self.use_gate = use_gate
        self.use_mlp = use_mlp
        self.use_scale = d_ff == 0
        self.threshold = threshold
        self.entity_types = entity_types or list()
        self.link_types = link_types or list()
        self.d_ff = d_ff or self.try_attr_options(
            "d_ff",  # T5 model family
            "intermediate_size",  # BERT model family
            "encoder_ffn_dim",  # BART model family
        )
        self.d_model = self.try_attr_options(
            "d_model",  # T5
            "hidden_size",  # BERT
        )

        PretrainedConfig.__init__(
            self,
            is_encoder_decoder=False,
            **kwargs
        )
        self.max_length = self.transformer_config.max_length

    @property
    def num_real_links(self):
        return len(self.link_types)

    @property
    def num_real_types(self):
        return len(self.entity_types)

    @property
    def model_kwargs(self) -> dict:
        if self.is_feature_use_t5_decoder and self.transformer_config.model_type not in {"bert", "longt5"}:
            return dict(is_decoder=self.is_feature_use_t5_decoder_as_decoder)
        return dict()

    def tokenizer_kwargs(self, use_fast=True) -> dict:
        tokenizer_cls = self.guess_tokenizer_class(use_fast)
        tokenizer_kwargs = dict()
        if tokenizer_cls == AutoTokenizer:
            tokenizer_kwargs['use_fast'] = use_fast
        if "roberta" in self.transformer_config.model_type or "bart" in self.transformer_config.model_type:
            tokenizer_kwargs['add_prefix_space'] = True
        return tokenizer_kwargs

    def guess_tokenizer_class(self, use_fast=False):
        if "t5" in self.transformer_config.model_type:
            return T5Tokenizer if not use_fast else T5TokenizerFast
        elif "albert" in self.transformer_config.model_type:
            return AlbertTokenizer if not use_fast else AlbertTokenizerFast
        elif "roberta" in self.transformer_config.model_type:
            return RobertaTokenizer if not use_fast else RobertaTokenizerFast
        return AutoTokenizer

    def guess_model_class(self):
        if "bert" == self.transformer_config.model_type:
            return BertModel
        elif "longt5" == self.transformer_config.model_type:
            return LongT5EncoderModel
        elif "albert" == self.transformer_config.model_type:
            return AlbertModel
        elif "roberta" in self.transformer_config.model_type:
            return RobertaModel
        elif "t5" in self.transformer_config.model_type:
            if self.is_feature_use_t5_decoder:
                raise NotImplementedError
            else:
                return T5EncoderModel
        else:
            return transformers.AutoModel

    def try_attr_options(self, *items):
        exceptions = []
        for item in items:
            try:
                return self.transformer_config.__getattribute__(item)
            except AttributeError as cause:
                exceptions.append(cause)
        raise AttributeError from exceptions[-1]

    @staticmethod
    def _get_generation_defaults() -> Dict[str, Any]:
        return {}

    def _get_non_default_generation_parameters(self) -> Dict[str, Any]:
        # hacky workaround as transformers errors out if you have overridden `max_length` as of v44
        return dict()