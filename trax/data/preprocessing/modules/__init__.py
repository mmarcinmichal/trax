"""Corpus-specific preprocessing modules."""

from trax.data.preprocessing.modules.c4 import C4Preprocess, C4Tokenize
from trax.data.preprocessing.modules.bert import (
    BertDoubleSentenceInputs,
    BertNextSentencePredictionInputs,
    BertSingleSentenceInputs,
    CreateBertInputs,
    mask_random_tokens,
)
from trax.data.preprocessing.modules.math import (
    CreateMathQAInputs,
    convert_float_to_mathqa,
    convert_to_subtract,
    execute_mathqa_dsl_program,
    execute_mathqa_program,
    process_single_mathqa_example,
)
from trax.data.preprocessing.modules.wmt import (
    WMTConcatInputsTargets,
    WMTConcatPreprocess,
    WMTEnsureInputsTargets,
    WMTFilterByLength,
    WMTPreprocess,
    WMTToInputsTargetsTuple,
    WMTTokenize,
)

__all__ = [
    "C4Preprocess",
    "C4Tokenize",
    "BertDoubleSentenceInputs",
    "BertNextSentencePredictionInputs",
    "BertSingleSentenceInputs",
    "CreateBertInputs",
    "mask_random_tokens",
    "CreateMathQAInputs",
    "convert_float_to_mathqa",
    "convert_to_subtract",
    "execute_mathqa_dsl_program",
    "execute_mathqa_program",
    "process_single_mathqa_example",
    "WMTConcatInputsTargets",
    "WMTConcatPreprocess",
    "WMTEnsureInputsTargets",
    "WMTFilterByLength",
    "WMTPreprocess",
    "WMTToInputsTargetsTuple",
    "WMTTokenize",
]
