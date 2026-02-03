"""Corpus-specific preprocessing modules."""

from trax.data.preprocessing.modules.c4 import C4Preprocess, C4Tokenize
from trax.data.preprocessing.modules.bert import (
    BertDoubleSentenceInputs,
    BertNextSentencePredictionInputs,
    BertSingleSentenceInputs,
    CreateBertInputs,
    mask_random_tokens,
)
from trax.data.preprocessing.modules.cifar import (
    Cifar10Augmentation,
    Cifar10AugmentationFlatten,
    Cifar10FlattenNoAugmentation,
    Cifar10NoAugmentation,
)
from trax.data.preprocessing.modules.aqua import CreateAquaInputs
from trax.data.preprocessing.modules.math import (
    CreateMathQAInputs,
    convert_float_to_mathqa,
    convert_to_subtract,
    execute_mathqa_dsl_program,
    execute_mathqa_program,
    process_single_mathqa_example,
)
from trax.data.preprocessing.modules.drop import (
    CreateAnnotatedDropInputs,
    CreateDropInputs,
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
from trax.data.preprocessing.modules.imagenet import DownsampledImagenetFlatten
from trax.data.preprocessing.modules.video import (
    BairRobotPushingPreprocess,
    bair_robot_pushing_hparams,
)
from trax.data.preprocessing.modules.t5 import (
    denoise_t5,
    select_random_chunk_t5,
    split_tokens_t5,
    t5_data,
)
from trax.data.preprocessing.modules.modern_bert import (
    AddPositionIds,
    AppendDocBoundaryTokens,
    BatchDictList,
    BatchDict,
    ChunkFixedLength,
    ModernBertSequencePacker,
    ModernBertAppendDocBoundaryTokens,
    ModernBertTokenize,
    SelectTextField,
    ToModelTuple,
    TokenizerEncode,
)

__all__ = [
    "C4Preprocess",
    "C4Tokenize",
    "BertDoubleSentenceInputs",
    "BertNextSentencePredictionInputs",
    "BertSingleSentenceInputs",
    "CreateBertInputs",
    "mask_random_tokens",
    "Cifar10Augmentation",
    "Cifar10AugmentationFlatten",
    "Cifar10FlattenNoAugmentation",
    "Cifar10NoAugmentation",
    "CreateMathQAInputs",
    "CreateAquaInputs",
    "CreateAnnotatedDropInputs",
    "CreateDropInputs",
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
    "bair_robot_pushing_hparams",
    "BairRobotPushingPreprocess",
    "DownsampledImagenetFlatten",
    "t5_data",
    "denoise_t5",
    "select_random_chunk_t5",
    "split_tokens_t5",
    "SelectTextField",
    "TokenizerEncode",
    "AppendDocBoundaryTokens",
    "ChunkFixedLength",
    "BatchDictList",
    "BatchDict",
    "ModernBertSequencePacker",
    "ModernBertAppendDocBoundaryTokens",
    "AddPositionIds",
    "ToModelTuple",
    "ModernBertTokenize",
]
