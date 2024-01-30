from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import (
    MT5600MModelMixin,
    MT51BModelMixin,
    MT53BModelMixin,
    MT513BModelMixin,
    MT51BModelLoraMixin,
    MADLADMT3BModelMixin,
    MADLADMT10BModelMixin,

)
from lme.training_argument_mixins import MT5FinetuneArgsMixin, MT513BFinetuneArgsMixin
from lme.experiments.translation.mixin import TranslationMixin
from lme.experiments.translation.pali import TranslationPali, TranslationPaliDevanagari
from lme.experiments.translation.sanskrit import TranslationSanskritParagraph
from lme.experiments.translation.chinese import TranslationChinese, TranslationChineseParagraph, PretrainingChinese
from lme.experiments.alignment.sentence_alignment import AlignmentCorrection
from lme.experiments.translation.mitra import TranslationMitra

class TranslationMT5ExperimentBase(MT5FinetuneArgsMixin, TranslationMixin, FinetuneExperimentBase):
    pass


class TranslationMT5600MExperiment(MT5600MModelMixin, TranslationMT5ExperimentBase):
    pass


class TranslationMT51BExperiment(MT51BModelMixin, TranslationMT5ExperimentBase):
    pass


class TranslationMT53BExperiment(MT53BModelMixin, TranslationMT5ExperimentBase):
    pass


class TranslationMT513BExperiment(MT513BModelMixin, MT513BFinetuneArgsMixin, TranslationMixin, FinetuneExperimentBase):
    pass

class TranslationPaliMT5ExperimentBase(MT5FinetuneArgsMixin, TranslationPali, FinetuneExperimentBase):
    pass

class TranslationPaliMT5600MExperiment(MT5600MModelMixin, TranslationPaliMT5ExperimentBase):
    pass

class TranslationPaliMT51BExperiment(MT51BModelMixin, TranslationPaliMT5ExperimentBase):
    pass


class TranslationPaliDevanagariMT5ExperimentBase(MT5FinetuneArgsMixin, TranslationPaliDevanagari, FinetuneExperimentBase):
    pass

class TranslationPaliDevanagariMT5600MExperiment(MT5600MModelMixin, TranslationPaliDevanagariMT5ExperimentBase):
    pass

class TranslationChineseMT5ExperimentBase(MT5FinetuneArgsMixin, TranslationChinese, FinetuneExperimentBase):
    pass

class TranslationChineseMT5600MExperiment(MT5600MModelMixin, TranslationChineseMT5ExperimentBase):
    pass

class TranslationChineseParagraphMT5ExperimentBase(MT5FinetuneArgsMixin, TranslationChineseParagraph, FinetuneExperimentBase):
    pass

class TranslationChineseParagraphMT5600MExperiment(MT5600MModelMixin, TranslationChineseParagraphMT5ExperimentBase):
    pass

class ChinesePretrainingMT5LoraExperimentBase(MT5FinetuneArgsMixin, PretrainingChinese, FinetuneExperimentBase):
    pass

class ChinesePretrainingMT51BLoraExperiment(MT51BModelLoraMixin, ChinesePretrainingMT5LoraExperimentBase):
    pass

class TranslationSanskritParagraphMT5ExperimentBase(MT5FinetuneArgsMixin, TranslationSanskritParagraph, FinetuneExperimentBase):
    pass

class TranslationSanskritParagraphMT5600MExperiment(MT5600MModelMixin, TranslationSanskritParagraphMT5ExperimentBase):
    pass

class TranslationSanskritParagraphMT51BExperiment(MT51BModelMixin, TranslationSanskritParagraphMT5ExperimentBase):
    pass

class TranslationSanskritParagraphMT53BExperiment(MT53BModelMixin, TranslationSanskritParagraphMT5ExperimentBase):
    pass

class AlignmentCorrectionMT5ExperimentBase(MT5FinetuneArgsMixin, AlignmentCorrection, FinetuneExperimentBase):
    pass

class AlignmentCorrectionMT5600MExperiment(MT5600MModelMixin, AlignmentCorrectionMT5ExperimentBase):
    pass

class AlignmentCorrectionMT51BExperiment(MT51BModelMixin, AlignmentCorrectionMT5ExperimentBase):
    pass

class TranslationMitraMADLADMTExperimentBase(MT5FinetuneArgsMixin, TranslationMitra, FinetuneExperimentBase):
    pass

class TranslationMitraMADLADMTExperiment(MADLADMT3BModelMixin, TranslationMitraMADLADMTExperimentBase):
    pass

class TranslationMitraMADLADMT10BExperiment(MADLADMT10BModelMixin, TranslationMitraMADLADMTExperimentBase):
    pass