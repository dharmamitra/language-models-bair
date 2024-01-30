from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import (
    MBARTModelMixin,
    MBARTManyToOneModelMixin,
    MBARTManyToManyModelMixin
)
from lme.training_argument_mixins import MBARTFinetuneArgsMixin
from lme.experiments.translation.pali import TranslationPali, TranslationPaliDevanagari
from lme.experiments.translation.chinese import TranslationChinese, TranslationChineseParagraph, TranslationChineseLiterary, TranslationChineseKorean
from lme.experiments.translation.sanskrit import TranslationSanskritParagraph, TranslationSanskitItihasa, TranslationSanskritItihasaTagged


class TranslationPaliMBARTExperimentBase(MBARTFinetuneArgsMixin, TranslationPali, FinetuneExperimentBase):
    pass


class TranslationPaliMBARTExperiment(MBARTModelMixin, TranslationPaliMBARTExperimentBase):
    pass

class TranslationPaliMBARTManyToOneExperiment(MBARTManyToOneModelMixin, TranslationPaliMBARTExperimentBase):
    pass

class TranslationPaliDevanagariMBARTExperimentBase(MBARTFinetuneArgsMixin, TranslationPaliDevanagari, FinetuneExperimentBase):
    pass
class TranslationPaliDevanagariMBARTExperiment(MBARTModelMixin, TranslationPaliDevanagariMBARTExperimentBase):
    pass

class TranslationPaliDevanagariMBARTManyToOneExperiment(MBARTManyToOneModelMixin, TranslationPaliDevanagariMBARTExperimentBase):
    pass

class TranslationChineseMBARTExperimentBase(MBARTFinetuneArgsMixin, TranslationChinese, FinetuneExperimentBase):
    pass

class TranslationChineseMBARTExperiment(MBARTModelMixin, TranslationChineseMBARTExperimentBase):
    pass

class TranslationChineseMBARTManyToOneExperiment(MBARTManyToOneModelMixin, TranslationChineseMBARTExperimentBase):
    pass

class TranslationChineseParagraphMBARTExperimentBase(MBARTFinetuneArgsMixin, TranslationChineseParagraph, FinetuneExperimentBase):
    pass

class TranslationChineseParagraphMBARTExperiment(MBARTModelMixin, TranslationChineseParagraphMBARTExperimentBase):
    pass

class TranslationChineseParagraphMBARTManyToOneExperiment(MBARTManyToOneModelMixin, TranslationChineseParagraphMBARTExperimentBase):
    pass

class TranslationChineseLiteraryMBARTExperimentBase(MBARTFinetuneArgsMixin, TranslationChineseLiterary, FinetuneExperimentBase):
    pass

class TranslationChineseLiteraryMBARTExperiment(MBARTModelMixin, TranslationChineseLiteraryMBARTExperimentBase):
    pass

class TranslationChineseLiteraryMBARTManyToOneExperiment(MBARTManyToOneModelMixin, TranslationChineseLiteraryMBARTExperimentBase):
    pass

class TranslationChineseKoreanMBARTExperimentBase(MBARTFinetuneArgsMixin, TranslationChineseKorean, FinetuneExperimentBase):
    pass

class TranslationChineseKoreanMBARTExperiment(MBARTManyToManyModelMixin, TranslationChineseKoreanMBARTExperimentBase):
    pass

class TranslationSanskritItihasaMBARTExperimentBase(MBARTFinetuneArgsMixin, TranslationSanskitItihasa, FinetuneExperimentBase):
    pass

class TranslationSanskritItihasaMBARTExperiment(MBARTModelMixin, TranslationSanskritItihasaMBARTExperimentBase):
    pass

class TranslationSanskritItihasaMBARTManyToOneExperiment(MBARTManyToOneModelMixin, TranslationSanskritItihasaMBARTExperimentBase):
    pass

class TranslationSanskritParagraphMBARTExperimentBase(MBARTFinetuneArgsMixin, TranslationSanskritParagraph, FinetuneExperimentBase):
    pass

class TranslationSanskritParagraphMBARTExperiment(MBARTModelMixin, TranslationSanskritParagraphMBARTExperimentBase):
    pass
