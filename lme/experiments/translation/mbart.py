from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import (
    MBARTModelMixin,
    MBARTManyToOneModelMixin,
)
from lme.training_argument_mixins import MBARTFinetuneArgsMixin
from lme.experiments.translation.pali import TranslationPali, TranslationPaliDevanagari
from lme.experiments.translation.chinese import TranslationChinese, TranslationChineseParagraph


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

