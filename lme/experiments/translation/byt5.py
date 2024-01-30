from lme.training_pipelines import FinetuneExperimentBase
from lme.model_mixins import (
    ByT5Google,
    ByT5GoogleLarge,
    ByT5Sanskrit,
)
from lme.training_argument_mixins import MT5FinetuneArgsMixin, MT5PretrainingArgsMixin
from lme.experiments.translation.sandhi import TranslationSandhi, TranslationSandhiSentence, TranslationSandhiHackathon, TranslationSandhiSighum, TranslationSanskritDP, PretrainingSanskrit, LemmatizeSanskrit, LemmatizeLongSanskrit
from lme.experiments.translation.sanskrit import TranslationSanskritParagraph


class TranslationByT5ExperimentBase(MT5FinetuneArgsMixin, TranslationSandhi, FinetuneExperimentBase):
    pass
class TranslationByT5GoogleExperiment(ByT5Google, TranslationByT5ExperimentBase):
    pass

class TranslationByT5SanskritExperiment(ByT5Sanskrit, TranslationByT5ExperimentBase):
    pass

class TranslationByT5SentenceExperimentBase(MT5FinetuneArgsMixin, TranslationSandhiSentence, FinetuneExperimentBase):
    pass

class TranslationByT5SentenceSanskritExperiment(ByT5Sanskrit, TranslationByT5SentenceExperimentBase):
    pass

class TranslationByT5SentenceGoogleExperiment(ByT5Google, TranslationByT5SentenceExperimentBase):
    pass

class TranslationByT5HackathonExperimentBase(MT5FinetuneArgsMixin, TranslationSandhiHackathon, FinetuneExperimentBase):
    pass

class TranslationByT5HackathonSanskritExperiment(ByT5Sanskrit, TranslationByT5HackathonExperimentBase):
    pass

class TranslationByT5HackathonGoogleExperiment(ByT5Google, TranslationByT5HackathonExperimentBase):
    pass

class TranslationByT5SighumExperimentBase(MT5FinetuneArgsMixin, TranslationSandhiSighum, FinetuneExperimentBase):
    pass

class TranslationByT5SighumSanskritExperiment(ByT5Sanskrit, TranslationByT5SighumExperimentBase):
    pass

class TranslationByT5SighumGoogleExperiment(ByT5Google, TranslationByT5SighumExperimentBase):
    pass

class TranslationByT5SanskritDPExperimentBase(MT5FinetuneArgsMixin, TranslationSanskritDP, FinetuneExperimentBase):
    pass

class TranslationByT5SanskritDPSanskritExperiment(ByT5Sanskrit, TranslationByT5SanskritDPExperimentBase):
    pass

class TranslationByT5SanskritDPGoogleExperiment(ByT5Google, TranslationByT5SanskritDPExperimentBase):
    pass

class PretrainingByT5SanskritExperimentBase(MT5PretrainingArgsMixin, PretrainingSanskrit, FinetuneExperimentBase):
    pass

class PretrainingByT5SanskritExperiment(ByT5Sanskrit, PretrainingByT5SanskritExperimentBase):
    pass

class LemmatizeByT5SanskritExperimentBase(MT5FinetuneArgsMixin, LemmatizeSanskrit, FinetuneExperimentBase):
    pass

class LemmatizeByT5SanskritExperiment(ByT5Sanskrit, LemmatizeByT5SanskritExperimentBase):
    pass

class TranslationByT5SanskritParagraphExperimentBase(MT5FinetuneArgsMixin, TranslationSanskritParagraph, FinetuneExperimentBase):
    pass

class TranslationByT5GoogleLargeParagraphExperiment(ByT5GoogleLarge, TranslationByT5SanskritParagraphExperimentBase):
    pass