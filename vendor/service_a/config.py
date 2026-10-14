"""Environment configuration; relative paths are relative to this project."""
from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

@dataclass
class Config:
    root: Path = ROOT
    target_language: str = 'English'
    source_language: str = 'auto'
    translation_provider: str = 'gemini'
    api_key: str = field(default='', repr=False)
    model: str = 'gemini-3.5-flash-lite'
    groq_api_key: str = field(default='', repr=False)
    groq_vision_model: str = 'qwen/qwen3.8-27b'
    gemini_api_key: str = field(default='', repr=False)
    gemini_model: str = 'gemini-3.5-flash-lite'
    stability_seconds: float = 5
    stability_timeout: float = 600
    max_retries: int = 2
    shortening_retries: int = 2
    # Three short lines give translations room for language expansion without
    # forcing the model to omit meaning or making the pipeline truncate a cue.
    max_lines: int = 3
    max_chars: int = 54
    batch_size: int = 12
    api_timeout: float = 120
    command_timeout: float = 7200
    ffmpeg: str = 'ffmpeg'
    ffprobe: str = 'ffprobe'
    ocr_enabled: bool = True
    ocr_provider: str = 'gemini'
    ocr_language: str = 'eng'
    ocr_interval: float = 0.25
    ocr_region: str = ''
    ocr_solid_bg_only: bool = False
    ocr_bg_variance_threshold: float = 30.0
    ocr_target_box: str = ''
    overlay_patch_color: str = '0x000000'
    box_text_color: str = '#FFFFFF'
    removal_mode: str = 'overlay'
    overlay_patch: bool = True
    font_name: str = ''
    font_size: int = 0
    subtitle_position: str = 'box'
    outline_width: float = 0.0
    shadow_width: float = 1.0
    min_subtitle_duration: float = 0.5
    max_words_per_segment: int = 4
    pill_mode: bool = True
    subtitle_stream: int | None = None

    def __post_init__(self):
        self.root = self.root.resolve()
        for attr, env, default in [('input_dir','INPUT_DIR','incoming'), ('processing_dir','PROCESSING_DIR','processing'), ('completed_dir','COMPLETED_DIR','completed'), ('failed_dir','FAILED_DIR','failed'), ('temp_dir','TEMP_DIR','temp'), ('log_dir','LOG_DIR','logs'), ('font_dir','FONT_DIR','fonts'), ('metadata_dir','METADATA_DIR','metadata'), ('script_dir','SCRIPT_DIR','scripts')]:
            value = os.getenv(env, default)
            if '://' in value or value.startswith(('\\\\', '//')):
                raise ValueError(f'{env} must be a local filesystem path')
            path = (self.root / value).resolve()
            path.mkdir(parents=True, exist_ok=True)
            setattr(self, attr, path)
        dirs = [self.input_dir,self.processing_dir,self.completed_dir,self.failed_dir]
        if any(a == b or a in b.parents or b in a.parents for i,a in enumerate(dirs) for b in dirs[i+1:]):
            raise ValueError('Media directories must be distinct and non-nested')
        (self.root / 'database').mkdir(exist_ok=True)
        configured_db = os.getenv('DB_PATH', 'database/jobs.db')
        if '://' in configured_db or configured_db.startswith(('\\\\', '//')):
            raise ValueError('DB_PATH must be a local filesystem path')
        self.db_path = Path(configured_db)
        if not self.db_path.is_absolute():
            self.db_path = self.root / self.db_path
        self.db_path = self.db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if min(self.stability_seconds,self.stability_timeout,self.ocr_interval,self.api_timeout,self.command_timeout) <= 0:
            raise ValueError('Timeouts and sampling intervals must be positive')
        if min(self.max_lines,self.max_chars,self.batch_size) < 1 or min(self.max_retries,self.shortening_retries) < 0:
            raise ValueError('Invalid limits')
        if not self.target_language.strip() or len(self.target_language) > 80:
            raise ValueError('Provide a target language of 1–80 characters')
        if self.translation_provider not in ('claude', 'gemini', 'groq'):
            raise ValueError("TRANSLATION_PROVIDER must be 'claude', 'gemini', or 'groq'")
        import shutil
        winget_pkgs = Path(os.environ.get('LOCALAPPDATA', '')) / 'Microsoft/WinGet/Packages'
        if not shutil.which(self.ffmpeg) and not Path(self.ffmpeg).is_file() and winget_pkgs.is_dir():
            for match in winget_pkgs.glob('**/ffmpeg.exe'):
                self.ffmpeg = str(match)
                break
        if not shutil.which(self.ffprobe) and not Path(self.ffprobe).is_file() and winget_pkgs.is_dir():
            for match in winget_pkgs.glob('**/ffprobe.exe'):
                self.ffprobe = str(match)
                break

    @classmethod
    def load(cls, target_language=None, source_language=None):
        load_dotenv(ROOT / '.env')
        mapping = {
            'target_language':('TARGET_LANGUAGE',str),
            'source_language':('SOURCE_LANGUAGE',str),
            'translation_provider':('TRANSLATION_PROVIDER',str),
            'api_key':('ANTHROPIC_API_KEY',str),
            'model':('TRANSLATION_MODEL',str),
            'groq_api_key':('GROQ_API_KEY',str),
            'groq_vision_model':('GROQ_VISION_MODEL',str),
            'gemini_api_key':('GEMINI_API_KEY',str),
            'gemini_model':('GEMINI_MODEL',str),
            'stability_seconds':('FILE_STABILITY_SECONDS',float),
            'stability_timeout':('FILE_STABILITY_TIMEOUT',float),
            'max_retries':('MAX_RETRIES',int),
            'shortening_retries':('SHORTENING_RETRIES',int),
            'max_lines':('MAX_SUBTITLE_LINES',int),
            'max_chars':('MAX_CHARS_PER_LINE',int),
            'batch_size':('TRANSLATION_BATCH_SIZE',int),
            'api_timeout':('API_TIMEOUT',float),
            'command_timeout':('COMMAND_TIMEOUT',float),
            'ffmpeg':('FFMPEG_PATH',str),
            'ffprobe':('FFPROBE_PATH',str),
            'ocr_provider':('OCR_PROVIDER',str),
            'ocr_language':('OCR_LANGUAGE',str),
            'ocr_interval':('OCR_SAMPLE_SECONDS',float),
            'ocr_region':('OCR_REGION',str),
            'ocr_target_box':('OCR_TARGET_BOX',str),
            'overlay_patch_color':('OVERLAY_PATCH_COLOR',str),
            'box_text_color':('BOX_TEXT_COLOR',str),
            'removal_mode':('REMOVAL_MODE',str),
            'font_name':('FONT_NAME',str),
            'font_size':('FONT_SIZE',int),
            'subtitle_position':('SUBTITLE_POSITION',str),
            'outline_width':('OUTLINE_WIDTH',float),
            'shadow_width':('SHADOW_WIDTH',float),
            'min_subtitle_duration':('MIN_SUBTITLE_DURATION',float),
            'subtitle_stream':('SUBTITLE_STREAM',int),
            'max_words_per_segment':('MAX_WORDS_PER_SEGMENT',int)
        }
        values = {k: cast(os.environ[env]) for k,(env,cast) in mapping.items() if os.getenv(env)}
        if 'translation_provider' in values:
            values['translation_provider'] = values['translation_provider'].lower()
        if os.getenv('CLAUDE_MODEL') and 'model' not in values:
            values['model'] = os.environ['CLAUDE_MODEL']
        if values.get('translation_provider') == 'gemini':
            if not values.get('gemini_model') and values.get('model'):
                values['gemini_model'] = values['model']
        for key,env in [('ocr_bg_variance_threshold','OCR_BG_VARIANCE_THRESHOLD')]:
            if env in os.environ: values[key]=float(os.environ[env])
        for key,env in [('ocr_enabled','OCR_ENABLED'),('overlay_patch','OVERLAY_PATCH'),('ocr_solid_bg_only','OCR_SOLID_BG_ONLY'),('pill_mode','PILL_MODE')]:
            if env in os.environ:
                if os.environ[env].lower() not in ('true','false'): raise ValueError(f'{env} must be true or false')
                values[key] = os.environ[env].lower() == 'true'
        if target_language: values['target_language'] = target_language
        if source_language: values['source_language'] = source_language
        return cls(**values)
