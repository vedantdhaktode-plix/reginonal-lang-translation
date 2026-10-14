import copy
import logging
import re
import pysubs2
from models.subtitle import SubtitleStyle

_LANGUAGE_FONTS = {
    'marathi':   'NotoSansDevanagari-Regular',
    'hindi':     'NotoSansDevanagari-Regular',
    'kannada':   'NotoSansKannada-Regular',
    'tamil':     'NotoSansTamil-Regular',
    'telugu':    'NotoSansTelugu-Regular',
    'malayalam': 'NotoSansMalayalam-Regular',
}

def choose_font(config, original=None):
    if original: return original  # libass performs system fallback when absent
    fonts=sorted([*config.font_dir.glob('*.ttf'),*config.font_dir.glob('*.otf')])
    preferred=_LANGUAGE_FONTS.get(getattr(config,'target_language','').lower())
    if preferred:
        match=next((f for f in fonts if f.stem.casefold()==preferred.casefold()),None)
        if match:
            try:
                from PIL import ImageFont
                return ImageFont.truetype(str(match),24).getname()[0]
            except OSError: pass
    if config.font_name:
        fn = config.font_name.strip()
        if fn.lower() in ('coco sharp', 'cocosharp', 'coco-sharp', 'cocosharp xl', 'coco sharp xl'):
            return 'CocoSharp XL'
        return fn
    if fonts:
        try:
            from PIL import ImageFont
            return ImageFont.truetype(str(fonts[0]),24).getname()[0]
        except OSError: pass
    logging.getLogger(__name__).warning('Font fallback: Arial (libass may substitute a system font)')
    return 'Arial'

def assign_styles(segments, metadata, config, document=None):
    font=choose_font(config)
    for segment in segments:
        if document is not None:
            event=document.events[segment.event_index]
            original=document.styles.get(event.style,document.styles.get('Default'))
            scale_y=metadata.height/float(document.info.get('PlayResY',metadata.height))
            scale_x=metadata.width/float(document.info.get('PlayResX',metadata.width))
            segment.style=SubtitleStyle(font_name=original.fontname,font_size=original.fontsize*scale_y,
                                       bold=original.bold,italic=original.italic,outline_width=original.outline*scale_y,
                                       shadow=original.shadow*scale_y,alignment=int(original.alignment))
            segment.fit_width=max(1,metadata.width-(event.marginl or original.marginl)*scale_x-(event.marginr or original.marginr)*scale_x)
            segment.fit_height=metadata.height
            continue
        if segment.style: continue
        box=segment.bbox
        position = getattr(config, 'subtitle_position', 'bottom').lower()
        cfg_size = getattr(config, 'font_size', 0)
        base_size = float(cfg_size) if cfg_size > 0 else max(36.0, round(metadata.height * 0.045))
        outline_w = getattr(config, 'outline_width', 0.0)
        shadow_w = getattr(config, 'shadow_width', 1.0)
        if position == 'box' and box:
            box_text_color = getattr(config, 'box_text_color', '#000000')
            patch_hex = getattr(box, 'patch_color', '') or getattr(config, 'overlay_patch_color', '')
            if patch_hex:
                clean = patch_hex.lstrip('#').lower().replace('0x', '')
                if len(clean) == 6:
                    r, g, b = int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16)
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    if lum < 100 and box_text_color.lower() in ('#000000', 'black'):
                        box_text_color = '#FFFFFF'
            # Caption-card boxes include background padding. A lower ratio keeps
            # Indic replacements close to the source glyph height instead of
            # scaling them to the full panel height.
            height_ratio = 0.30 if getattr(config, 'pill_mode', False) else 0.46
            calc_size = max(24.0, min(metadata.height * 0.054, box.height * height_ratio))
            center_x = float(box.x) + float(box.width) / 2.0
            center_y = float(box.y) + float(box.height) / 2.0
            outline_col = '#000000' if box_text_color.upper() == '#FFFFFF' else box_text_color
            segment.style=SubtitleStyle(font_name=font,font_size=calc_size,
                                        bold=True,primary_color=box_text_color,outline_color=outline_col,
                                        outline_width=1.5,shadow=0.0,alignment=5,
                                        x=center_x,y=center_y)
            if getattr(config, 'pill_mode', False):
                segment.fit_width = metadata.width * 0.90
                segment.fit_height = max(float(box.height) * 1.5, metadata.height * 0.15)
            else:
                segment.fit_width=float(box.width) * 0.92
                segment.fit_height=float(box.height) * 0.90
        elif position == 'top':
            segment.style=SubtitleStyle(font_name=font,font_size=base_size,outline_width=outline_w,shadow=shadow_w,alignment=8,x=None,y=None)
            segment.fit_width=metadata.width*0.90
            segment.fit_height=metadata.height*0.30
        else:
            segment.style=SubtitleStyle(font_name=font,font_size=base_size,bold=True,outline_width=outline_w,shadow=shadow_w,alignment=2,x=None,y=None)
            segment.fit_width=metadata.width*0.90
            segment.fit_height=metadata.height*0.30

def measured_fit(text, segment, config):
    """Conservative glyph measurement with dynamic downscaling to fit inside the box."""
    if segment.style is None or segment.fit_width is None: return text
    from PIL import ImageFont
    padding=2*segment.style.outline_width+segment.style.shadow
    width=segment.fit_width-padding
    target_size=max(18,round(segment.style.font_size))
    candidates=[*config.font_dir.glob('*.ttf'),*config.font_dir.glob('*.otf')]
    for size in range(target_size,17,-2):
        font=None
        for candidate in candidates:
            try:
                loaded=ImageFont.truetype(str(candidate),size)
                if loaded.getname()[0].casefold()==segment.style.font_name.casefold(): font=loaded; break
            except OSError: continue
        if font is None:
            for candidate in (segment.style.font_name,segment.style.font_name+'.ttf','C:/Windows/Fonts/Nirmala.ttc','C:/Windows/Fonts/nirmala.ttf','C:/Windows/Fonts/arial.ttf','arial.ttf','DejaVuSans.ttf'):
                try: font=ImageFont.truetype(candidate,size); break
                except OSError: continue
        if font is None: font=ImageFont.load_default(size=size)
        lines=[]
        fits_words=True
        for paragraph in text.splitlines():
            line=''
            for word in paragraph.split():
                proposed=(line+' '+word).strip()
                if font.getlength(proposed)<=width: line=proposed
                else:
                    if line: lines.append(line)
                    if font.getlength(word)>width: fits_words=False; break
                    line=word
            if line: lines.append(line)
        if not fits_words or not lines or len(lines)>config.max_lines: continue
        try:
            ascent,descent=font.getmetrics()
            height=(ascent+descent)*len(lines)+padding
        except AttributeError: height=size*1.25*len(lines)+padding
        if segment.fit_height and height>segment.fit_height: continue
        segment.style.font_size=float(size)
        return '\n'.join(lines)
    return None

def parse_color(hex_str, default=pysubs2.Color(255, 255, 255, 0)):
    if not hex_str: return default
    clean = hex_str.lstrip('#')
    try:
        if len(clean) == 6:
            return pysubs2.Color(int(clean[0:2],16), int(clean[2:4],16), int(clean[4:6],16), 0)
        elif len(clean) == 8:
            return pysubs2.Color(int(clean[0:2],16), int(clean[2:4],16), int(clean[4:6],16), int(clean[6:8],16))
    except ValueError: pass
    return default

def normalize_devanagari(text):
    if not text: return text
    text = text.replace('\u0905\u0945', '\u0972')      # अ + ॅ -> ॲ
    text = text.replace('\u0905\u0949', '\u0911')      # अ + ॉ -> ऑ
    text = text.replace('\u0905\u093e\u0945', '\u0972')  # आ + ॅ -> ॲ
    text = text.replace('\u0905\u093e\u0949', '\u0911')  # आ + ॉ -> ऑ
    return text

def escape_text(text):
    # Prevent translated text from injecting ASS override commands and normalize combining marks.
    text = normalize_devanagari(text)
    return text.replace('\\','＼').replace('{','｛').replace('}','｝').replace('\n',r'\N')

def generate(segments, metadata, config, folder, document=None):
    doc=copy.deepcopy(document) if document is not None else pysubs2.SSAFile()
    warnings=[]
    if document is None:
        doc.info['PlayResX']=str(metadata.width)
        doc.info['PlayResY']=str(metadata.height)
        doc.info['WrapStyle']='2'
    doc.events=[]
    for segment in segments:
        if document is not None:
            event=copy.deepcopy(document.events[segment.event_index])
            # Preserve leading visual overrides, but lyric timing and drawing modes cannot be remapped.
            leading=re.match(r'^(?:\{[^}]*\})*',event.text).group()
            if re.search(r'\\(?:[kK][fo]?\d|t\(|move\()',event.text):
                warnings.append(f'Segment {segment.id}: translated text cannot preserve word-level karaoke/animation semantics')
            leading=re.sub(r'\\[kK][fo]?\d+','',leading)
            event.text=leading+escape_text(segment.translated_text)
        else:
            style=segment.style
            name=f'Segment{segment.id}'
            margin_v = int(metadata.height * 0.08) if style.y is None else 10
            p_color = parse_color(style.primary_color, pysubs2.Color(255, 255, 255, 0))
            o_color = parse_color(style.outline_color, pysubs2.Color(0, 0, 0, 0))
            doc.styles[name]=pysubs2.SSAStyle(fontname=style.font_name,fontsize=style.font_size,bold=style.bold,italic=style.italic,
                                            primarycolor=p_color,outlinecolor=o_color,
                                            outline=style.outline_width,shadow=style.shadow,alignment=pysubs2.Alignment(style.alignment),
                                            marginv=margin_v)
            prefix=rf'{{\pos({style.x:.2f},{style.y:.2f})}}' if style.x is not None else ''
            event=pysubs2.SSAEvent(start=round(segment.start*1000),end=round(segment.end*1000),style=name,text=prefix+escape_text(segment.translated_text))
        doc.events.append(event)
    # SRT keeps millisecond timestamps; ASS only supports centiseconds.
    exact_ms=any(round(s.start*1000)%10 or round(s.end*1000)%10 for s in segments)
    plain=document is not None and document.format=='srt'
    suffix='srt' if plain and exact_ms else 'ass'
    if suffix=='ass' and exact_ms: warnings.append('ASS rendering quantizes timestamps to centiseconds; source timing remains exact in metadata')
    path=folder/f'translated.{suffix}'
    doc.save(str(path),encoding='utf-8')
    if document is not None:
        warnings.append('ASS font resolution uses libass fallback; exact fonts and animated text extents require visual review')
    return path,warnings
