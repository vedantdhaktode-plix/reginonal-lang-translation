import copy
import json
import logging
import os
import random
import re
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from models.subtitle import BoundingBox
from pipeline.video_analyzer import analyze
from pipeline.subtitle_detector import detect
from pipeline.subtitle_extractor import extract
from pipeline.burned_subtitle_ocr import extract_burned
from pipeline.translator import ClaudeTranslationProvider, GeminiTranslationProvider, GroqTranslationProvider, translate, detect_source_language
from pipeline.subtitle_style import assign_styles, generate
from pipeline.subtitle_renderer import render
from pipeline.ffmpeg_utils import output_filename
from pipeline.qc import check

class WorkerLock:
    """OS-owned lock releases on process exit, including abnormal termination."""
    def __init__(self,path): self.path=path
    def __enter__(self):
        self.file=self.path.open('a+b')
        self.file.seek(0); self.file.write(b'0'); self.file.flush(); self.file.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError('Another processor is running. Stop it before starting this command.')
        return self
    def __exit__(self,*args): self.file.close()


def protect_compounds(text):
    compounds = [
        r'\bVitamin\s+[A-Za-z0-9]+[,\.]?',
        r'\bHyaluronic\s+Acid[,\.]?',
        r'\bPlix\s+Glowy\s+Skin[,\.]?',
        r'\bantioxidant\s+support[,\.]?',
        r'\bglowy\s+looking[,\.]?',
        r'\ba\s+few\s+days[,\.]?',
        r'\bdon[\'’]t\s+try\b[,\.]?'
    ]
    protected = text
    for pat in compounds:
        protected = re.sub(pat, lambda m: m.group(0).replace(' ', '\uE000'), protected, flags=re.IGNORECASE)
    return protected

def chunk_segments_to_word_limit(segments, max_words=4):
    chunked = []
    for s in segments:
        text = (s.translated_text or '').strip()
        words = text.split()

        # Pre-split on clause boundaries before doing word chunking.
        # This ensures e.g. "This is Plix Glowy Skin, it contains L-glutathione"
        # becomes two cards rather than getting chunked mid-phrase.
        clause_pat = r',\s*(?=(?:it\s+has|it\s+contains|contains|with|and\s+it|which|that\s+has)\b)'
        raw_clauses = [c.strip() for c in re.split(clause_pat, text, flags=re.IGNORECASE) if c.strip()]

        if len(raw_clauses) > 1:
            # Distribute time proportionally across clauses, then chunk each clause by word limit
            total_len = sum(len(c) for c in raw_clauses)
            total_dur = max(0.5, s.end - s.start)
            clause_start = s.start
            for ci, clause in enumerate(raw_clauses):
                if ci == len(raw_clauses) - 1:
                    clause_end = s.end
                else:
                    share = (len(clause) / total_len) * total_dur
                    clause_end = round(clause_start + share, 2)
                sub = copy.deepcopy(s)
                sub.start = clause_start
                sub.end = clause_end
                sub.translated_text = clause
                clause_words = clause.split()
                if len(clause_words) <= max_words:
                    chunked.append(sub)
                else:
                    # Chunk this clause further by word limit
                    protected = protect_compounds(clause)
                    tokens = [w.replace('\uE000', ' ') for w in protected.split()]
                    word_chunks = []
                    cur = []
                    cur_words = 0
                    for tok in tokens:
                        w_cnt = len(tok.split())
                        if cur and (cur_words + w_cnt > max_words):
                            word_chunks.append(' '.join(cur))
                            cur = [tok]
                            cur_words = w_cnt
                        else:
                            cur.append(tok)
                            cur_words += w_cnt
                    if cur:
                        word_chunks.append(' '.join(cur))
                    chunk_total_len = sum(len(wc) for wc in word_chunks)
                    chunk_dur = max(0.5, clause_end - clause_start)
                    wc_start = clause_start
                    for wi, wc in enumerate(word_chunks):
                        if wi == len(word_chunks) - 1:
                            wc_end = clause_end
                        else:
                            wc_share = (len(wc) / chunk_total_len) * chunk_dur
                            wc_end = round(wc_start + wc_share, 2)
                        wsub = copy.deepcopy(s)
                        wsub.start = wc_start
                        wsub.end = wc_end
                        wsub.translated_text = wc
                        chunked.append(wsub)
                        wc_start = wc_end
                clause_start = clause_end
            continue

        if len(words) <= max_words:
            chunked.append(copy.deepcopy(s))
            continue

        protected = protect_compounds(text)
        tokens = [w.replace('\uE000', ' ') for w in protected.split()]
        
        if len(tokens) <= 1:
            chunked.append(copy.deepcopy(s))
            continue
            
        word_chunks = []
        cur = []
        cur_words = 0
        for tok in tokens:
            w_cnt = len(tok.split())
            if cur and (cur_words + w_cnt > max_words):
                word_chunks.append(' '.join(cur))
                cur = [tok]
                cur_words = w_cnt
            else:
                cur.append(tok)
                cur_words += w_cnt
        if cur:
            word_chunks.append(' '.join(cur))

        if len(word_chunks) <= 1:
            chunked.append(copy.deepcopy(s))
            continue

        total_len = sum(len(c) for c in word_chunks)
        total_dur = max(0.5, s.end - s.start)
        cur_start = s.start
        for i, c in enumerate(word_chunks):
            if i == len(word_chunks) - 1:
                cur_end = s.end
            else:
                share = (len(c) / total_len) * total_dur
                cur_end = round(cur_start + share, 2)
            sub = copy.deepcopy(s)
            sub.start = cur_start
            sub.end = cur_end
            sub.translated_text = c
            chunked.append(sub)
            cur_start = cur_end


    # Clean up empty segments
    chunked = [s for s in chunked if (s.translated_text or s.source_text or '').strip()]
    if not chunked:
        return segments

    # Ensure start times are ordered
    chunked.sort(key=lambda s: s.start)

    MIN_SEG_DUR = 0.25
    # First pass: enforce minimum duration and forward progression
    for i in range(len(chunked) - 1):
        if chunked[i+1].start < chunked[i].start + MIN_SEG_DUR:
            chunked[i+1].start = round(chunked[i].start + MIN_SEG_DUR, 2)
            if chunked[i+1].end <= chunked[i+1].start:
                chunked[i+1].end = round(chunked[i+1].start + MIN_SEG_DUR, 2)

        if chunked[i].end > chunked[i+1].start:
            chunked[i].end = chunked[i+1].start

    # Filter out any degenerate segment where end <= start
    clean_chunked = []
    for s in chunked:
        if s.end <= s.start:
            continue
        if s.end - s.start < 0.15:
            s.end = round(s.start + 0.15, 2)
        clean_chunked.append(s)
    chunked = clean_chunked

    # Bridge small gaps between consecutive segments (< 0.8s) so box doesn't flicker
    for i in range(len(chunked) - 1):
        gap = chunked[i+1].start - chunked[i].end
        if 0 < gap <= 0.8:
            chunked[i].end = chunked[i+1].start

    for i, s in enumerate(chunked, 1):
        s.id = i
    return chunked

def update_segment_boxes(segments, metadata, config):
    try:
        from PIL import ImageFont
        font_file = config.font_dir / 'Coco-Sharp-X-L-Bold.ttf'
        font = ImageFont.truetype(str(font_file), 42)
    except Exception:
        font = None

    meta_w = getattr(metadata, 'width', 1080)
    meta_h = getattr(metadata, 'height', 1080)

    def clamp_box(box):
        """Return valid integer geometry fully contained by the video frame."""
        width = max(1, min(meta_w, int(round(box.width))))
        height = max(1, min(meta_h, int(round(box.height))))
        x = max(0, min(meta_w - width, int(round(box.x))))
        y = max(0, min(meta_h - height, int(round(box.y))))
        return BoundingBox(x, y, width, height, patch_color=box.patch_color)

    # Detector samples can legitimately come from different shots.  A maximum
    # width/height combined with a position from another shot may cross an edge,
    # so normalize source geometry before it is used for layout decisions.
    for segment in segments:
        if segment.bbox:
            segment.bbox = clamp_box(segment.bbox)

    # Detect baseline container position and color from incoming segments if available
    detected_box = next((s.bbox for s in segments if s.bbox and s.bbox.height > 0), None)
    if detected_box:
        default_y = detected_box.y
        default_h = detected_box.height
        default_color = detected_box.patch_color or getattr(config, 'overlay_patch_color', '0x000000')
    else:
        default_y = int(meta_h * 0.881)
        default_h = int(meta_h * 0.080)
        default_color = getattr(config, 'overlay_patch_color', '0x000000')

    is_pill = getattr(config, 'pill_mode', False)

    for s in segments:
        # Keep source geometry intact when the renderer has separate, tight line
        # masks. Expanding the card box here would collapse the preserved gaps.
        if s.line_regions:
            continue
        cur_y = s.bbox.y if (s.bbox and s.bbox.height > 0) else default_y
        cur_h = s.bbox.height if (s.bbox and s.bbox.height > 0) else default_h
        cur_color = (s.bbox.patch_color if (s.bbox and s.bbox.patch_color) else default_color)

        txt = (s.translated_text or s.source_text or '').strip()
        tw = 400
        if font and txt:
            tw = font.getlength(txt)

        if is_pill:
            # Keep the mask anchored to the detected source location. The renderer
            # fits translated text inside this area; moving it to frame center can
            # expose source text when the original card is off-center.
            source_box = s.bbox
            if source_box and source_box.width > 0 and source_box.height > 0:
                # Tracker and agent boxes already include background margin. Do
                # not expand them again: bbox_track contains the original geometry,
                # and double padding makes the replacement spill outside the card.
                s.bbox = BoundingBox(source_box.x, source_box.y, source_box.width,
                                     source_box.height, patch_color=cur_color)
            else:
                pad_x = max(8, int(cur_h * 0.15))
                pad_y = max(4, int(cur_h * 0.08))
                source_w = min(int(meta_w * 0.85), max(440, int(tw + 120)))
                source_x = int((meta_w - source_w) / 2)
                bx = max(0, source_x - pad_x)
                right = min(meta_w, source_x + source_w + pad_x)
                bw = right - bx
                bh = min(meta_h, max(cur_h, 1) + 2 * pad_y)
                by = max(0, min(meta_h - bh, cur_y - pad_y))
                s.bbox = BoundingBox(bx, by, bw, bh, patch_color=cur_color)
        else:
            if s.bbox and s.bbox.width > 0 and s.bbox.height > 0:
                if s.bbox.y > meta_h * 0.4:
                    s.bbox = BoundingBox(0, s.bbox.y, meta_w, s.bbox.height, patch_color=s.bbox.patch_color)
                continue
            bw = min(int(meta_w * 0.85), max(440, int(tw + 120)))
            bx = int((meta_w - bw) / 2)
            s.bbox = BoundingBox(bx, cur_y, bw, cur_h, patch_color=cur_color)
    # FFmpeg clips an oversized overlay without reporting an error, but QC must
    # receive the same in-frame geometry that was actually intended.  Track
    # widths are metadata only; the rendered PNG uses the final base-box size.
    for segment in segments:
        if not segment.bbox:
            continue
        segment.bbox = clamp_box(segment.bbox)
        if segment.bbox_track:
            render_w = segment.bbox.width
            render_h = segment.bbox.height
            clamped_track = []
            for point in segment.bbox_track:
                timestamp, x, y, _width, _height = point
                track_x = max(0, min(meta_w - render_w, int(round(x))))
                track_y = max(0, min(meta_h - render_h, int(round(y))))
                clamped_track.append((timestamp, track_x, track_y, render_w, render_h))
            segment.bbox_track = clamped_track
    return segments

def export_and_review_script(segments, job, config):
    source_lang = detect_source_language(segments, getattr(job, 'source_language', ''))
    job.source_language = source_lang

    lines = [
        "=" * 80,
        f"TRANSLATED SCRIPT REVIEW | Job: {job.job_id}",
        f"Video: {job.filename} | Source: {source_lang} | Target: {job.target_language} | Total Segments: {len(segments)}",
        "=" * 80,
    ]
    srt_entries = []

    def format_srt_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def format_review_time(seconds):
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:05.2f}"

    for i, s in enumerate(segments, 1):
        lines.append(f"[{format_review_time(s.start)} -> {format_review_time(s.end)}]")
        lines.append(f"  Original ({source_lang}):  {s.source_text}")
        lines.append(f"  {job.target_language}:         {s.translated_text}")
        lines.append("-" * 80)
        
        # Dual-language SRT format for side-by-side verification
        src_text = " ".join(s.source_text.strip().split()) if s.source_text else ""
        tgt_text = " ".join(s.translated_text.strip().split()) if s.translated_text else ""
        if src_text and tgt_text:
            cue_body = f"[{source_lang}]: {src_text}\n[{job.target_language}]: {tgt_text}"
        elif tgt_text:
            cue_body = f"[{job.target_language}]: {tgt_text}"
        else:
            cue_body = f"[{source_lang}]: {src_text}"
            
        srt_entries.append(f"{i}\n{format_srt_time(s.start)} --> {format_srt_time(s.end)}\n{cue_body}\n")

    review_text = "\n".join(lines) + "\n"
    srt_text = "\n".join(srt_entries) + "\n"

    try:
        print("\n" + review_text, flush=True)
    except Exception:
        logging.getLogger(__name__).info("Translated script:\n%s", review_text)

    try:
        config.script_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"{Path(job.filename).stem}_{job.target_language}"
        (config.script_dir / f"{base_name}.txt").write_text(review_text, encoding="utf-8")
        
        # Write SRT; if locked/open in another process, write to job-specific file as fallback
        srt_target = config.script_dir / f"{base_name}.srt"
        try:
            srt_target.write_text(srt_text, encoding="utf-8")
        except PermissionError:
            fallback_target = config.script_dir / f"{base_name}_{job.job_id[:8]}.srt"
            fallback_target.write_text(srt_text, encoding="utf-8")
            logging.getLogger(__name__).warning("Main SRT file locked; saved verification SRT to %s", fallback_target.name)

        (config.script_dir / f"{base_name}.json").write_text(
            json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logging.getLogger(__name__).info("Saved translated scripts to %s (.txt, .srt, .json)", config.script_dir)
    except Exception as exc:
        logging.getLogger(__name__).warning("Could not export script files: %s", exc)

class Processor:
    def __init__(self,config,db,provider=None):
        self.config=config; self.db=db; self.provider=provider

    def stage(self,job,stage,action):
        self.db.update(job.job_id,stage)
        log=logging.LoggerAdapter(logging.getLogger(__name__),{'job_id':job.job_id,'stage':stage})
        for attempt in range(self.config.max_retries+1):
            try:
                log.info('Starting stage (attempt %s)',attempt+1)
                return action()
            except Exception as exc:
                log.exception('Stage failed')
                if isinstance(exc,ImportError):
                    log.error('Dependency missing in the running Python environment; retries cannot install it')
                    raise
                if attempt==self.config.max_retries: raise
                status=getattr(exc,'status_code',None) or getattr(exc,'code',None)
                if stage=='TRANSLATING' and status in (400,401,403,404):
                    log.error('Translation request/configuration error; retrying unchanged will not help')
                    raise
                current=self.db.get(job.job_id)
                self.db.update(job.job_id,retry_count=current.retry_count+1)
                delay=min(2**attempt,10)
                if stage=='TRANSLATING' and status in (429,500,502,503,504,529):
                    delay=min(30*2**attempt+random.uniform(0,5),120)
                log.warning('Retrying %s in %.1f seconds',stage,delay)
                time.sleep(delay)

    def process(self,job):
        config=self.config
        folder=Path(job.processing_path).parent
        video=Path(job.processing_path)
        if folder.parent!=config.processing_dir or folder.name!=job.job_id:
            self.db.update(job.job_id,'FAILED',error_message='Unsafe processing path in database')
            return False
        try:
            metadata=self.stage(job,'ANALYZING',lambda:analyze(video,config))
            self.db.update(job.job_id,metadata_json=json.dumps(asdict(metadata)))
            track=detect(metadata,config)
            if track:
                segments,document=self.stage(job,'EXTRACTING_SUBTITLES',lambda:extract(video,track,folder,config))
            else:
                segments=self.stage(job,'OCR',lambda:extract_burned(video,metadata,config)); document=None
            if not segments:
                destination=config.completed_dir/output_filename(job.filename,job.target_language)
                shutil.copy2(video, destination)
                self.db.update(job.job_id,'COMPLETED',output_path=str(destination),error_message=None)
                return True
            detected_src = detect_source_language(segments, job.source_language, track)
            if detected_src and (not job.source_language or job.source_language.lower() in ('auto', 'und', 'unknown')):
                job.source_language = detected_src
                try: self.db.update(job.job_id, source_language=detected_src)
                except Exception: pass
            self.stage(job,'STYLE_ANALYSIS',lambda:assign_styles(segments,metadata,config,document))
            def translation():
                if self.provider is None:
                    if config.translation_provider == 'gemini':
                        self.provider = GeminiTranslationProvider(config)
                    elif config.translation_provider == 'groq':
                        self.provider = GroqTranslationProvider(config)
                    else:
                        self.provider = ClaudeTranslationProvider(config)
                return translate(segments,self.provider,config,job.source_language,job.target_language)
            segments=self.stage(job,'TRANSLATING',translation)
            is_agent = (getattr(config, 'ocr_target_box', '').lower() == 'agent' or
                        any(s.bbox and s.bbox.width > 0 for s in segments))
            if not is_agent:
                segments=chunk_segments_to_word_limit(segments, max_words=getattr(config, 'max_words_per_segment', 4))
            if not track:
                from pipeline.subtitle_renderer import detect_line_regions
                segments=self.stage(job,'STYLE_ANALYSIS',lambda:detect_line_regions(video,segments,config))
            originals=copy.deepcopy(segments)
            segments=update_segment_boxes(segments, metadata, config)
            self.stage(job,'STYLE_ANALYSIS',lambda:assign_styles(segments,metadata,config,document))
            export_and_review_script(segments,job,config)
            def styling():
                return generate(segments,metadata,config,folder,document)
            subtitles,warnings=self.stage(job,'STYLE_ANALYSIS',styling)
            if not track:
                warnings.extend(['Burned-in source text uses overlay only; background is not reconstructed.',f'OCR timing resolution is approximately {config.ocr_interval}s.'])
            (folder/'segments.json').write_text(json.dumps([asdict(s) for s in segments],ensure_ascii=False,indent=2),encoding='utf-8')
            output=self.stage(job,'RENDERING',lambda:render(video,subtitles,segments,config,folder,burned=not track,metadata=metadata))
            qc_segments = [s for s in segments if s.end > s.start + 0.01]
            self.stage(job,'QC',lambda:check(metadata,output,originals,qc_segments,config,config.metadata_dir/f'{job.job_id}_qc.json',warnings))
            destination=config.completed_dir/output_filename(job.filename,job.target_language)
            # Copy to the destination filesystem first, then publish atomically without overwriting.
            staging=config.completed_dir/f'.{job.job_id}.partial'
            try:
                with staging.open('wb') as handle,output.open('rb') as stream:
                    shutil.copyfileobj(stream,handle)
                    handle.flush(); os.fsync(handle.fileno())
                try: os.link(staging,destination)
                except FileExistsError:
                    destination=destination.with_name(f'{destination.stem}_{job.job_id}.mp4')
                    destination.unlink(missing_ok=True)
                    os.link(staging,destination)
            finally: staging.unlink(missing_ok=True)
            output.unlink()
            self.db.update(job.job_id,'COMPLETED',output_path=str(destination),error_message=None)
            return True
        except Exception as exc:
            current=self.db.get(job.job_id)
            details={'job_id':job.job_id,'stage':current.current_stage,'error':str(exc),'traceback':traceback.format_exc(),'timestamp':time.time()}
            try:
                folder.mkdir(parents=True,exist_ok=True)
                (folder/'failure.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
                failed=config.failed_dir/job.job_id
                if failed.exists(): raise RuntimeError(f'Failure destination already exists: {failed}')
                shutil.move(str(folder),str(failed))
                self.db.update(job.job_id,processing_path=str(failed/video.name))
            except Exception: logging.getLogger(__name__).exception('Could not archive failed job')
            self.db.update(job.job_id,'FAILED',error_message=str(exc))
            self.db.update(job.job_id,current_stage=current.current_stage)
            return False

    def recover(self):
        # Called only with the exclusive worker lock. Never reclaim live workers.
        for job_id in self.db.unfinished():
            job=self.db.get(job_id)
            if job.status in ('DETECTED','WAITING_FOR_FILE'): continue  # ingestion may run separately
            self.db.update(job_id,'QUEUED',error_message='Recovered interrupted worker; stages will restart')

    def drain(self,stop=None,continuous=False):
        failures=0
        with WorkerLock(self.config.root/'database/worker.lock'):
            self.recover()
            while stop is None or not stop.is_set():
                job=self.db.claim()
                if job: failures+=not self.process(job)
                elif not continuous: break
                elif stop: stop.wait(0.5)
        return failures

def retry_job(job_id,config,db):
    with WorkerLock(config.root/'database/ingestion.lock'):
        return _retry_job(job_id,config,db)

def _retry_job(job_id,config,db):
    job=db.get(job_id)
    source=Path(job.processing_path)
    if not source.is_file(): raise ValueError('Source video is missing; restore it before retrying')
    if source.parent.name!=job_id or source.parent.parent not in (config.failed_dir,config.processing_dir): raise ValueError('Unsafe retry path')
    db.reserve_retry(job_id)
    destination=config.processing_dir/job_id
    try:
        if source.parent!=destination:
            if destination.exists(): raise ValueError('Processing destination already exists')
            shutil.move(str(source.parent),str(destination))
        db.update(job_id,'QUEUED',processing_path=str(destination/source.name))
    except Exception as exc:
        db.update(job_id,'FAILED',error_message=str(exc))
        raise
