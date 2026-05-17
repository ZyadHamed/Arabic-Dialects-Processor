import { Component, ElementRef, ViewChild, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DialectApiService } from '../../services/dialect-api.service';
import { firstValueFrom } from 'rxjs'; // <-- Add this import

interface Word { text: string; time: number; shown: boolean; }

@Component({
  selector: 'app-transcription',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './transcription.component.html',
  styleUrls: ['./transcription.component.css']
})
export class TranscriptionComponent implements OnDestroy {
  @ViewChild('audioEl') audioEl!: ElementRef<HTMLAudioElement>;

  fileName = '';
  audioUrl = '';
  isTranscribing = false;
  isPlaying = false;
  fullText = '';
  words: Word[] = [];
  shownWords: Word[] = [];
  private timers: any[] = [];

  constructor(
  private api: DialectApiService, 
  private cdr: ChangeDetectorRef // <-- Inject it here
) {}

  async onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    this.clearState();
    this.fileName = file.name;
    this.audioUrl = URL.createObjectURL(file);
    await this.transcribe(file);
  }

async transcribe(file: File) {
  this.isTranscribing = true;
  this.cdr.detectChanges(); // Force the "Transcribing..." UI to show immediately

  try {
    // Cast to 'any' temporarily so TypeScript doesn't yell at you if the shape is different
    const res = await firstValueFrom(this.api.transcribe(file)) as any; 
    
    // LOOK AT THIS LOG IN YOUR BROWSER CONSOLE
    console.log('RAW BACKEND RESPONSE:', res); 
    
    // Safely grab the text depending on what your backend ACTUALLY returned
    this.fullText = res.transcription;
    
    this.buildWords(this.fullText);
    
  } catch (e) {
    console.error('Transcription crash:', e);
    this.fullText = '(Transcription failed – check console for errors)';
  } finally {
    this.isTranscribing = false;
    this.cdr.detectChanges(); // <-- FORCE THE DOM TO SHOW THE TEXT
  }
}

private rafId: number | null = null;

buildWords(text: string) {
  const raw = text.trim().split(/\s+/);
  // We don't know real duration yet — use a placeholder; 
  // it gets recalculated in syncWords() once audio metadata loads
  this.words = raw.map((w, i) => ({ text: w, time: i, shown: false }));
  this.shownWords = [];
}

onPlay() {
  const audio = this.audioEl?.nativeElement;
  if (!audio) return;
  this.isPlaying = true;

  const dur = audio.duration || 30;
  const raw = this.words.map(w => w.text);
  this.words = raw.map((text, i) => ({
    text,
    time: (i / raw.length) * dur,
    shown: false  // ← reset shown flags on every play
  }));
  this.shownWords = [];

  this.clearTimers();
  this.syncWords();
}

syncWords() {
  const audio = this.audioEl?.nativeElement;
  if (!audio || !this.isPlaying) return;

  const currentTime = audio.currentTime;
  let changed = false;

  for (const w of this.words) {
    if (!w.shown && w.time <= currentTime) {
      w.shown = true;
      this.shownWords = [...this.shownWords, w];
      changed = true;
    }
  }

  if (changed) {
    this.cdr.detectChanges(); // ← THIS is what was missing
  }

  this.rafId = requestAnimationFrame(() => this.syncWords());
}
onPause() {
  this.isPlaying = false;
  if (this.rafId !== null) {
    cancelAnimationFrame(this.rafId);
    this.rafId = null;
  }
}

onEnded() {
  this.isPlaying = false;
  if (this.rafId !== null) {
    cancelAnimationFrame(this.rafId);
    this.rafId = null;
  }
}


  clearState() {
    this.clearTimers();
    this.words = [];
    this.shownWords = [];
    this.fullText = '';
    this.isPlaying = false;
  }

  clearTimers() {
  this.timers.forEach(clearTimeout);
  this.timers = [];
  if (this.rafId !== null) {
    cancelAnimationFrame(this.rafId);
    this.rafId = null;
  }
}

onMetadataLoaded() {
  // Pre-calculate timings as soon as duration is known
  const audio = this.audioEl?.nativeElement;
  if (!audio || !this.words.length) return;
  const dur = audio.duration;
  this.words = this.words.map((w, i) => ({
    ...w,
    time: (i / this.words.length) * dur
  }));
}

  ngOnDestroy() {
    this.clearTimers();
    if (this.audioUrl) URL.revokeObjectURL(this.audioUrl);
  }
}
