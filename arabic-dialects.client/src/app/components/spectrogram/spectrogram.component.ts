import {
  Component, ElementRef, ViewChild, OnDestroy, ChangeDetectorRef 
} from '@angular/core';
import { CommonModule} from '@angular/common';
import { Subscription } from 'rxjs';
import { AudioUtilService } from '../../services/audio-util.service';

export type SpectrogramMode = 'stft' | 'mel';

@Component({
  selector: 'app-spectrogram',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './spectrogram.component.html',
  styleUrls: ['./spectrogram.component.css']
})
export class SpectrogramComponent implements OnDestroy {
  @ViewChild('spectroCanvas') spectroCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('waveCanvas')   waveCanvas!:   ElementRef<HTMLCanvasElement>;
  @ViewChild('audioPlayer')  audioPlayer!:  ElementRef<HTMLAudioElement>;

  fileName   = '';
  duration   = 0;
  sampleRate = 0;
  hasFile    = false;
  audioUrl   = '';
  audioBuffer: AudioBuffer | null = null;

  isLoadingWave    = false;
  isLoadingSpectro = false;
  spectroMode: SpectrogramMode = 'stft';

  private currentFile:   File | null = null;
  private spectroSub:    Subscription | null = null;

constructor(private audioUtil: AudioUtilService, private cdr: ChangeDetectorRef) {}

  // ─── File selection ───────────────────────────────────────────────────────

  async onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    const file  = input.files?.[0];
    if (!file) return;

    this.currentFile = file;
    this.fileName    = file.name;
    this.hasFile     = false;           // hide canvases while loading
    this.cdr.detectChanges(); // ← flush *ngIf so canvases are stamped into DOM
    if (this.audioUrl) URL.revokeObjectURL(this.audioUrl);
    this.audioUrl = URL.createObjectURL(file);

    this.isLoadingWave = true;
    try {
      this.audioBuffer = await this.audioUtil.decodeFile(file);
      this.duration    = this.audioBuffer.duration;
      this.sampleRate  = this.audioBuffer.sampleRate;
      this.hasFile     = true;  

      this.cdr.detectChanges();

      this.audioUtil.drawWaveform(this.waveCanvas.nativeElement, this.audioBuffer!);
      this.drawSpectrogram();
    } catch (e) {
      console.error('Failed to decode audio file', e);
    } finally {
      this.isLoadingWave = false;
    }
  }

  // ─── Spectrogram mode toggle ──────────────────────────────────────────────

  setMode(mode: SpectrogramMode) {
    if (mode === this.spectroMode) return;
    this.spectroMode = mode;
    this.drawSpectrogram();
  }

  // ─── Draw helpers ─────────────────────────────────────────────────────────

  private drawSpectrogram() {
    if (!this.currentFile) return;

    // Cancel any in-flight request
    this.spectroSub?.unsubscribe();
    this.isLoadingSpectro = true;

    const canvas = this.spectroCanvas.nativeElement;
    const draw$  = this.spectroMode === 'stft'
      ? this.audioUtil.drawSpectrogram(canvas, this.currentFile)
      : this.audioUtil.drawMelSpectrogram(canvas, this.currentFile, { nMels: 128 });

this.spectroSub = draw$.subscribe({
    next: () => { 
      this.isLoadingSpectro = false; 
      this.cdr.detectChanges(); // <-- Tell Angular to update the view
    },
    error: (e) => { 
      this.isLoadingSpectro = false; 
      this.cdr.detectChanges(); // <-- Clear the loader on error too
      console.error('Spectrogram error', e); 
    },
  });
  }

  // ─── Utilities ────────────────────────────────────────────────────────────

  formatDuration(s: number): string {
    const m   = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
  }

  ngOnDestroy() {
    this.spectroSub?.unsubscribe();
    if (this.audioUrl) URL.revokeObjectURL(this.audioUrl);
  }
}