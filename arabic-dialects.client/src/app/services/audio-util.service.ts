import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map, switchMap } from 'rxjs/operators';
export type SpectrogramType = 'stft' | 'mel';

export interface MelSpectrogramOptions {
  nMels?: number;   // default 128
  fmin?: number;    // default 0
  fmax?: number;    // default sr/2 — omit to let backend decide
}

@Injectable({ providedIn: 'root' })
export class AudioUtilService {

  private readonly base = 'http://127.0.0.1:8000';

  constructor(private http: HttpClient) {}

  getSpectrogramUrl(file: File): Observable<string> {
    return this.http
      .post(`${this.base}/spectrogram`, this._formData(file), { responseType: 'blob' })
      .pipe(map(blob => URL.createObjectURL(blob)));
  }

  getMelSpectrogramUrl(file: File, opts: MelSpectrogramOptions = {}): Observable<string> {
    let params = new HttpParams();
    if (opts.nMels !== undefined) params = params.set('n_mels', opts.nMels);
    if (opts.fmin  !== undefined) params = params.set('fmin',   opts.fmin);
    if (opts.fmax  !== undefined) params = params.set('fmax',   opts.fmax);

    return this.http
      .post(`${this.base}/mel-spectrogram`, this._formData(file), { params, responseType: 'blob' })
      .pipe(map(blob => URL.createObjectURL(blob)));
  }

  drawSpectrogram(canvas: HTMLCanvasElement, file: File): Observable<void> {
    return this.getSpectrogramUrl(file).pipe(
      switchMap(url => this._paintUrlToCanvas(canvas, url))
    );
  }

  drawMelSpectrogram(canvas: HTMLCanvasElement, file: File, opts: MelSpectrogramOptions = {}): Observable<void> {
    return this.getMelSpectrogramUrl(file, opts).pipe(
      switchMap(url => this._paintUrlToCanvas(canvas, url))
    );
  }

  // Converts a blob URL → draws it on the canvas → revokes the URL
  private _paintUrlToCanvas(canvas: HTMLCanvasElement, url: string): Observable<void> {
    return new Observable<void>(observer => {
      const img = new Image();
      img.onload = () => {
        canvas.width  = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.getContext('2d')!.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
        observer.next();
        observer.complete();
      };
      img.onerror = err => {
        URL.revokeObjectURL(url);
        observer.error(err);
      };
      img.src = url;
      // No teardown needed — image loading isn't cancellable anyway
    });
  }

  // ─── Waveform (still client-side — no server trip needed) ───────────────

  async decodeFile(file: File): Promise<AudioBuffer> {
    const ctx = new AudioContext();
    const ab = await file.arrayBuffer();
    return ctx.decodeAudioData(ab);
  }

  drawWaveform(canvas: HTMLCanvasElement, buffer: AudioBuffer, color = '#1a73e8'): void {
    const ctx = canvas.getContext('2d')!;
    const data = buffer.getChannelData(0);
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#f0f4f8';
    ctx.fillRect(0, 0, w, h);

    const step = Math.ceil(data.length / w);
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x++) {
      let min = 1, max = -1;
      for (let j = 0; j < step; j++) {
        const v = data[x * step + j] ?? 0;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      ctx.moveTo(x, ((1 - min) / 2) * h);
      ctx.lineTo(x, ((1 - max) / 2) * h);
    }
    ctx.stroke();
  }

  // ─── Audio blending helpers (unchanged, server-free) ────────────────────

  blendWeightedSum(a: AudioBuffer, b: AudioBuffer, weightA: number): AudioBuffer {
    const sampleRate = a.sampleRate;
    const length = Math.max(a.length, b.length);
    const ctx = new OfflineAudioContext(1, length, sampleRate);
    const out = new Float32Array(length);
    const dataA = a.getChannelData(0);
    const dataB = b.getChannelData(0);
    const wB = 1 - weightA;
    for (let i = 0; i < length; i++) {
      out[i] = (i < dataA.length ? dataA[i] : 0) * weightA
             + (i < dataB.length ? dataB[i] : 0) * wB;
    }
    const buf = ctx.createBuffer(1, length, sampleRate);
    buf.copyToChannel(out, 0);
    return buf;
  }

  blendTimeSplice(
    a: AudioBuffer,
    b: AudioBuffer,
    outputLengthSec: number,
    portionA: number,
    firstFile: 'a' | 'b'
  ): AudioBuffer {
    const sampleRate = a.sampleRate;
    const total = Math.floor(outputLengthSec * sampleRate);
    const sampA = Math.floor(total * portionA);
    const sampB = total - sampA;
    const dataA = a.getChannelData(0);
    const dataB = b.getChannelData(0);
    const out = new Float32Array(total);
    if (firstFile === 'a') {
      for (let i = 0; i < sampA && i < dataA.length; i++) out[i] = dataA[i];
      for (let i = 0; i < sampB && i < dataB.length; i++) out[sampA + i] = dataB[i];
    } else {
      for (let i = 0; i < sampB && i < dataB.length; i++) out[i] = dataB[i];
      for (let i = 0; i < sampA && i < dataA.length; i++) out[sampB + i] = dataA[i];
    }
    const ctx2 = new OfflineAudioContext(1, total, sampleRate);
    const buf = ctx2.createBuffer(1, total, sampleRate);
    buf.copyToChannel(out, 0);
    return buf;
  }

  audioBufferToWavBlob(buffer: AudioBuffer): Blob {
    const samples = buffer.getChannelData(0);
    const byteLength = 44 + samples.length * 2;
    const ab = new ArrayBuffer(byteLength);
    const view = new DataView(ab);
    const ws = (o: number, s: string) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
    ws(0, 'RIFF'); view.setUint32(4, byteLength - 8, true);
    ws(8, 'WAVE'); ws(12, 'fmt '); view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, buffer.sampleRate, true);
    view.setUint32(28, buffer.sampleRate * 2, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    ws(36, 'data'); view.setUint32(40, samples.length * 2, true);
    let off = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      off += 2;
    }
    return new Blob([ab], { type: 'audio/wav' });
  }

  // ─── Private ─────────────────────────────────────────────────────────────

  private _formData(file: File): FormData {
    const fd = new FormData();
    fd.append('file', file, file.name);
    return fd;
  }
}
