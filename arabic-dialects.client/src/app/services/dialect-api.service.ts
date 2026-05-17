import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PredictResult {
  filename: string;
  predicted_dialect: string;
  confidence: number;
    features?: Record<string, number>; // add this

}

export interface DialectVizData {
  color: string;
  train_target_points: { x: number[]; y: number[]; z: number[] };
  train_other_points:  { x: number[]; y: number[]; z: number[] };
  hyperplane:          { x: number[][]; y: number[][]; z: number[][] };
  test_sample:         { x: number[]; y: number[]; z: number[] };
}

export type VisualizeResult = {
  dialects: Record<string, DialectVizData>;
  lda_top_features?: Record<string, { feature_name: string; weight: number }[]>;
}

export interface TranscribeResult {
  text: string;
}

export interface TranslateResult {
  translated_text: string;
}

const DIALECT_LABEL_MAP: Record<string, string> = {
  'Egyptian': 'المصرية',
  'Levantine': 'السورية',
  'Maghrebi': 'المغربية',
  'Gulf': 'السعودية',
};

@Injectable({ providedIn: 'root' })
export class DialectApiService {
  private base = 'http://127.0.0.1:8000';

  constructor(private http: HttpClient) {}

  predict(file: File): Observable<PredictResult> {
    const fd = new FormData();
    fd.append('file', file);
    return this.http.post<PredictResult>(`${this.base}/predict`, fd);
  }

  visualize(features: Record<string, number>): Observable<VisualizeResult> {
  return this.http.post<VisualizeResult>(`${this.base}/visualize_prediction`, features);
}

  transcribe(file: File): Observable<TranscribeResult> {
    const fd = new FormData();
    fd.append('file', file);
    return this.http.post<TranscribeResult>(`${this.base}/transcribe`, fd);
  }

tts(text: string, dialect: string, speakerId: number): Observable<Blob> {
  return this.http.post(`${this.base}/tts`, {
    text,
    dialect,          // ← pass through as-is, mapping already done in the component
    speaker_id: speakerId
  }, { responseType: 'blob' });
}

  translate(text: string, original: string, target: string): Observable<TranslateResult> {
    return this.http.post<TranslateResult>(`${this.base}/translate`, {
      text,
      original_dialect: DIALECT_LABEL_MAP[original] ?? original,
      target_dialect: DIALECT_LABEL_MAP[target] ?? target,
    });
  }

  getDialectLabel(key: string): string {
    return DIALECT_LABEL_MAP[key] ?? key;
  }
}
