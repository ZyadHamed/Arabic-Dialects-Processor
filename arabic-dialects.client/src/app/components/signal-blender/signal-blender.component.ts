import { Component, ElementRef, ViewChild, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AudioUtilService } from '../../services/audio-util.service';
import { DialectApiService, PredictResult, VisualizeResult, DialectVizData } from '../../services/dialect-api.service';
import * as Plotly from 'plotly.js-dist-min';  // or use CDN version below

type BlendMode = 'weighted-sum' | 'time-splice';

interface FileSlot {
  file: File | null;
  buffer: AudioBuffer | null;
  name: string;
  duration: number;
  url: string;
}

interface DialectResult {
  label: string;
  arabic: string;
  value: number;
  color: string;
}
  const dialects = ['Egyptian', 'Levantine', 'Maghrebi', 'Gulf'];

@Component({
  selector: 'app-signal-blender',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './signal-blender.component.html',
  styleUrls: ['./signal-blender.component.css']
})
export class SignalBlenderComponent implements OnDestroy {
  @ViewChild('waveA') waveA!: ElementRef<HTMLCanvasElement>;
  @ViewChild('waveB') waveB!: ElementRef<HTMLCanvasElement>;
  @ViewChild('waveOut') waveOut!: ElementRef<HTMLCanvasElement>;

  slotA: FileSlot = { file: null, buffer: null, name: '', duration: 0, url: '' };
  slotB: FileSlot = { file: null, buffer: null, name: '', duration: 0, url: '' };

  blendMode: BlendMode = 'weighted-sum';

  // Mode A: weighted sum
  weightA = 50; // % for file A

  // Mode B: time splice
  maxOutputSec = 0;        // set to min(durA, durB)
  desiredOutputSec = 0;
  portionA = 50;           // % of output from file A
  firstFile: 'a' | 'b' = 'a';

  outputUrl = '';
  outputBlob: Blob | null = null;
  isBlending = false;
  isClassifying = false;
  hasResult = false;
  visualData: Record<string, DialectVizData> | null = null;
  ldaTopFeatures: Record<string, { feature_name: string; weight: number }[]> = {};
  results: DialectResult[] = [];
  topLabel = '';
  topArabic = '';
  topConf = 0;

  readonly dialectMeta: Record<string, { arabic: string; color: string }> = {
    Egyptian:  { arabic: 'المصرية', color: '#1a73e8' },
    Levantine: { arabic: 'السورية', color: '#34a853' },
    Maghrebi:  { arabic: 'المغربية', color: '#e8a82b' },
    Gulf:      { arabic: 'السعودية', color: '#9c27b0' },
  };

  constructor(
    private audioUtil: AudioUtilService,
    private api: DialectApiService,
    private cdr: ChangeDetectorRef
  ) {}


normalizeDialect(raw: string): string {
  return dialects.find(k =>
    raw.toLowerCase().includes(k.toLowerCase())
  ) ?? raw;
}
  async onFileSelected(event: Event, slot: 'a' | 'b') {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    const target = slot === 'a' ? this.slotA : this.slotB;
    if (target.url) URL.revokeObjectURL(target.url);
    target.file = file;
    target.name = file.name;
    target.url = URL.createObjectURL(file);
    target.buffer = await this.audioUtil.decodeFile(file);
    target.duration = target.buffer.duration;
    this.hasResult = false;
    this.outputUrl = '';

    // Update max output for time-splice
    if (this.slotA.buffer && this.slotB.buffer) {
      this.maxOutputSec = Math.min(this.slotA.duration, this.slotB.duration);
      this.desiredOutputSec = parseFloat(this.maxOutputSec.toFixed(1));
    }
    this.cdr.detectChanges();
    // Draw waveform
    setTimeout(() => {
      const canvas = slot === 'a' ? this.waveA?.nativeElement : this.waveB?.nativeElement;
      if (canvas && target.buffer) {
        this.audioUtil.drawWaveform(canvas, target.buffer, slot === 'a' ? '#1a73e8' : '#34a853');
      }
    }, 50);
  }

  async blend() {
    if (!this.slotA.buffer || !this.slotB.buffer) return;
    this.isBlending = true;
    this.hasResult = false;
    this.outputUrl = '';
    try {
      let blended: AudioBuffer;
      if (this.blendMode === 'weighted-sum') {
        blended = this.audioUtil.blendWeightedSum(
          this.slotA.buffer, this.slotB.buffer, this.weightA / 100
        );
      } else {
        blended = this.audioUtil.blendTimeSplice(
          this.slotA.buffer, this.slotB.buffer,
          this.desiredOutputSec, this.portionA / 100,
          this.firstFile
        );
      }
      this.outputBlob = this.audioUtil.audioBufferToWavBlob(blended);
      this.outputUrl = URL.createObjectURL(this.outputBlob);

      this.cdr.detectChanges();

      setTimeout(() => {
        if (this.waveOut?.nativeElement) {
          this.audioUtil.drawWaveform(this.waveOut.nativeElement, blended, '#9c27b0');
        }
      }, 50);

      await this.classify();
    } finally {
      this.isBlending = false;
    }
  }

async classify() {
  if (!this.outputBlob) return;
  this.isClassifying = true;
  this.cdr.detectChanges();
  try {
    const file = new File([this.outputBlob], 'blended.wav', { type: 'audio/wav' });
    const res = await this.api.predict(file).toPromise() as PredictResult;
    const winner = this.normalizeDialect(res.predicted_dialect);
this.topLabel = winner;
this.topArabic = this.dialectMeta[winner]?.arabic ?? '';
this.topConf = Math.round(res.confidence * 100);
this.buildBars(winner, res.confidence);

    if (res.features) {
      const vizResult = await this.api.visualize(res.features).toPromise() ?? null;
      this.visualData = vizResult?.dialects ?? null;
      this.ldaTopFeatures = vizResult?.lda_top_features ?? {};
    }

    this.hasResult = true;
    if (this.visualData) {
      this.cdr.detectChanges();
      this.renderPlots();
    }
  } catch (e) {
    console.error(e);
  } finally {
    this.isClassifying = false;
    this.cdr.detectChanges();
  }
}

renderPlots() {
  if (!this.visualData) return;
  Object.entries(this.visualData).forEach(([dialect, data], i) => {
    const traces: any[] = [
      {
        type: 'scatter3d', mode: 'markers', name: dialect,
        x: data.train_target_points.x, y: data.train_target_points.y, z: data.train_target_points.z,
        marker: { color: data.color, size: 4, opacity: 0.8 }
      },
      {
        type: 'scatter3d', mode: 'markers', name: 'Other',
        x: data.train_other_points.x, y: data.train_other_points.y, z: data.train_other_points.z,
        marker: { color: '#aaaaaa', size: 3, opacity: 0.3 }
      },
      {
        type: 'surface',
        x: data.hyperplane.x, y: data.hyperplane.y, z: data.hyperplane.z,
        opacity: 0.2, colorscale: [[0, data.color], [1, data.color]],
        showscale: false, name: 'Boundary'
      },
      {
        type: 'scatter3d', mode: 'markers', name: 'Your sample',
        x: data.test_sample.x, y: data.test_sample.y, z: data.test_sample.z,
        marker: { color: '#ffffff', size: 8, symbol: 'diamond', line: { color: data.color, width: 3 } }
      }
    ];
    Plotly.newPlot(`blender-plot-${i}`, traces, {
      title: { text: `${dialect} vs All`, font: { color: '#fff' } },
      paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
      scene: {
        xaxis: { title: { text: 'LD1' } as any, color: '#888' },
        yaxis: { title: { text: 'LD2' } as any, color: '#888' },
        zaxis: { title: { text: 'LD3' } as any, color: '#888' },
      },
      legend: { font: { color: '#ccc' } },
      margin: { t: 40, b: 0, l: 0, r: 0 }
    }, { responsive: true });
  });
}

  buildBars(winner: string, winConf: number) {
    const dialects = ['Egyptian', 'Levantine', 'Maghrebi', 'Gulf'];
    const rem = 1 - winConf;
    const others = dialects.filter(d => d !== winner);
    const weights = [0.5, 0.3, 0.2];
    this.results = dialects.map(d => {
      const val = d === winner ? winConf : rem * weights[others.indexOf(d)];
      return {
        label: d,
        arabic: this.dialectMeta[d].arabic,
        value: Math.round(val * 100),
        color: this.dialectMeta[d].color
      };
    }).sort((a, b) => b.value - a.value);
  }

  get canBlend() {
    return !!this.slotA.buffer && !!this.slotB.buffer;
  }

  get topColor() {
    return this.dialectMeta[this.topLabel]?.color ?? '#1a73e8';
  }

  get clampedDesired() {
    return Math.min(this.desiredOutputSec, this.maxOutputSec);
  }

  ngOnDestroy() {
    if (this.slotA.url) URL.revokeObjectURL(this.slotA.url);
    if (this.slotB.url) URL.revokeObjectURL(this.slotB.url);
    if (this.outputUrl) URL.revokeObjectURL(this.outputUrl);
  }
}
