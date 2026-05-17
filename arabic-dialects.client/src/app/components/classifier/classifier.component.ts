import { Component, ChangeDetectorRef  } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DialectApiService, DialectVizData, PredictResult, VisualizeResult } from '../../services/dialect-api.service';
import { AudioUtilService } from '../../services/audio-util.service';
import { FormsModule } from '@angular/forms';
import * as Plotly from 'plotly.js-dist-min';  // or use CDN version below
import { AbsPipe } from '../abs.pipe';

interface DialectBar {
  label: string;
  arabic: string;
  value: number;
  color: string;
}

@Component({
  selector: 'app-classifier',
  standalone: true,
  imports: [CommonModule, FormsModule, AbsPipe],
  templateUrl: './classifier.component.html',
  styleUrls: ['./classifier.component.css']
})
export class ClassifierComponent {
  fileName = '';
  isLoading = false;
  hasResult = false;
  predicted = '';
  confidence = 0;
  audioUrl = '';

  readonly dialectMeta: Record<string, { arabic: string; color: string; features: string[] }> = {
    'Egyptian':   { arabic: 'المصرية', color: '#1a73e8', features: ['Qaf→Hamza shift','ʕ vowel raising','Rising intonation','High F2 formant'] },
    'Levantine':  { arabic: 'السورية', color: '#34a853', features: ['Qaf→G shift','Imala vowel shift','Cluster reduction','Mid-pitch contour'] },
    'Maghrebi':   { arabic: 'المغربية', color: '#e8a82b', features: ['Vowel deletion','Berber substrate','Consonant clusters','Labial pharyngeal'] },
    'Gulf':       { arabic: 'السعودية', color: '#9c27b0', features: ['Qaf preservation','Guttural emphasis','Long vowels','Low F1 resonance'] },
  };

  bars: DialectBar[] = [];
  topFeatures: string[] = [];
  ldaTopFeatures: Record<string, { feature_name: string; weight: number }[]> = {};
  currentFile: File | null = null;

constructor(private api: DialectApiService, private audioUtil: AudioUtilService, private cdr: ChangeDetectorRef) {}

  async onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    this.currentFile = file;
    this.fileName = file.name;
    this.audioUrl = URL.createObjectURL(file);
    this.hasResult = false;
    await this.runPrediction(file);
  }

visualData: Record<string, DialectVizData> | null = null;

renderPlots() {
  if (!this.visualData) return;

  Object.entries(this.visualData).forEach(([dialect, data], i) => {
    const traces: any[] = [
      {
        type: 'scatter3d', mode: 'markers',
        name: dialect,
        x: data.train_target_points.x,
        y: data.train_target_points.y,
        z: data.train_target_points.z,
        marker: { color: data.color, size: 4, opacity: 0.8 }
      },
      {
        type: 'scatter3d', mode: 'markers',
        name: 'Other',
        x: data.train_other_points.x,
        y: data.train_other_points.y,
        z: data.train_other_points.z,
        marker: { color: '#aaaaaa', size: 3, opacity: 0.3 }
      },
      {
        type: 'surface',
        x: data.hyperplane.x,
        y: data.hyperplane.y,
        z: data.hyperplane.z,
        opacity: 0.2,
        colorscale: [[0, data.color], [1, data.color]],
        showscale: false,
        name: 'Boundary'
      },
      {
        type: 'scatter3d', mode: 'markers',
        name: 'Your sample',
        x: data.test_sample.x,
        y: data.test_sample.y,
        z: data.test_sample.z,
        marker: { color: '#ffffff', size: 8, symbol: 'diamond',
                  line: { color: data.color, width: 3 } }
      }
    ];

    Plotly.newPlot(`plot-${i}`, traces, {
      title: { text: `${dialect} vs All`, font: { color: '#fff' } },
      paper_bgcolor: 'transparent',
      plot_bgcolor:  'transparent',
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

async runPrediction(file: File) {
  this.isLoading = true;
  this.hasResult = false;
  try {
    const result = await this.api.predict(file).toPromise() as PredictResult;
    this.predicted = result.predicted_dialect;
    this.confidence = Math.round(result.confidence * 100);
    this.buildBars(result.predicted_dialect, result.confidence);
    this.topFeatures = this.dialectMeta[result.predicted_dialect]?.features ?? [];

    // Call visualize if features came back
    if (result.features) {
      const vizResult = await this.api.visualize(result.features).toPromise() ?? null;
      this.visualData = vizResult?.dialects ?? null;
      this.ldaTopFeatures = vizResult?.lda_top_features ?? {};
    }

    this.hasResult = true;
    if (this.visualData) {
      this.cdr.detectChanges(); // force *ngIf to render the plot divs NOW
      this.renderPlots();
    }
  } catch (e) {
    console.error(e);
  } finally {
    this.isLoading = false;
  }
}
  buildBars(winner: string, winConf: number) {
    const dialects = ['Egyptian', 'Levantine', 'Maghrebi', 'Gulf'];
    const remaining = 1 - winConf;
    const others = dialects.filter(d => d !== winner);
    // distribute remaining among others pseudo-randomly but deterministic
    const weights = [0.5, 0.3, 0.2];
    this.bars = dialects.map((d, i) => {
      let val: number;
      if (d === winner) val = winConf;
      else {
        const oi = others.indexOf(d);
        val = remaining * weights[oi];
      }
      return {
        label: d,
        arabic: this.dialectMeta[d].arabic,
        value: Math.round(val * 100),
        color: this.dialectMeta[d].color
      };
    }).sort((a, b) => b.value - a.value);
  }

  get predictedColor(): string {
    return this.dialectMeta[this.predicted]?.color ?? '#1a73e8';
  }

  get predictedArabic(): string {
    return this.dialectMeta[this.predicted]?.arabic ?? '';
  }
}
