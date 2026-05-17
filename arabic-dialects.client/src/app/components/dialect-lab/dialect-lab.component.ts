import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SpectrogramComponent } from '../spectrogram/spectrogram.component';
import { ClassifierComponent } from '../classifier/classifier.component';
import { TranscriptionComponent } from '../transcription/transcription.component';
import { TtsTranslatorComponent } from '../tts-translator/tts-translator.component';
import { SignalBlenderComponent } from '../signal-blender/signal-blender.component';

type Tab = 'spectrogram' | 'classifier' | 'transcription' | 'tts' | 'blender';

interface TabDef {
  id: Tab;
  label: string;
  icon: string;
  desc: string;
}

@Component({
  selector: 'app-dialect-lab',
  standalone: true,
  imports: [
    CommonModule,
    SpectrogramComponent,
    ClassifierComponent,
    TranscriptionComponent,
    TtsTranslatorComponent,
    SignalBlenderComponent,
  ],
  templateUrl: './dialect-lab.component.html',
  styleUrls: ['./dialect-lab.component.css']
})
export class DialectLabComponent {
  active: Tab = 'spectrogram';

  tabs: TabDef[] = [
    { id: 'spectrogram', label: 'Spectrogram', desc: 'Visualize audio', icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>` },
    { id: 'classifier',  label: 'Classifier',  desc: 'Detect dialect',  icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>` },
    { id: 'transcription', label: 'Transcription', desc: 'Real-time text', icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8"/></svg>` },
    { id: 'tts',         label: 'Translate & TTS', desc: 'Change dialect', icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>` },
    { id: 'blender',     label: 'Signal Blender', desc: 'Mix two files', icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 3H2v6M16 3h6v6M2 21h6M16 21h6v-6M5 3v18M19 3v18"/></svg>` },
  ];

  setTab(id: Tab) { this.active = id; }
}
