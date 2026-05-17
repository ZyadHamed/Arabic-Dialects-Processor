import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DialectApiService } from '../../services/dialect-api.service';
import { firstValueFrom } from 'rxjs'; // <-- 1. Add this import
@Component({
  selector: 'app-tts-translator',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './tts-translator.component.html',
  styleUrls: ['./tts-translator.component.css']
})
export class TtsTranslatorComponent {
  readonly dialects = ['Egyptian', 'Levantine', 'Maghrebi', 'Gulf'];
  readonly dialectArabic: Record<string, string> = {
    Egyptian: 'المصرية', Levantine: 'السورية',
    Maghrebi: 'المغربية', Gulf: 'السعودية'
  };
  readonly speakers = [1, 2, 3, 4];

  // Step 1 – transcription source
  fileName = '';
  audioUrl = '';
  transcribedText = '';
  isTranscribing = false;
  isSourceReady = false;

  // Step 2 – translation settings
  originalDialect = 'Egyptian';
  targetDialect = 'Levantine';
  speakerId = 1;

  // Step 3 – results
  translatedText = '';
  ttsAudioUrl = '';
  isTranslating = false;
  isTtsLoading = false;
  step: 'idle' | 'transcribed' | 'translated' | 'tts' = 'idle';

  readonly ttsDialectMap: Record<string, string> = {
  Egyptian:  'Egyptian',
  Levantine: 'Levantine',
  Maghrebi:  'Maghrebi',
  Gulf:      'Khaliji',   // ← the only one that differs
};

constructor(
    private api: DialectApiService,
    private cdr: ChangeDetectorRef // <-- 2. Inject it here
  ) {}

  async onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    
    this.clearResults();
    this.fileName = file.name;
    this.audioUrl = URL.createObjectURL(file);
    this.isTranscribing = true;
    this.step = 'idle';
    
    // Optional: Force UI to show the loading state instantly
    this.cdr.detectChanges(); 

    try {
      // 3. Swap to firstValueFrom
      const res = await firstValueFrom(this.api.transcribe(file)) as any;
      this.transcribedText = res.transcription ?? res.text ?? '';
      this.isSourceReady = true;
      this.step = 'transcribed';
    } catch (e) {
      console.error('Transcription failed:', e); // <-- 4. Stop swallowing errors!
      this.transcribedText = '';
        this.isSourceReady = false;  // ← add this
    } finally {
      this.isTranscribing = false;
      this.cdr.detectChanges(); // <-- 5. Force UI to update after await finishes
    }
  }

  async translate() {
    console.log(this.transcribedText)
    if (!this.transcribedText || this.isTranslating) return;
    this.isTranslating = true;
    this.translatedText = '';
    this.ttsAudioUrl = '';
    this.cdr.detectChanges(); // Force loading UI

    try {
      const res = await firstValueFrom(this.api.translate(
        this.transcribedText,
        this.originalDialect,
        this.targetDialect
      )) as { translated_text: string };
      
      this.translatedText = res.translated_text;
      this.step = 'translated';
    } catch (e) {
      console.error('Translation failed:', e);
      this.translatedText = '(Translation error)';
    } finally {
      this.isTranslating = false;
      this.cdr.detectChanges(); // <-- Update view with translation
    }
  }

async synthesize() {
  if (!this.translatedText) return;
  this.isTtsLoading = true;
  this.ttsAudioUrl = '';
  this.cdr.detectChanges();

  try {
    const ttsDialect = this.ttsDialectMap[this.targetDialect]; // 'Khaliji' instead of 'Gulf'
    const blob = await firstValueFrom(this.api.tts(
      this.translatedText,
      ttsDialect,   // ← mapped value for backend
      this.speakerId
    )) as Blob;

    this.ttsAudioUrl = URL.createObjectURL(blob);
    this.step = 'tts';
  } catch (e) {
    console.error('TTS failed:', e);
  } finally {
    this.isTtsLoading = false;
    this.cdr.detectChanges();
  }
}

  clearResults() {
    this.transcribedText = '';
    this.translatedText = '';
    this.ttsAudioUrl = '';
    this.isSourceReady = false;
    this.step = 'idle';
    // If this is called from a template event, Angular updates automatically. 
    // If called asynchronously somewhere else, add this.cdr.detectChanges() here too.
  }

  get availableTargets() {
    return this.dialects.filter(d => d !== this.originalDialect);
  }
}
