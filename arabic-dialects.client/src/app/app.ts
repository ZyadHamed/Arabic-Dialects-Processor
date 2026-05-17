import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { DialectLabComponent } from './components/dialect-lab/dialect-lab.component';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, DialectLabComponent],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  protected readonly title = signal('arabic-dialects.client');
}
