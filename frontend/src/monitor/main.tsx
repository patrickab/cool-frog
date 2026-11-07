import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import Monitor from './Monitor';
import { ThemeProvider } from '../lib/ui';
import '../style.css';

createRoot(document.getElementById('root')!).render(<StrictMode><ThemeProvider><Monitor /></ThemeProvider></StrictMode>)
