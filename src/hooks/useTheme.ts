import { useState, useEffect, useCallback } from 'react';

export type Theme = 'dark' | 'light';

const STORAGE_KEY = 'theme';

/**
 * Theme state, stamped onto <html data-theme> so index.css's token blocks can
 * do the actual work.
 *
 * Resolution order is: an explicit choice the user has made before, then the
 * OS preference, then dark — dark being this app's original look, so anyone
 * who has never touched the toggle and has no OS preference sees no change.
 *
 * The OS preference is only a *default*. Once someone picks a theme it is
 * written to localStorage and stops tracking the system, because a user who
 * deliberately chose light does not want it flipping at sunset.
 */
function readStoredTheme(): Theme | null {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === 'dark' || v === 'light' ? v : null;
  } catch {
    return null;  // private mode / storage disabled
  }
}

function systemTheme(): Theme {
  try {
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  } catch {
    return 'dark';
  }
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(() => readStoredTheme() ?? systemTheme());

  // Apply to the document element rather than a React-rendered wrapper: the
  // tokens have to reach <body> and any portalled content too, and body's
  // background is set in CSS.
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // Follow the OS only while the user has made no explicit choice.
  useEffect(() => {
    if (readStoredTheme()) return;
    let mq: MediaQueryList;
    try {
      mq = window.matchMedia('(prefers-color-scheme: light)');
    } catch {
      return;
    }
    const onChange = (e: MediaQueryListEvent) => {
      if (!readStoredTheme()) setThemeState(e.matches ? 'light' : 'dark');
    };
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // still applies for this session
    }
  }, []);

  const toggleTheme = useCallback(
    () => setTheme(theme === 'dark' ? 'light' : 'dark'),
    [theme, setTheme]);

  return { theme, setTheme, toggleTheme };
}
