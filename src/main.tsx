import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, AlertTriangle, ArrowRight, Check, ChevronDown, Circle, Clock3, CloudUpload, Crosshair, Map, Pause, Play, RotateCcw, ShieldCheck, Square, Target, X } from './icons';
import './style.css';

type Level = 'Easy' | 'Medium' | 'Hard';
type Detection = { label: string; category: string; confidence: number; box: number[] };
type Prediction = { action: 'GO' | 'STOP'; reasons: string[]; detections: Detection[]; model: string };
type Scene = { id: string; title: string; location: string; expected: 'GO' | 'STOP'; required: string[]; note: string };
type Result = { scene: Scene; prediction: Prediction; correct: boolean };
const scenes: Record<string, Scene> = {
  'clear-green': { id: 'clear-green', title: 'Open boulevard', location: 'Market Street', expected: 'GO', required: ['green_light'], note: 'Green signal. Clear lane.' },
  'red-pedestrian': { id: 'red-pedestrian', title: 'Occupied crossing', location: 'Civic Center', expected: 'STOP', required: ['red_light', 'pedestrian', 'crosswalk'], note: 'Red signal and pedestrian in path.' },
  'dog-crossing': { id: 'dog-crossing', title: 'Animal crossing', location: 'Residential district', expected: 'STOP', required: ['dog'], note: 'Dog entering the lane.' },
  'rainy-decoy': { id: 'rainy-decoy', title: 'Reflected signals', location: 'Financial district', expected: 'STOP', required: ['red_light'], note: 'Red light with a misleading green advertisement.' },
  'cat-cone': { id: 'cat-cone', title: 'Unexpected obstacle', location: 'North avenue', expected: 'STOP', required: ['cat', 'obstacle'], note: 'Cat and cone in the lane.' },
};
const routes: Record<Level, string[]> = {
  Easy: ['clear-green', 'red-pedestrian', 'dog-crossing'],
  Medium: ['clear-green', 'red-pedestrian', 'dog-crossing', 'cat-cone'],
  Hard: ['clear-green', 'red-pedestrian', 'dog-crossing', 'rainy-decoy', 'cat-cone'],
};
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
const pct = (n: number) => `${Math.round(n * 100)}%`;
const scenePoints = (result: Result) => {
  const actionPoints = result.correct ? 60 : 0;
  const found = result.scene.required.filter(category => result.prediction.detections.some(detection => detection.category === category)).length;
  return actionPoints + Math.round(40 * found / result.scene.required.length);
};

function App() {
  const [level, setLevel] = useState<Level>('Medium');
  const [model, setModel] = useState<string | null>(null);
  const [backendError, setBackendError] = useState('');
  const [uploading, setUploading] = useState(false);
  const [phase, setPhase] = useState<'idle' | 'running' | 'finished'>('idle');
  const [paused, setPaused] = useState(false);
  const [index, setIndex] = useState(0);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [results, setResults] = useState<Result[]>([]);
  const [showBoxes, setShowBoxes] = useState(true);
  const [seconds, setSeconds] = useState(0);
  const runToken = useRef(0);
  const pausedRef = useRef(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const route = routes[level];
  const scene = scenes[route[Math.min(index, route.length - 1)]];
  const score = results.reduce((sum, result) => sum + scenePoints(result), 0);
  const scorePercent = results.length ? Math.round(score / results.length) : 0;
  const correctCount = results.filter(result => result.correct).length;
  const unsafe = results.filter(result => !result.correct && result.scene.expected === 'STOP').length;

  useEffect(() => {
    fetch('/api/status').then(async response => { if (!response.ok) throw new Error('Inference service unavailable'); return response.json(); })
      .then(data => { setModel(data.model); setBackendError(''); })
      .catch(() => setBackendError('Inference service offline. Start the Python server to run a model.'));
  }, []);
  useEffect(() => { if (phase !== 'running' || paused) return; const timer = window.setInterval(() => setSeconds(value => value + 1), 1000); return () => clearInterval(timer); }, [phase, paused]);
  useEffect(() => { pausedRef.current = paused; }, [paused]);
  useEffect(() => { if (phase !== 'running') return; const token = runToken.current; let cancelled = false;
    const runScene = async () => {
      setPrediction(null);
      try {
        await delay(700);
        if (cancelled || token !== runToken.current) return;
        const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scene_id: scene.id }) });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Inference failed');
        if (cancelled || token !== runToken.current) return;
        const result = data as Prediction;
        setPrediction(result);
        setResults(previous => [...previous, { scene, prediction: result, correct: result.action === scene.expected }]);
        await delay(2800);
        while (pausedRef.current && !cancelled && token === runToken.current) await delay(150);
        if (cancelled || token !== runToken.current) return;
        if (index + 1 >= route.length) { setPhase('finished'); } else { setIndex(index + 1); }
      } catch (error) { if (!cancelled) { setBackendError(error instanceof Error ? error.message : 'Inference failed'); setPhase('idle'); } }
    };
    void runScene();
    return () => { cancelled = true; };
  }, [phase, index, level]);

  const reset = () => { runToken.current++; pausedRef.current = false; setPhase('idle'); setPaused(false); setIndex(0); setPrediction(null); setResults([]); setSeconds(0); };
  const start = () => { if (!model) { setBackendError('Add a YOLOv8 .pt file in models/ or upload one before starting.'); return; } reset(); setBackendError(''); setPhase('running'); };
  const upload = async (file: File) => {
    setUploading(true); setBackendError('');
    try { const response = await fetch(`/api/model?name=${encodeURIComponent(file.name)}`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file }); const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Upload failed'); setModel(data.model); reset(); }
    catch (error) { setBackendError(error instanceof Error ? error.message : 'Upload failed'); }
    finally { setUploading(false); }
  };
  const levelChange = (next: Level) => { reset(); setLevel(next); };
  const decisionText = prediction?.reasons.join(' · ') || (phase === 'running' ? 'Analyzing camera frame…' : 'Awaiting model prediction');

  return <div className="app">
    <header className="topbar"><div className="brand">AUTONOMOUS <span>/ LAB</span></div><div className="brand-sub">VISION DRIVE SIMULATOR</div><div className={`top-status ${phase}`}> <span className="status-dot" /> {phase === 'running' ? (paused ? 'PAUSED' : 'RUNNING') : phase === 'finished' ? 'RUN COMPLETE' : 'READY TO DRIVE'}</div><div className="top-model"><Crosshair size={14}/><span>{model || 'No model loaded'}</span></div><div className="top-spacer"/><div className="clock"><Clock3 size={15}/> {String(Math.floor(seconds / 60)).padStart(2, '0')}:{String(seconds % 60).padStart(2, '0')}</div><div className="header-rule"/><button className="icon-button" title="Toggle detection boxes" onClick={() => setShowBoxes(value => !value)}><Target size={18}/></button></header>
    <div className="workspace">
      <aside className="route-rail"><div className="rail-label">ROUTE</div><div className="route-distance">{(route.length * 0.8).toFixed(1)} <small>km</small></div><div className="route-line">{route.map((id, i) => <div key={id} className={`waypoint ${i < index || phase === 'finished' ? 'done' : i === index ? 'active' : ''}`}><span className="waypoint-dot">{i < index || phase === 'finished' ? <Check size={11}/> : null}</span><span className="waypoint-text"><strong>0{i + 1}</strong><small>{scenes[id].title}</small></span></div>)}</div><div className="rail-map"><div className="map-lines"/><div className="map-route"><span/></div></div><div className="rail-footer">{Math.max(0, (route.length - index) * .8).toFixed(1)} km <small>to destination</small></div></aside>
      <main className="camera-area"><div className={`camera-stage ${phase === 'running' && prediction?.action === 'GO' && !paused ? 'is-moving' : ''}`} key={scene.id}><img src={`/scenes/${scene.id}.jpg`} alt={`First person road view: ${scene.title}`}/><div className="camera-vignette"/>{showBoxes && prediction?.detections.filter(d => d.category !== 'other').map((d, i) => <div key={`${d.label}-${i}`} className={`bbox ${d.category.includes('light') ? 'signal-box' : ''}`} style={{ left: `${d.box[0] * 100}%`, top: `${d.box[1] * 100}%`, width: `${(d.box[2] - d.box[0]) * 100}%`, height: `${(d.box[3] - d.box[1]) * 100}%` }}><span>{d.category.replace('_', ' ')} {pct(d.confidence)}</span></div>)}<div className="camera-top"><div><span className="camera-rec"/> CAMERA 01 <span className="camera-slash">/</span> FRONT WINDSHIELD</div><div>1920 × 1080 <span className="camera-slash">/</span> LIVE FEED</div></div><div className="camera-location"><Map size={14}/> {scene.location.toUpperCase()} <span>·</span> {scene.title.toUpperCase()}</div><div className="camera-center"><span/><span/></div><div className="camera-bottom"><div className="speed-block"><span className="speed-value">{phase === 'running' && !paused && prediction?.action === 'GO' ? '32' : '00'}</span><span className="speed-unit">KM/H</span></div><div className="camera-progress"><div className="progress-kicker">TRIP PROGRESS <strong>{Math.round(((phase === 'finished' ? route.length : index) / route.length) * 100)}%</strong></div><div className="progress-track"><span style={{width: `${((phase === 'finished' ? route.length : index) / route.length) * 100}%`}}/></div></div><div className={`camera-action ${prediction?.action.toLowerCase() || ''}`}><span className="action-tiny">MODEL COMMAND</span><strong>{prediction?.action || (phase === 'running' ? 'SCAN' : 'STANDBY')}</strong></div></div></div><div className="camera-caption"><div className="caption-left"><Activity size={16}/> PERCEPTION ENGINE <span className="caption-divider"/> {prediction ? `${prediction.detections.length} OBJECTS IDENTIFIED` : 'AWAITING FRAME'}</div><div className="caption-right">SCENE {String(index + 1).padStart(2, '0')} <span>/</span> {String(route.length).padStart(2, '0')}</div></div></main>
      <aside className="inspector"><div className="panel-heading"><span>MISSION CONTROL</span><Circle size={9} fill="currentColor"/></div><div className="panel-section difficulty"><div className="section-label">SCENARIO DIFFICULTY <ChevronDown size={13}/></div><div className="segments">{(['Easy','Medium','Hard'] as Level[]).map(item => <button key={item} className={level === item ? 'selected' : ''} onClick={() => levelChange(item)} disabled={phase === 'running'}>{item}</button>)}</div><p>{level === 'Hard' ? 'Rain, reflections and visual decoys.' : level === 'Medium' ? 'Animals and lane obstacles added.' : 'Clear light and crossing scenarios.'}</p></div><div className="panel-section model-section"><div className="section-label">VISION MODEL</div><input ref={fileInput} type="file" accept=".pt" hidden onChange={event => { const file = event.target.files?.[0]; if (file) void upload(file); event.target.value = ''; }}/><button className="upload-button" onClick={() => fileInput.current?.click()} disabled={uploading || phase === 'running'}><CloudUpload size={18}/>{uploading ? 'Loading model…' : 'Upload YOLOv8 model'}<ArrowRight size={15}/></button><div className="model-file"><span className={model ? 'model-ready' : 'model-missing'}>{model ? <Check size={11}/> : <X size={11}/>}</span><span>{model || 'No local weights found'}</span></div>{backendError && <div className="inline-error"><AlertTriangle size={13}/>{backendError}</div>}</div><div className="panel-section detections"><div className="section-label">LIVE DETECTIONS <span>{prediction?.detections.length || '00'}</span></div><div className="detection-list">{prediction?.detections.length ? prediction.detections.slice(0, 7).map((d, i) => <div className="detection-row" key={`${d.label}-${i}`}><div className={`detection-icon ${d.category}`}>{d.category.includes('light') ? '◉' : d.category === 'pedestrian' ? '♙' : d.category === 'dog' || d.category === 'cat' ? '◆' : '▣'}</div><span className="detection-name">{d.category === 'other' ? d.label : d.category.replace('_', ' ')}</span><span className="detection-bar"><i style={{width: pct(d.confidence)}}/></span><span className="detection-confidence">{pct(d.confidence)}</span></div>) : <div className="empty-detections"><Crosshair size={23}/><span>Detections appear as the camera frame is analyzed.</span></div>}</div></div><div className="panel-section decision"><div className="section-label">DRIVE DECISION <span>MODEL OUTPUT</span></div><div className={`decision-tile ${prediction?.action.toLowerCase() || ''}`}><div className="decision-symbol">{prediction?.action === 'STOP' ? <Square fill="currentColor" size={15}/> : prediction?.action === 'GO' ? <ArrowRight size={22}/> : <Crosshair size={22}/>}</div><strong>{prediction?.action || (phase === 'running' ? 'ANALYZING' : 'STANDBY')}</strong></div><p>{decisionText}</p></div><div className="panel-section metrics"><div className="section-label">RUN METRICS</div><div className="metric-grid"><div><small>Scenes assessed</small><strong>{results.length} <em>/ {route.length}</em></strong></div><div><small>Correct decisions</small><strong>{correctCount}</strong></div><div><small>Unsafe passes</small><strong className={unsafe ? 'danger-text' : ''}>{unsafe}</strong></div><div><small>Trip score</small><strong className="score-text">{scorePercent}<em> / 100</em></strong></div></div></div></aside>
    </div>
    <footer className="bottom-bar"><div className="scene-selector"><span>SCENARIO {String(index + 1).padStart(2, '0')}</span><strong>{scene.title}</strong><ChevronDown size={15}/></div><div className="transport"><button className="round-button" title="Reset run" onClick={reset}><RotateCcw size={17}/></button><button className="main-play" title={phase === 'running' ? paused ? 'Resume' : 'Pause' : 'Start run'} onClick={() => { if (phase === 'running') setPaused(v => !v); else start(); }}>{phase === 'running' && !paused ? <Pause size={19} fill="currentColor"/> : <Play size={19} fill="currentColor"/>}</button><span>{phase === 'running' ? paused ? 'RESUME' : 'PAUSE' : 'START RUN'}</span></div><div className="bottom-track"><div><span style={{width: `${(results.length / route.length) * 100}%`}}/></div><small>{results.length} / {route.length} SCENES</small></div><div className="bottom-score"><small>SESSION SCORE</small><strong>{scorePercent}<span> / 100</span></strong></div><button className="end-button" onClick={reset}>{phase === 'running' ? 'END RUN' : 'RESET'} <span>ESC</span></button></footer>
    {phase === 'finished' && <div className="result-overlay"><div className="result-modal"><div className="result-head"><ShieldCheck size={21}/> RUN COMPLETE <button onClick={reset} aria-label="Close report"><X size={19}/></button></div><div className="result-score">{scorePercent}<span>/ 100</span></div><h2>{unsafe ? 'Review the missed hazards.' : 'Route completed safely.'}</h2><p>{correctCount} of {route.length} driving decisions were safe. Score combines decision accuracy (60%) and required hazard recognition (40%). {unsafe} unsafe pass{unsafe === 1 ? '' : 'es'}.</p><div className="result-list">{results.map((r, i) => <div key={i}><span className={r.correct ? 'good-mark' : 'bad-mark'}>{r.correct ? <Check size={15}/> : <X size={15}/>}</span><span>{r.scene.title}</span><small>{scenePoints(r)} PTS · EXPECTED {r.scene.expected}</small><strong>{r.prediction.action}</strong></div>)}</div><div className="result-actions"><button onClick={reset}>CLOSE REPORT</button><button onClick={start}>RUN AGAIN <ArrowRight size={16}/></button></div></div></div>}
  </div>;
}
createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
