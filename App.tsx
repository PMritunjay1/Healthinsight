import React, { useState, useCallback } from 'react';
import {
  PipelineAnalysisResult,
  SpecialistCardData,
  DemoCase
} from './types';
import {
  runMedicalAnalysisPipeline,
  extractTextFromDocument,
  DEMO_CLINICAL_CASES,
  API_BASE_URL
} from './services/analysisService';

// ──────────────────────────────────────────────
// INLINE SVG ICONS
// ──────────────────────────────────────────────

const HeartIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
  </svg>
);

const LungsIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
  </svg>
);

const BrainIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
);

const StethoscopeIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
  </svg>
);

const PillIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
  </svg>
);

const ChevronDownIcon = ({ className = "w-4 h-4" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
  </svg>
);

const ChevronUpIcon = ({ className = "w-4 h-4" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
  </svg>
);

const CheckCircleIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const ExclamationIcon = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
  </svg>
);

export const App: React.FC = () => {
  // Clinical Input State
  const [patientText, setPatientText] = useState<string>('');
  const [patientHistory, setPatientHistory] = useState<string>('');
  const [showHistory, setShowHistory] = useState<boolean>(false);
  const [selectedDemoId, setSelectedDemoId] = useState<string | null>(null);

  // Execution & Response State
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<PipelineAnalysisResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [showTelemetry, setShowTelemetry] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);

  // Load a pre-configured clinical demo case
  
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const file = e.target.files[0];
    setIsUploading(true);
    setErrorMessage(null);
    try {
      const extractedText = await extractTextFromDocument(file);
      // Append or replace? The instructions say 'Show the extracted text before clinical processing where practical so that the user can inspect what the OCR system actually read.'
      setPatientText((prev) => prev ? prev + "\n\n" + extractedText : extractedText);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to upload and extract text from document.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleSelectDemo = (demo: DemoCase) => {
    setSelectedDemoId(demo.id);
    setPatientText(demo.symptoms);
    setPatientHistory(demo.history);
    setErrorMessage(null);
    if (demo.history) setShowHistory(true);
  };

  // Reset form
  const handleClear = () => {
    setPatientText('');
    setPatientHistory('');
    setSelectedDemoId(null);
    setAnalysisResult(null);
    setErrorMessage(null);
  };

  // Execute clinical analysis through local multi-agent orchestrator
  const handleAnalyze = useCallback(async () => {
    if (!patientText.trim()) {
      setErrorMessage("Please enter patient clinical presentation or select a demo case above.");
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);
    setAnalysisResult(null); // Prevent any stale state from previous case

    try {
      const result = await runMedicalAnalysisPipeline(patientText, patientHistory);
      setAnalysisResult(result);
    } catch (err: any) {
      console.error("Clinical analysis failed:", err);
      setErrorMessage(err.message || "Failed to communicate with the local diagnostic server.");
    } finally {
      setIsLoading(false);
    }
  }, [patientText, patientHistory]);

  const getSpecialistIcon = (id: string) => {
    if (id === 'cardiologist') return <HeartIcon className="w-6 h-6 text-rose-600" />;
    if (id === 'pulmonologist') return <LungsIcon className="w-6 h-6 text-sky-600" />;
    if (id === 'neurologist') return <BrainIcon className="w-6 h-6 text-violet-600" />;
    return <StethoscopeIcon className="w-6 h-6 text-emerald-600" />;
  };

  const getSpecialistBadgeColor = (id: string) => {
    if (id === 'cardiologist') return 'bg-rose-50 text-rose-700 border-rose-200';
    if (id === 'pulmonologist') return 'bg-sky-50 text-sky-700 border-sky-200';
    if (id === 'neurologist') return 'bg-violet-50 text-violet-700 border-violet-200';
    return 'bg-emerald-50 text-emerald-700 border-emerald-200';
  };

  const getRiskColor = (level: string) => {
    const l = (level || '').toLowerCase();
    if (l.includes('high') || l.includes('resuscitation') || l.includes('emergent') || l.includes('1') || l.includes('2')) {
      return 'bg-red-100 text-red-800 border-red-300';
    }
    if (l.includes('moderate') || l.includes('urgent') || l.includes('3')) {
      return 'bg-amber-100 text-amber-800 border-amber-300';
    }
    return 'bg-emerald-100 text-emerald-800 border-emerald-300';
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans flex flex-col">
      {/* ──────────────────────────────────────────────
          TOP APP HEADER
      ────────────────────────────────────────────── */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-600 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-sky-500/20">
              <StethoscopeIcon className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-lg font-bold text-slate-900 tracking-tight">Health Insights CDS</h1>
                <span className="px-2 py-0.5 text-xs font-semibold bg-sky-100 text-sky-800 rounded-full border border-sky-200">
                  v3.0 Multi-Specialist
                </span>
              </div>
              <p className="text-xs text-slate-500">Autonomous Multi-Agent Clinical Decision Support &amp; RAC Consensus</p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <div className="hidden md:flex items-center space-x-1.5 px-3 py-1 bg-emerald-50 border border-emerald-200 rounded-lg text-emerald-700 text-xs font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>100% Offline Local CPU Execution</span>
            </div>
            <a
              href={`${API_BASE_URL}/docs`}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-slate-600 hover:text-sky-600 px-2.5 py-1.5 rounded-lg border border-slate-200 hover:border-sky-300 transition-colors font-medium"
            >
              API Docs
            </a>
            <button
              onClick={handleClear}
              className="text-xs text-slate-700 bg-slate-100 hover:bg-slate-200 px-3 py-1.5 rounded-lg transition-colors font-medium"
            >
              New Case
            </button>
          </div>
        </div>
      </header>

      {/* ──────────────────────────────────────────────
          MAIN CONTAINER
      ────────────────────────────────────────────── */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        
        {/* 1. AUDITED CLINICAL SCENARIOS (1-CLICK TEST PRESETS) */}
        <section className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Audited Clinical Scenarios (1-Click Test Cases)
            </h2>
            <span className="text-xs text-slate-400">Click to load pre-verified clinical presentations</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
            {DEMO_CLINICAL_CASES.map((demo) => {
              const isSelected = selectedDemoId === demo.id;
              return (
                <button
                  key={demo.id}
                  onClick={() => handleSelectDemo(demo)}
                  className={`text-left p-3 rounded-xl border transition-all ${
                    isSelected
                      ? 'bg-sky-50 border-sky-400 ring-2 ring-sky-200 shadow-xs'
                      : 'bg-slate-50 border-slate-200 hover:border-slate-300 hover:bg-slate-100/70'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-bold text-slate-800 line-clamp-1">{demo.label}</span>
                    <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-white text-slate-600 border border-slate-200 ml-1 shrink-0">
                      {demo.category}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-500 line-clamp-2">{demo.description}</p>
                </button>
              );
            })}
          </div>
        </section>

        {/* 2. CLINICAL INPUT PANEL */}
        <section className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-2.5 h-2.5 rounded-full bg-sky-500"></div>
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-700">Patient Presentation</h2>
            </div>
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="text-xs text-sky-600 hover:text-sky-700 font-medium flex items-center space-x-1"
            >
              <span>{showHistory ? 'Hide Medical History' : '+ Add Past Medical History'}</span>
              {showHistory ? <ChevronUpIcon className="w-3.5 h-3.5" /> : <ChevronDownIcon className="w-3.5 h-3.5" />}
            </button>
          </div>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="block text-xs font-semibold text-slate-600">
                Clinical Narrative / Chief Complaint / Vitals &amp; Symptoms:
              </label>
              <div className="relative">
                <input type="file" id="ocr-upload" accept="image/*,application/pdf" className="hidden" onChange={handleFileUpload} disabled={isUploading} />
                <label htmlFor="ocr-upload" className={`cursor-pointer text-xs font-medium px-2 py-1 rounded bg-slate-100 border ${isUploading ? 'text-slate-400 border-slate-200' : 'text-sky-700 border-sky-200 hover:bg-sky-50 transition-colors'}`}>
                  {isUploading ? 'Extracting OCR...' : 'Upload Document (PDF/Img)'}
                </label>
              </div>
            </div>
            <textarea
              rows={4}
              value={patientText}
              onChange={(e) => setPatientText(e.target.value)}
              placeholder="Enter patient narrative (e.g. 64-year-old male presenting with acute substernal chest pressure radiating to left arm, diaphoresis, dyspnea, troponin elevation...)"
              className="w-full px-3.5 py-2.5 rounded-xl border border-slate-300 focus:ring-2 focus:ring-sky-500 focus:border-sky-500 text-sm text-slate-800 placeholder-slate-400 font-sans transition-all resize-y"
            />
          </div>

          {showHistory && (
            <div className="pt-2">
              <label className="block text-xs font-semibold text-slate-600 mb-1.5">
                Past Medical History / Comorbidities:
              </label>
              <textarea
                rows={2}
                value={patientHistory}
                onChange={(e) => setPatientHistory(e.target.value)}
                placeholder="Optional comorbidities (e.g. Hypertension, Type 2 Diabetes, 30 pack-year smoking history, previous ischemic stroke...)"
                className="w-full px-3.5 py-2 rounded-xl border border-slate-300 focus:ring-2 focus:ring-sky-500 focus:border-sky-500 text-sm text-slate-800 placeholder-slate-400 font-sans transition-all resize-y"
              />
            </div>
          )}

          {errorMessage && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-xl flex items-start space-x-2 text-red-700 text-xs">
              <ExclamationIcon className="w-4 h-4 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Diagnostic Execution Error</p>
                <p>{errorMessage}</p>
              </div>
            </div>
          )}

          <div className="flex items-center justify-between pt-2">
            <button
              onClick={handleClear}
              className="px-4 py-2 text-xs font-semibold text-slate-600 hover:text-slate-800 rounded-xl border border-slate-200 hover:bg-slate-50 transition-colors"
            >
              Clear Form
            </button>

            <button
              onClick={handleAnalyze}
              disabled={isLoading || !patientText.trim()}
              className={`px-6 py-2.5 rounded-xl text-xs font-bold text-white shadow-md flex items-center space-x-2 transition-all ${
                isLoading || !patientText.trim()
                  ? 'bg-slate-300 cursor-not-allowed shadow-none'
                  : 'bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-700 hover:to-indigo-700 shadow-sky-500/25 active:scale-[0.99]'
              }`}
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>Executing Pipeline...</span>
                </>
              ) : (
                <>
                  <span>Run Diagnostic Analysis</span>
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                </>
              )}
            </button>
          </div>
        </section>

        {/* 3. PIPELINE EXECUTION STEPPER */}
        {analysisResult && (
          <section className="bg-slate-900 text-white rounded-2xl p-4 shadow-md">
            <div className="flex items-center justify-between mb-3 text-xs">
              <span className="font-mono text-slate-400 uppercase tracking-wider">Execution Pipeline Flow</span>
              <span className="font-mono text-sky-400">Total Latency: {Math.round(analysisResult.totalLatencyMs)}ms</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
              <div className="p-2.5 rounded-lg bg-slate-800/80 border border-slate-700">
                <div className="text-[10px] text-slate-400 font-mono">STEP 1</div>
                <div className="font-bold text-sky-300">RIE Engine</div>
                <div className="text-[11px] text-slate-400 mt-0.5">{analysisResult.symptoms.length} Entities Found</div>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-800/80 border border-slate-700">
                <div className="text-[10px] text-slate-400 font-mono">STEP 2</div>
                <div className="font-bold text-sky-300">Specialist Router</div>
                <div className="text-[11px] text-slate-400 mt-0.5">{analysisResult.routing.selectedSpecialists.join(' + ') || 'General'}</div>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-800/80 border border-slate-700">
                <div className="text-[10px] text-slate-400 font-mono">STEP 3</div>
                <div className="font-bold text-sky-300">Active Specialists</div>
                <div className="text-[11px] text-slate-400 mt-0.5">{analysisResult.activeSpecialists.length} Specialists Fired</div>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-800/80 border border-slate-700">
                <div className="text-[10px] text-slate-400 font-mono">STEP 4</div>
                <div className="font-bold text-sky-300">Risk &amp; RAC</div>
                <div className="text-[11px] text-slate-400 mt-0.5">{analysisResult.risk.riskLevel} • {Math.round(analysisResult.rac.agreementScore * 100)}% Consensus</div>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-800/80 border border-slate-700 col-span-2 md:col-span-1">
                <div className="text-[10px] text-slate-400 font-mono">STEP 5</div>
                <div className="font-bold text-emerald-400">Deterministic Fusion</div>
                <div className="text-[11px] text-slate-400 mt-0.5">Evidence Protocol</div>
              </div>
            </div>
          </section>
        )}

        {/* 4. ACTIVE SPECIALISTS SECTION */}
        {analysisResult && analysisResult.activeSpecialists.length > 0 && (
          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-700">
                Active Specialist Models ({analysisResult.activeSpecialists.length})
              </h2>
              <span className="text-xs text-slate-500">Only specialists activated by router are displayed</span>
            </div>

            <div className={`grid grid-cols-1 ${analysisResult.activeSpecialists.length > 1 ? 'md:grid-cols-2 lg:grid-cols-3' : 'md:grid-cols-1'} gap-4`}>
              {analysisResult.activeSpecialists.map((specialist: SpecialistCardData) => (
                <div
                  key={specialist.id}
                  className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs flex flex-col justify-between"
                >
                  <div>
                    {/* Header */}
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <div className="p-2 rounded-xl bg-slate-50 border border-slate-100">
                          {getSpecialistIcon(specialist.id)}
                        </div>
                        <div>
                          <h3 className="font-bold text-slate-900 text-sm">{specialist.name}</h3>
                          <span className={`inline-block text-[10px] font-semibold px-2 py-0.5 rounded-full border mt-0.5 ${getSpecialistBadgeColor(specialist.id)}`}>
                            {specialist.domain}
                          </span>
                        </div>
                      </div>
                      <span className="text-[10px] font-mono px-2 py-0.5 bg-slate-100 text-slate-600 rounded">
                        {specialist.predictedUrgency}
                      </span>
                    </div>

                    {/* Chief Complaint */}
                    <p className="text-xs text-slate-600 mb-3 bg-slate-50 p-2.5 rounded-xl border border-slate-100">
                      <strong className="text-slate-800">Domain Focus:</strong> {specialist.predictedComplaint}
                    </p>

                    {/* Predictions / Differential Bars */}
                    <div className="space-y-2 mb-4">
                      <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                        Top Differential Predictions:
                      </span>
                      {Object.entries(specialist.diseaseProbabilities).length > 0 ? (
                        Object.entries(specialist.diseaseProbabilities).slice(0, 4).map(([disease, prob], idx) => (
                          <div key={idx} className="space-y-1">
                            <div className="flex justify-between text-xs">
                              <span className="font-medium text-slate-800 line-clamp-1">{disease}</span>
                              <span className="font-mono text-slate-600 font-semibold">{Math.round(prob * 100)}%</span>
                            </div>
                            <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                              <div
                                className={`h-1.5 rounded-full ${
                                  idx === 0 ? 'bg-sky-600' : 'bg-slate-400'
                                }`}
                                style={{ width: `${Math.min(100, prob * 100)}%` }}
                              ></div>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="text-xs text-slate-600 p-2 bg-slate-50 rounded-lg">
                          <strong>Primary:</strong> {specialist.primaryDiagnosis} ({Math.round(specialist.confidence * 100)}%)
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Latency & Status */}
                  <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400 font-mono">
                    <span>Status: Complete</span>
                    <span>{Math.round(specialist.latencyMs)}ms</span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* 5. CONSENSUS DIAGNOSTIC REPORT & TREATMENT ENGINE */}
        {analysisResult && (
          <section className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-6">
            
            {/* Header Badge & Diagnosis */}
            <div className="flex flex-col md:flex-row md:items-center md:justify-between border-b border-slate-200 pb-5 gap-4">
              <div>
                <div className="flex items-center space-x-2 mb-1">
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-indigo-50 text-indigo-700 border border-indigo-200">
                    Consensus Diagnosis
                  </span>
                  <span className="text-xs font-mono text-slate-500">
                    Confidence: {Math.round(analysisResult.overallConfidence * 100)}%
                  </span>
                </div>
                <h2 className="text-2xl font-black text-slate-900 tracking-tight">
                  {analysisResult.finalDiagnosis}
                </h2>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                {/* Risk Badge */}
                <div className={`px-3.5 py-1.5 rounded-xl border text-xs font-bold flex items-center space-x-1.5 ${getRiskColor(analysisResult.risk.riskLevel)}`}>
                  <ExclamationIcon className="w-4 h-4" />
                  <span>Acuity: {analysisResult.risk.riskLevel}</span>
                </div>

                {/* RAC Consensus Badge */}
                <div className="px-3.5 py-1.5 rounded-xl border border-slate-200 bg-slate-50 text-slate-700 text-xs font-semibold flex items-center space-x-1.5">
                  <CheckCircleIcon className="w-4 h-4 text-emerald-600" />
                  <span>RAC Consensus: {Math.round(analysisResult.rac.agreementScore * 100)}%</span>
                </div>
              </div>
            </div>

            {/* Clinical Reasoning Narrative */}
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">
                Clinical Rationale &amp; Synthesis
              </h3>
              <p className="text-sm text-slate-700 leading-relaxed bg-slate-50 p-4 rounded-xl border border-slate-100 font-normal">
                {analysisResult.clinicalExplanation}
              </p>
            </div>

            {/* Treatment Protocol Engine */}
            <div>
              <div className="flex items-center space-x-2 mb-3">
                <PillIcon className="w-5 h-5 text-indigo-600" />
                <h3 className="text-sm font-bold text-slate-800">
                  Evidence-Based Clinical Protocol ({analysisResult.treatment.medicationCategory || 'Standard Care'})
                </h3>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                {/* Pharmacotherapy */}
                <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-2">
                  <span className="font-bold text-indigo-900 uppercase tracking-wide block">
                    Recommended Treatment / Pharmacotherapy
                  </span>
                  <p className="text-slate-700 leading-relaxed">{analysisResult.treatment.recommendedTreatment}</p>
                </div>

                {/* Contraindications & Critical Warnings */}
                <div className="bg-red-50/50 border border-red-200 rounded-xl p-4 space-y-2">
                  <span className="font-bold text-red-900 uppercase tracking-wide block flex items-center space-x-1">
                    <span>Contraindications &amp; Warnings</span>
                  </span>
                  <p className="text-red-800 font-medium leading-relaxed">{analysisResult.treatment.contraindications}</p>
                </div>

                {/* Non-Pharmacological / Lifestyle */}
                <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-2">
                  <span className="font-bold text-slate-800 uppercase tracking-wide block">
                    Lifestyle &amp; Non-Pharmacological
                  </span>
                  <p className="text-slate-700 leading-relaxed">{analysisResult.treatment.lifestyleRecommendations}</p>
                </div>

                {/* Referral & Follow-up */}
                <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-2">
                  <span className="font-bold text-slate-800 uppercase tracking-wide block">
                    Referral &amp; Follow-up
                  </span>
                  <p className="text-slate-700">
                    <strong>Referral:</strong> {analysisResult.treatment.specialistReferral}
                  </p>
                  <p className="text-slate-700 mt-1">
                    <strong>Timeline:</strong> {analysisResult.treatment.followUpAdvice}
                  </p>
                </div>
              </div>

              {/* Guideline Citation */}
              {analysisResult.treatment.evidenceSource && (
                <div className="mt-3 text-[11px] text-slate-500 font-mono flex items-center space-x-1">
                  <span className="font-bold text-slate-700">Clinical Guideline Reference:</span>
                  <span>{analysisResult.treatment.evidenceSource} (v{analysisResult.treatment.guidelineVersion})</span>
                </div>
              )}
            </div>

            {/* Differential Diagnoses */}
            {analysisResult.differentials.length > 0 && (
              <div className="pt-4 border-t border-slate-200">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">
                  Integrated Differential Diagnoses
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
                  {analysisResult.differentials.map((diff, idx) => (
                    <div key={idx} className="p-2.5 rounded-xl border border-slate-200 bg-slate-50 flex items-center justify-between text-xs">
                      <span className="font-semibold text-slate-800">{diff.condition}</span>
                      {diff.probability !== undefined && (
                        <span className="font-mono text-slate-600">{Math.round(diff.probability * 100)}%</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Disclaimer */}
            <div className="p-3 bg-amber-50 border border-amber-200 rounded-xl text-[11px] text-amber-900 leading-normal">
              <strong>Medical Disclaimer:</strong> Health Insights is an AI-assisted Clinical Decision Support research tool. All diagnostic predictions, consensus assessments, and treatment suggestions must be verified by a licensed healthcare professional before taking clinical action.
            </div>
          </section>
        )}

        {/* 6. TELEMETRY & AUDIT INSPECTOR (COLLAPSIBLE) */}
        {analysisResult && (
          <section className="bg-white rounded-2xl border border-slate-200 p-5 shadow-xs">
            <button
              onClick={() => setShowTelemetry(!showTelemetry)}
              className="w-full flex items-center justify-between text-xs font-bold uppercase tracking-wider text-slate-600 hover:text-slate-900"
            >
              <div className="flex items-center space-x-2">
                <span>Technical Telemetry &amp; System Audit</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-500 font-normal">
                  {Math.round(analysisResult.totalLatencyMs)}ms • {analysisResult.symptoms.length} symptoms • {analysisResult.routing.selectedSpecialists.length} specialists
                </span>
              </div>
              {showTelemetry ? <ChevronUpIcon className="w-4 h-4" /> : <ChevronDownIcon className="w-4 h-4" />}
            </button>

            {showTelemetry && (
              <div className="mt-4 pt-4 border-t border-slate-200 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                    <span className="font-bold text-slate-700 block mb-1">RIE Extracted Entities</span>
                    <p className="text-slate-600 font-mono text-[11px]">
                      <strong>Symptoms:</strong> {analysisResult.symptoms.join(', ') || 'None'}
                    </p>
                    <p className="text-slate-600 font-mono text-[11px] mt-1">
                      <strong>Risk Factors:</strong> {analysisResult.riskFactors.join(', ') || 'None'}
                    </p>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                    <span className="font-bold text-slate-700 block mb-1">Router Dispatch</span>
                    <p className="text-slate-600 font-mono text-[11px]">
                      <strong>Specialists:</strong> {analysisResult.routing.selectedSpecialists.join(', ') || 'General'}
                    </p>
                    <p className="text-slate-600 font-mono text-[11px] mt-1">
                      <strong>Primary Domain:</strong> {analysisResult.routing.primaryDomain}
                    </p>
                  </div>
                  <div className="p-3 bg-slate-50 rounded-xl border border-slate-200">
                    <span className="font-bold text-slate-700 block mb-1">RAC Multi-Agent Consensus</span>
                    <p className="text-slate-600 font-mono text-[11px]">
                      <strong>Consensus Ratio:</strong> {Math.round(analysisResult.rac.agreementScore * 100)}%
                    </p>
                    <p className="text-slate-600 font-mono text-[11px] mt-1">
                      <strong>Conflict Penalty:</strong> {analysisResult.rac.conflictPenalty}
                    </p>
                  </div>
                </div>

                <div>
                  <span className="text-[11px] font-bold text-slate-700 block mb-1.5">Raw JSON Response from FastAPI Backend:</span>
                  <pre className="p-4 bg-slate-900 text-slate-200 rounded-xl text-[11px] font-mono overflow-x-auto max-h-72 custom-scrollbar">
                    {JSON.stringify(analysisResult.raw, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </section>
        )}
      </main>

      {/* ──────────────────────────────────────────────
          FOOTER
      ────────────────────────────────────────────── */}
      <footer className="bg-white border-t border-slate-200 py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between text-xs text-slate-500 gap-2">
          <div>
            Health Insights CDS System • Major Project Thesis Evaluation
          </div>
          <div className="flex items-center space-x-4">
            <span>Specialists: Cardiologist V2 | Pulmonologist V1 | Neurologist V1</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
