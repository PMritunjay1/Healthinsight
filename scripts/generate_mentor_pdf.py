import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PROJECT_ROOT = r"c:\Users\pmrit\Downloads\Health-Insights-Major-Project-main"
OUTPUT_PDF_PATH = os.path.join(PROJECT_ROOT, "reports", "mentor_progress_report.pdf")

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(40, 756, "HealthInsight — Student Project Progress & Mentor Review Report")
            self.drawRightString(572, 756, "October 2026")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(40, 750, 572, 750)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(40, 38, 572, 38)
        self.drawString(40, 26, "HealthInsight: Multi-Specialist Clinical Decision Support | Pratyush Mritunjay")
        self.drawRightString(572, 26, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_pdf():
    # Page width: 612 pt, margins: 40 pt -> printable width = 532 pt
    # Page height: 792 pt, margins: 42 pt -> printable height = 708 pt
    doc = SimpleDocTemplate(
        OUTPUT_PDF_PATH,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()
    
    # Palette
    c_primary = colors.HexColor("#1A365D")   # Deep Navy
    c_secondary = colors.HexColor("#2B6CB0") # Slate Blue
    c_accent = colors.HexColor("#2C7A7B")    # Teal Accent
    c_text = colors.HexColor("#2D3748")      # Charcoal Body Text
    c_light_bg = colors.HexColor("#F7FAFC")  # Table Alt Row
    c_border = colors.HexColor("#E2E8F0")    # Table Border
    c_box_bg = colors.HexColor("#EDF2F7")    # Code/Box Background

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=21,
        textColor=c_primary,
        spaceAfter=2
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=6
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13.5,
        textColor=c_primary,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=c_secondary,
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=11,
        textColor=c_text,
        spaceAfter=3.5
    )
    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.6,
        leading=10.5,
        textColor=c_text,
        leftIndent=10,
        spaceAfter=2
    )
    algo_step_style = ParagraphStyle(
        'AlgoStep',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10.5,
        textColor=c_text,
        leftIndent=12,
        firstLineIndent=-12,
        spaceAfter=2
    )
    algo_note_style = ParagraphStyle(
        'AlgoNote',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#4A5568"),
        spaceBefore=2,
        spaceAfter=2
    )
    callout_style = ParagraphStyle(
        'Callout_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.6,
        leading=10.5,
        textColor=c_primary
    )
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9,
        textColor=colors.white,
        alignment=1 # Center
    )
    table_text = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        textColor=c_text
    )
    table_bold = ParagraphStyle(
        'TableBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=c_text
    )

    story = []

    # ==================== PAGE 1 ====================
    # Title & Metadata
    story.append(Paragraph("HealthInsight: Project Progress & Mentor Review Report", title_style))
    story.append(Paragraph(
        "<b>Student:</b> Pratyush Mritunjay &nbsp;|&nbsp; "
        "<b>Project:</b> Multi-Specialist Clinical Decision Support System &nbsp;|&nbsp; "
        "<b>Context:</b> Informal Progress Report for Supervisor Meeting (5–10 Min Walkthrough) &nbsp;|&nbsp; "
        "<b>Date:</b> October 2026",
        meta_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceAfter=5))

    # SECTION 1
    story.append(Paragraph("1. What My Mentor Asked Me to Improve", h1_style))
    story.append(Paragraph(
        "During our previous project discussions, my mentor gave me clear feedback: HealthInsight should not be treated as a finished, production-ready system, but as a base framework that needed rigorous engineering, calibration, and honest evaluation. "
        "Specifically, my mentor asked me to address nine core areas: (1) clarify the execution architecture and multi-agent interaction, (2) fix neural overconfidence using formal calibration, (3) introduce selective abstention for clinician referral, "
        "(4) run end-to-end full-system evaluation across clinical encounters, (5) benchmark against standard baselines, (6) provide statistical significance tests and reliability curves, (7) document training data splits and patient isolation, "
        "(8) test adversarial robustness under specialist conflict and degraded narratives (Direction 9), and (9) audit citations and transparently document limitations.",
        body_style
    ))
    story.append(Paragraph("Below is a status summary of how I addressed each of these nine requirements in the codebase:", body_style))

    # Table 1: Status of the Nine Mentor Requirements (532 pt total: 95, 175, 192, 70)
    req_headers = [
        Paragraph("Mentor Requirement", table_header),
        Paragraph("What I Implemented in the Codebase", table_header),
        Paragraph("Source Code & Evidence", table_header),
        Paragraph("Status", table_header)
    ]
    req_rows = [
        req_headers,
        [
            Paragraph("<b>1. Architecture clarification</b>", table_text),
            Paragraph("Traced exact pipeline; verified selective subset execution (1 to 3 specialists); clarified that RAC is an agreement module, not RAG.", table_text),
            Paragraph("<code>backend/orchestrator.py</code><br/><code>backend/specialist_router.py</code>", table_text),
            Paragraph("<font color='#276749'><b>Completed</b></font>", table_bold)
        ],
        [
            Paragraph("<b>2. Calibration</b>", table_text),
            Paragraph("Fitted logistic regression and isotonic calibrators on 226 validation cases; generated reliability curves; calculated ECE and Brier score.", table_text),
            Paragraph("<code>scripts/train_calibrator.py</code><br/><code>results/isotonic_experiment.json</code><br/><code>results/graphs/reliability_diagram.png</code>", table_text),
            Paragraph("<font color='#276749'><b>Completed</b></font>", table_bold)
        ],
        [
            Paragraph("<b>3. Selective abstention</b>", table_text),
            Paragraph("Built a dual-threshold decision gate in RAC (tau=0.10, H_limit=0.70) that withholds low-confidence or high-entropy predictions.", table_text),
            Paragraph("<code>backend/rac_module.py</code><br/><code>backend/models/abstention_config.json</code>", table_text),
            Paragraph("<font color='#276749'><b>Completed</b></font>", table_bold)
        ],
        [
            Paragraph("<b>4. Full-system evaluation</b>", table_text),
            Paragraph("Ran all 209 hold-out cases end-to-end; conducted a diagnostic scope audit to investigate why overall accuracy is 32.06%.", table_text),
            Paragraph("<code>scripts/evaluate_full_system.py</code><br/><code>reports/full_system_accuracy_audit.md</code>", table_text),
            Paragraph("<font color='#276749'><b>Completed</b></font>", table_bold)
        ],
        [
            Paragraph("<b>5. Baseline comparisons</b>", table_text),
            Paragraph("Evaluated 5 baseline setups on identical 209 cases: uncalibrated, deterministic, simple voting, calibrated abstention, and trust-weighted.", table_text),
            Paragraph("<code>results/full_system_baseline_comparison.json</code><br/><code>reports/ieee_project_report.md</code> (Table 9)", table_text),
            Paragraph("<font color='#276749'><b>Completed</b></font>", table_bold)
        ],
        [
            Paragraph("<b>6. Statistical evidence</b>", table_text),
            Paragraph("Ran exact McNemar binomial paired tests on fusion outputs; verified bootstrap CIs for individual models; noted missing full-system CIs.", table_text),
            Paragraph("<code>scripts/stats_full_system.py</code><br/><code>results/direction9_robustness_results.json</code>", table_text),
            Paragraph("<font color='#B7791F'><b>Partially completed</b></font>", table_bold)
        ],
        [
            Paragraph("<b>7. Data pipeline documentation</b>", table_text),
            Paragraph("Documented MIMIC-IV training cohorts (101k cardio, 97k pulmo, 67k neuro) with 0% patient leakage; documented 209-case benchmark sources.", table_text),
            Paragraph("<code>docs/POST_E1_EVIDENCE_MANIFEST.json</code><br/><code>docs/final_leakage_audit.json</code>", table_text),
            Paragraph("<font color='#276749'><b>Completed</b></font>", table_bold)
        ],
        [
            Paragraph("<b>8. Robustness & Direction 9</b>", table_text),
            Paragraph("Implemented dynamic trust weighting based on logit sampling instability; stress-tested on 209 cases across clean, conflict, and degraded settings.", table_text),
            Paragraph("<code>backend/orchestrator.py</code><br/><code>results/direction9_robustness_results.json</code>", table_text),
            Paragraph("<font color='#276749'><b>Completed</b></font>", table_bold)
        ],
        [
            Paragraph("<b>9. Report & reference quality</b>", table_text),
            Paragraph("Audited all 36 IEEE citations; added Section 9.7 and Table 10 to the main report; verified PDF formatting with zero visual flaws.", table_text),
            Paragraph("<code>reports/ieee_project_report.md</code><br/><code>docs/Health_Insights_Post_E1_Specialist_Report.pdf</code>", table_text),
            Paragraph("<font color='#276749'><b>Completed</b></font>", table_bold)
        ]
    ]
    t_req = Table(req_rows, colWidths=[105, 175, 175, 77])
    t_req.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_primary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_req)
    story.append(Spacer(1, 6))

    # SECTION 2
    story.append(Paragraph("2. What I Changed in HealthInsight", h1_style))
    story.append(Paragraph(
        "To turn the original prototype into an auditable, transparent engineering system, I made four major architectural and evaluation updates:",
        body_style
    ))
    story.append(Paragraph("• <b>Replaced external cloud APIs with local neural specialists:</b> Replaced cloud LLM dependencies with three frozen, local multi-task models fine-tuned from PubMedBERT-base on PhysioNet MIMIC-IV encounters: Cardiologist V2 (6 cardiac classes), Pulmonologist V1 (6 respiratory classes), and Neurologist V1 (6 neurological classes). Models are quantized to INT8 ONNX and execute locally on standard CPU.", bullet_style))
    story.append(Paragraph("• <b>Audited routing and multi-agent execution:</b> Input narratives are preprocessed by the RIE regex engine (entity and negation extraction). An Intent Agent (DistilBERT ONNX) predicts general clinical task type. The Specialist Router then evaluates domain keyword and symptom scores. Crucially, the router invokes only a <b>selective subset</b>: a single specialist runs if the domain is unambiguous (91.4% of benchmark cases), while 2 or 3 specialists run in parallel only when symptoms overlap (8.6% of cases). RAC is our <b>Robust Agreement & Consensus Module</b>—a mathematical routine measuring agreement and entropy, not text retrieval (RAG).", bullet_style))
    story.append(Paragraph("• <b>Implemented Direction 1 (Calibration & Abstention):</b> Fitted a logistic regression calibrator on validation cases mapping raw confidence, specialist instability, and domain entropy to calibrated probabilities, backed by a dual-threshold gate to abstain on high uncertainty.", bullet_style))
    story.append(Paragraph("• <b>Implemented Direction 9 (Trust-Weighted Consensus):</b> In multi-agent conflicts, replaced static confidence maximization with dynamic trust weighting derived from specialist logit sampling instability.", bullet_style))

    story.append(PageBreak())

    # ==================== PAGE 2 ====================
    # SECTION 3
    story.append(Paragraph("3. How Direction 1 Works (Calibration and Selective Prediction)", h1_style))
    story.append(Paragraph(
        "<i>Calibration adjusts the interpretation of confidence scores so that a predicted probability matches empirical correctness, while selective abstention allows the system to withhold a prediction when its decision rule indicates high uncertainty.</i>",
        body_style
    ))

    # Algorithm 1 Box
    algo1_content = [
        Paragraph("<b>Algorithm A: Direction 1 — Calibration and Abstention Pipeline (Implemented)</b>", h2_style),
        Paragraph("<b>1. Feature Extraction:</b> The active specialist models output primary diagnoses and raw confidences. The RAC module extracts three features: (a) base confidence <i>c_base = max_i(c_i)</i>, (b) prediction instability <i>u_instability</i> from Bernoulli sampling variance of the winning logit, and (c) domain entropy <i>E_dom = -∑ p_i · ln(p_i)</i> over normalized specialist confidences (or 0.0 if only one specialist ran).", algo_step_style),
        Paragraph("<b>2. Logistic Calibration:</b> The feature vector <i>x = [c_base, u_instability, E_dom]</i> is fed to the fitted logistic regression model in <code>backend/models/calibrator.pkl</code>, computing calibrated probability <i>c_cal = sigmoid(w^T · x + b)</i>.", algo_step_style),
        Paragraph("<b>3. Threshold Loading:</b> The system retrieves decision thresholds from <code>backend/models/abstention_config.json</code>: confidence threshold <i>tau = 0.10</i> and domain entropy limit <i>H_limit = 0.70</i>.", algo_step_style),
        Paragraph("<b>4. Abstention Decision Gate:</b> The case is evaluated against the dual decision rule: if <i>c_cal &lt; tau</i> OR <i>E_dom &gt; H_limit</i>, the system withholds diagnosis and outputs <i>'Uncertain — refer to clinician'</i>.", algo_step_style),
        Paragraph("<b>5. Prediction Output:</b> If the case passes both thresholds, the deterministic fusion winner is returned as the final predicted disease.", algo_step_style),
        Paragraph("<b>6. Metric Recording:</b> Coverage (<i>N_covered / N_total</i>), covered-case accuracy (<i>N_correct_covered / N_covered</i>), ECE, and Brier score are evaluated strictly across the answered cohort.", algo_step_style),
        Paragraph("<i>Verified implementation:</i> <code>backend/rac_module.py</code> (lines 118–168), <code>backend/models/abstention_config.json</code>, and <code>scripts/evaluate_full_system.py</code>.", algo_note_style)
    ]
    t_algo1 = Table([[algo1_content]], colWidths=[532])
    t_algo1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ('BOX', (0, 0), (-1, -1), 1, c_secondary),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_algo1)
    story.append(Spacer(1, 6))

    # SECTION 4
    story.append(Paragraph("4. How Direction 9 Works (Trust-Weighted Consensus)", h1_style))
    story.append(Paragraph(
        "<i>Direction 9 dynamically adjusts the influence of specialist predictions during multi-agent consensus by down-weighting specialists that exhibit high prediction instability, rather than relying strictly on uncalibrated confidence scores.</i>",
        body_style
    ))

    # Algorithm 2 Box
    algo2_content = [
        Paragraph("<b>Algorithm B: Direction 9 — Dynamic Trust-Weighted Consensus (Implemented)</b>", h2_style),
        Paragraph("<b>1. Multi-Specialist Co-Activation:</b> The Specialist Router identifies domains scoring ≥ 0.35. If secondary domains score at least 50% of the top domain, multiple specialists are co-activated (occurring in 8.6% of benchmark cases).", algo_step_style),
        Paragraph("<b>2. Specialist Inference:</b> Each active specialist executes its local INT8 PubMedBERT ONNX graph, producing 6 disease probabilities.", algo_step_style),
        Paragraph("<b>3. Instability Estimation:</b> In <code>specialist_agents.py</code>, the specialist applies temperature scaling (<i>T = 1.0</i>) to its primary logit <i>z_primary</i> to get <i>p = sigmoid(z_primary)</i>, draws <i>m = 5</i> Bernoulli samples, and computes instability as the sample variance <i>u_instability = Var({y_1, ..., y_5})</i>.", algo_step_style),
        Paragraph("<b>4. Trust Weight Calculation:</b> For each active specialist <i>i</i>, an individual trust score is computed: <i>Trust_i = c_i · (1.0 - u_instability, i)</i>, where <i>c_i</i> is raw disease confidence.", algo_step_style),
        Paragraph("<b>5. Weighted Consensus Scoring:</b> The trust score is modulated by the domain routing weight: <i>Score_i = W_domain(primary, i) · Trust_i</i>. Secondary specialists receive a minimum base weight of 0.70.", algo_step_style),
        Paragraph("<b>6. Final Diagnosis Selection:</b> The specialist achieving <i>max_i(Score_i)</i> is selected as the dominant consensus winner, and its primary disease is emitted.", algo_step_style),
        Paragraph("<b>7. Stress Testing:</b> Assessed across 209 cases under three conditions: clean narratives, severe conflict (injected conflicting specialist logits), and degraded narratives (noisy/scrambled symptoms).", algo_step_style),
        Paragraph("<i>Verified implementation:</i> <code>backend/orchestrator.py</code> (lines 170–243), <code>backend/agents/specialist_agents.py</code>, and <code>results/direction9_robustness_results.json</code>.", algo_note_style)
    ]
    t_algo2 = Table([[algo2_content]], colWidths=[532])
    t_algo2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ('BOX', (0, 0), (-1, -1), 1, c_secondary),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_algo2)
    story.append(Spacer(1, 6))

    # SECTION 5.1
    story.append(Paragraph("5. Results and What They Tell Us", h1_style))
    story.append(Paragraph("<b>5.1 Standalone Calibration Results (226 Held-Out Cases)</b>", h2_style))
    story.append(Paragraph(
        "Before full-system integration, I evaluated the calibrators on 226 held-out validation encounters to verify whether probability distortion was corrected:",
        body_style
    ))

    # Table 2: Standalone Calibration (532 pt total: 140, 70, 75, 70, 177)
    cal_headers = [
        Paragraph("Calibrator Method", table_header),
        Paragraph("ECE (Lower is Better)", table_header),
        Paragraph("Brier Score (Lower is Better)", table_header),
        Paragraph("Accuracy (100% Cov)", table_header),
        Paragraph("Calibration Outcome", table_header)
    ]
    cal_rows = [
        cal_headers,
        [Paragraph("<b>Uncalibrated Baseline</b>", table_text), Paragraph("0.4974", table_text), Paragraph("0.4485", table_text), Paragraph("28.76%", table_text), Paragraph("Severely overconfident raw probabilities", table_text)],
        [Paragraph("<b>Logistic Calibration (Direction 1)</b>", table_text), Paragraph("<b>0.0447</b>", table_bold), Paragraph("<b>0.1870</b>", table_bold), Paragraph("28.76%", table_text), Paragraph("Well-calibrated (11.1x reduction in ECE)", table_text)],
        [Paragraph("<b>Isotonic Regression</b>", table_text), Paragraph("<b>0.0366</b>", table_bold), Paragraph("<b>0.1898</b>", table_bold), Paragraph("28.76%", table_text), Paragraph("Comparable non-parametric monotonic fit", table_text)]
    ]
    t_cal = Table(cal_rows, colWidths=[140, 70, 75, 70, 177])
    t_cal.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_secondary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_cal)
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<i>Key takeaway:</i> Both calibrators dramatically resolve overconfidence (ECE dropped from ~0.50 to ~0.04). However, calibration alone does not change raw classification accuracy under full coverage (remains 28.76%) because monotonic scaling preserves the argmax winner. The reliability plot is saved at <code>results/graphs/reliability_diagram.png</code>.",
        body_style
    ))

    story.append(PageBreak())

    # ==================== PAGE 3 ====================
    # SECTION 5.2
    story.append(Paragraph("5.2 Full-System Benchmark & Diagnostic Scope Audit (209 Cases)", h2_style))
    story.append(Paragraph(
        "Across 209 untouched hold-out clinical cases evaluated end-to-end (raw narrative to final consensus), the reported exact-match accuracy is <b>32.06% (67/209 correct)</b>. "
        "To understand why accuracy was 32.06%, I audited all 209 cases (<code>reports/full_system_accuracy_audit.md</code>). The findings are summarized below:",
        body_style
    ))

    # Table 3: Full-Cohort Scope Breakdown (532 pt total: 125, 38, 38, 48, 48, 235)
    scope_headers = [
        Paragraph("Cohort Stratum", table_header),
        Paragraph("Cases (N)", table_header),
        Paragraph("Correct", table_header),
        Paragraph("Accuracy", table_header),
        Paragraph("Coverage", table_header),
        Paragraph("Primary Failure Mode / Root Cause", table_header)
    ]
    scope_rows = [
        scope_headers,
        [Paragraph("<b>Full Retrospective Cohort</b>", table_text), Paragraph("209", table_text), Paragraph("67", table_text), Paragraph("<b>32.06%</b>", table_bold), Paragraph("100.0%", table_text), Paragraph("Scope mismatch (52.8%) + specialist error (35.9%) + routing (11.3%)", table_text)],
        [Paragraph("<b>Out-of-Scope Diseases</b>", table_text), Paragraph("75", table_text), Paragraph("0", table_text), Paragraph("<b>0.00%</b>", table_bold), Paragraph("100.0%", table_text), Paragraph("Unsupported targets: ILD (20), TB (8), Spondylosis (20), Migraine (20), MS (7)", table_text)],
        [Paragraph("<b>In-Scope Diagnostic Subset</b>", table_text), Paragraph("134", table_text), Paragraph("67", table_text), Paragraph("<b>50.00%</b>", table_bold), Paragraph("100.0%", table_text), Paragraph("Specialist misclassifications (N=51) + routing errors (N=16)", table_text)],
        [Paragraph("<b>In-Scope & Correctly Routed</b>", table_text), Paragraph("118", table_text), Paragraph("67", table_text), Paragraph("<b>56.78%</b>", table_bold), Paragraph("100.0%", table_text), Paragraph("Specialist discrimination limits & MIMIC-IV class skew (e.g. angina 0/20)", table_text)],
        [Paragraph("<b>In-Scope & Misrouted</b>", table_text), Paragraph("16", table_text), Paragraph("0", table_text), Paragraph("<b>0.00%</b>", table_bold), Paragraph("100.0%", table_text), Paragraph("Cross-domain symptom overlap (dyspnea in CHF, headache in HTN)", table_text)],
        [Paragraph("<b>Direction 1 (Full Cohort)</b>", table_text), Paragraph("209", table_text), Paragraph("56", table_text), Paragraph("<b>35.67%</b>", table_bold), Paragraph("75.12%", table_text), Paragraph("Selective prediction: abstained on 52 cases (21 in-scope, 31 out-of-scope)", table_text)],
        [Paragraph("<b>Direction 1 (In-Scope Only)</b>", table_text), Paragraph("134", table_text), Paragraph("56", table_text), Paragraph("<b>49.56%</b>", table_bold), Paragraph("84.33%", table_text), Paragraph("Abstained on 21 in-scope cases; covered accuracy 49.56% (56/113 answered)", table_text)],
        [Paragraph("<b>Majority-Class Baseline</b>", table_text), Paragraph("209", table_text), Paragraph("20", table_text), Paragraph("<b>9.57%</b>", table_bold), Paragraph("100.0%", table_text), Paragraph("Always predicting dominant cohort disease (angina, N=20)", table_text)],
        [Paragraph("<b>Random Guessing (18 Classes)</b>", table_text), Paragraph("209", table_text), Paragraph("—", table_text), Paragraph("<b>5.56%</b>", table_bold), Paragraph("100.0%", table_text), Paragraph("Uniform random guessing across 18 specialist disease classes (1/18)", table_text)]
    ]
    t_scope = Table(scope_rows, colWidths=[125, 38, 38, 48, 48, 235])
    t_scope.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_secondary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 3.5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_scope)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Key Insights from the 209-Case Evaluation:</b><br/>"
        "1. <b>The 75 Out-of-Scope Cases (35.89% of cohort):</b> The 3 specialist models predict 6 specific diseases each (18 total). But the 209-case benchmark includes diseases from external datasets like interstitial lung disease (20), tuberculosis (8), cervical spondylosis (20), migraine (20), and multiple sclerosis (7). Because these labels do not exist in the models' output heads, accuracy on them was strictly 0.00% (0/75), driving 52.8% of all 142 cohort errors.<br/>"
        "2. <b>In-Scope Accuracy Rises to 50.00% (67/134):</b> Restricting evaluation to diseases the models were actually trained to recognize raises accuracy from 32.06% to 50.00% (+17.94 pp), and to 56.78% (67/118) when the router selected the correct specialist.<br/>"
        "3. <b>Why In-Scope Performance Still Leaves 50% Error:</b> Out-of-scope cases do not explain all failures. Among the 134 in-scope cases, 16 failed due to multi-organ routing overlap (e.g., dyspnea in heart failure routed to pulmonology), and 51 failed due to specialist misclassifications. Crucially, <b>angina achieved 0/20 accuracy</b>: in MIMIC-IV training data, angina had only 113 records (0.11% prevalence) versus 13,315 for hypertension. Under argmax selection, dominant classes (arrhythmia in 17 cases, hypertension in 3 cases) consistently outscored angina.",
        body_style
    ))
    story.append(Spacer(1, 4))

    # SECTION 5.3
    story.append(Paragraph("5.3 Direction 9 Robustness Results (209 Cases)", h2_style))
    story.append(Paragraph(
        "I stress-tested Direction 9 against deterministic fusion (Baseline B) on all 209 cases across three perturbation regimes:",
        body_style
    ))

    # Table 4: Direction 9 Adversarial Breakdown (532 pt total: 85, 80, 80, 57, 55, 175)
    d9_headers = [
        Paragraph("Test Setting", table_header),
        Paragraph("Deterministic (B)", table_header),
        Paragraph("Trust-Weighted (Dir 9)", table_header),
        Paragraph("Discordant", table_header),
        Paragraph("p-value", table_header),
        Paragraph("Empirical Interpretation", table_header)
    ]
    d9_rows = [
        d9_headers,
        [Paragraph("<b>Clean Inputs</b>", table_text), Paragraph("53 / 209 (25.36%)", table_text), Paragraph("52 / 209 (24.88%)", table_text), Paragraph("2 vs. 3", table_text), Paragraph("p = 1.0000", table_text), Paragraph("Nearly identical performance on clean clinical text", table_text)],
        [Paragraph("<b>Severe Conflict</b>", table_text), Paragraph("6 / 209 (2.87%)", table_text), Paragraph("<b>9 / 209 (4.31%)</b>", table_bold), Paragraph("0 vs. 3", table_text), Paragraph("p = 0.2500", table_text), Paragraph("Trust weighting successfully down-weights erratic logits", table_text)],
        [Paragraph("<b>Input Degradation</b>", table_text), Paragraph("51 / 209 (24.40%)", table_text), Paragraph("37 / 209 (17.70%)", table_text), Paragraph("17 vs. 3", table_text), Paragraph("<b>p = 0.00257</b>", table_bold), Paragraph("Instability penalization is overly conservative under noise", table_text)]
    ]
    t_d9 = Table(d9_rows, colWidths=[85, 80, 80, 57, 55, 175])
    t_d9.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_secondary),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_d9)
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<i>What this tells us:</i> Direction 9 functions as intended under conflict (improving correct cases from 6 to 9). However, it is not universally superior: under noisy inputs, it penalizes valid specialists too aggressively, significantly reducing accuracy (p=0.00257). This highlights a realistic clinical engineering trade-off.",
        body_style
    ))

    story.append(PageBreak())

    # ==================== PAGE 4 ====================
    # SECTION 6
    story.append(Paragraph("6. What Is Still Incomplete", h1_style))
    story.append(Paragraph(
        "To maintain scientific integrity during review, I have explicitly documented what remains incomplete or unfeasible:",
        body_style
    ))
    story.append(Paragraph("• <b>Abstention is not an out-of-distribution detector:</b> Direction 1 abstained on 52 cases, but 44 out-of-scope cases (58.67% of unsupported cases) were still covered and given incorrect diagnoses. Entropy gating alone cannot distinguish unsupported medical conditions from confident misclassifications. True OOD detection requires an explicit density-based or open-set rejection model.", bullet_style))
    story.append(Paragraph("• <b>In-scope accuracy (50.00%) does not establish clinical readiness:</b> Failing on half of the in-scope encounters confirms that HealthInsight is an offline research prototype, not a safe diagnostic system. Multi-label argmax scoring must be replaced with threshold-relative margins to prevent severe class collapse on imbalanced conditions like angina (0/20).", bullet_style))
    story.append(Paragraph("• <b>Full-system confidence intervals across multiple seeds:</b> Bootstrap 95% confidence intervals were computed for individual specialist models across 39,594 records, but multi-seed bootstrap intervals for end-to-end full-system fusion on the 209 cases remain incomplete due to benchmark freezing.", bullet_style))
    story.append(Paragraph("• <b>No monolithic joint baseline (Baseline D):</b> A single monolithic multi-specialty transformer was not trained during the original project phase and cannot be synthesized without retraining from raw MIMIC-IV text.", bullet_style))
    story.append(Spacer(1, 6))

    # SECTION 7
    story.append(Paragraph("7. What I Can Explain in the Meeting (Talking Points)", h1_style))
    story.append(Paragraph(
        "If my mentor asks for a 2-minute verbal summary of the project's current state, here is the exact progression I will explain:",
        body_style
    ))

    # Summary Callout Box for Verbal Explanation
    verbal_box_data = [
        [Paragraph(
            "<b>Verbal Walkthrough for Supervisor Meeting:</b><br/>"
            "1. <b>Architecture:</b> HealthInsight does not execute all specialists on every case. The router runs only the single relevant specialist in 91.4% of benchmark cases, co-activating multiple agents only when symptoms overlap (8.6%). RAC is our mathematical Agreement & Consensus module, not text retrieval (RAG).<br/>"
            "2. <b>The 32.06% Full-Cohort Result:</b> Evaluated on 209 untouched hold-out cases, accuracy was 32.06% (67/209). My audit revealed that 75 cases (35.89%) present diseases like tuberculosis, interstitial lung disease, or spondylosis that simply do not exist in our specialists' 6-class output heads, driving 52.8% of all system errors.<br/>"
            "3. <b>In-Scope Performance (50.00%):</b> On the 134 cases the models were trained to recognize, accuracy is 50.00% (and 56.78% when routed correctly). The remaining errors stem from 16 cross-domain routing failures and 51 specialist mistakes—notably angina at 0/20 due to extreme 0.11% MIMIC-IV training prevalence.<br/>"
            "4. <b>Calibration & Direction 1:</b> Logistic calibration reduced ECE from 0.4974 to 0.0447 on 226 validation encounters. Direction 1 abstains on 24.88% of cases, raising covered accuracy to 35.67% overall and 49.56% on in-scope cases. However, abstention is not an OOD detector and still passed 44 unsupported cases.<br/>"
            "5. <b>Direction 9 Robustness:</b> Instability-weighted consensus helps under severe conflict (9 vs 6 correct) by down-weighting erratic logits, but exhibits conservative trade-offs under degraded inputs (37 vs 51).<br/>"
            "6. <b>Current Status:</b> All 36 IEEE references have been verified, Section 9.7 and Table 10 have been incorporated into the main report, and all model weights and evaluation files remain frozen.",
            callout_style
        )]
    ]
    t_verbal = Table(verbal_box_data, colWidths=[532])
    t_verbal.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
        ('BOX', (0, 0), (-1, -1), 1, c_secondary),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_verbal)

    # Build PDF
    print(f"Building PDF at: {OUTPUT_PDF_PATH}...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print("Mentor progress report PDF built successfully!")

if __name__ == "__main__":
    build_pdf()
