"""Generates a professional PDF document for the Enterprise Agentic RAG Platform Architecture and Guide."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


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
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "Enterprise Agentic RAG Platform — Architecture & Production Guide")
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 742, letter[0] - 54, 742)
            
        # Footer
        text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 36, text)
        self.drawString(54, 36, "CONFIDENTIAL & PROPRIETARY — AZURE AI FOUNDRY EDITION")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 48, letter[0] - 54, 48)
        self.restoreState()


def build_pdf(filename="Enterprise_Agentic_RAG_Architecture_Guide.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0f172a")     # Deep slate
    c_accent = colors.HexColor("#1d4ed8")      # Azure Blue
    c_body = colors.HexColor("#334155")        # Charcoal text
    c_code_bg = colors.HexColor("#f8fafc")     # Light background
    c_box_border = colors.HexColor("#cbd5e1")  # Border grey

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=c_primary,
        spaceAfter=6,
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=c_accent,
        spaceAfter=20,
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=c_primary,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=c_accent,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=c_body,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "BulletText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=c_body,
        leftIndent=14,
        spaceAfter=4,
    )

    code_style = ParagraphStyle(
        "CodeText",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
    )

    story = []

    # Title & Header
    story.append(Paragraph("Enterprise Agentic RAG Platform", title_style))
    story.append(Paragraph("End-to-End System Architecture, Code Design, LangGraph Workflow & Azure Deployment Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_accent, spaceBefore=0, spaceAfter=15))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary & Problem Statement", h1_style))
    story.append(Paragraph(
        "This project implements an enterprise-grade, multi-agent conversational AI platform capable of retrieving answers from a "
        "knowledge base containing <b>100,000+ unstructured documents</b> (PDF, DOCX, PPTX, TXT, HTML) and performing <b>real-time web search</b> via "
        "the Model Context Protocol (MCP). The entire architecture is built with Clean Architecture, LangGraph, FastAPI, Streamlit, and Azure Cloud Services.",
        body_style
    ))
    story.append(Paragraph("<b>Core System Principles:</b>", body_style))
    story.append(Paragraph("• <b>Strict Minimalist UX:</b> ChatGPT / Copilot style interface with zero administrative clutter.", bullet_style))
    story.append(Paragraph("• <b>Zero-Cost Local Embeddings:</b> CPU ONNX FastEmbed engine avoiding API costs and rate limits.", bullet_style))
    story.append(Paragraph("• <b>Serverless Cloud Infrastructure:</b> Scaled-to-Zero ($0 idle cost) on Azure Container Apps and Azure Cosmos DB Free Tier.", bullet_style))
    story.append(Paragraph("• <b>Verified Grounding:</b> Real-time bracket citations ([1], [2]) verified by an automated citation guardrails agent.", bullet_style))

    story.append(Spacer(1, 10))

    # 2. Key Architecture Components
    story.append(Paragraph("2. System Architecture & Components", h1_style))
    
    arch_data = [
        [Paragraph("<b>Component</b>", body_style), Paragraph("<b>Technology</b>", body_style), Paragraph("<b>Key Role & Configuration</b>", body_style)],
        [Paragraph("<b>Frontend UI</b>", body_style), Paragraph("Streamlit 1.39+", body_style), Paragraph("Minimalist chat UI, thread switching, citation expanders, feedback (port 8501).", body_style)],
        [Paragraph("<b>Backend API</b>", body_style), Paragraph("FastAPI (Async First)", body_style), Paragraph("Asynchronous REST API, CORS middleware, dependency injection (port 8000).", body_style)],
        [Paragraph("<b>Agent Framework</b>", body_style), Paragraph("LangGraph + LangChain", body_style), Paragraph("State machine orchestrating Planner, Retrieval, MCP Tool, Response, and Citation nodes.", body_style)],
        [Paragraph("<b>Vector Search</b>", body_style), Paragraph("Azure AI Search (Free Tier)", body_style), Paragraph("Hybrid dense vector search (HNSW) combined with BM25 keyword matching.", body_style)],
        [Paragraph("<b>Embeddings Engine</b>", body_style), Paragraph("FastEmbed (ONNX)", body_style), Paragraph("BAAI/bge-small-en-v1.5 running locally on CPU (384-dimensional dense vectors).", body_style)],
        [Paragraph("<b>Tool Integration</b>", body_style), Paragraph("MCP (Model Context Protocol)", body_style), Paragraph("Streamable HTTP client connecting to hosted Tavily MCP for live web news & search.", body_style)],
        [Paragraph("<b>Session Memory</b>", body_style), Paragraph("Azure Cosmos DB (Free Tier)", body_style), Paragraph("Document store partitioned by /userId for sub-5ms history lookups.", body_style)],
        [Paragraph("<b>Foundation Models</b>", body_style), Paragraph("Azure AI Foundry", body_style), Paragraph("Llama-3.3-70B for intent planning & Phi-4 for grounded response generation.", body_style)],
        [Paragraph("<b>Container Hosting</b>", body_style), Paragraph("Azure Container Apps", body_style), Paragraph("Serverless Kubernetes with KEDA autoscaling, scale-to-zero, and automated HTTPS.", body_style)],
    ]

    t_arch = Table(arch_data, colWidths=[1.4 * inch, 1.6 * inch, 4.0 * inch])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('TEXTCOLOR', (0, 0), (-1, -1), c_body),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, c_box_border),
    ]))
    story.append(t_arch)

    story.append(Spacer(1, 12))

    # 3. LangGraph Workflow
    story.append(Paragraph("3. LangGraph Agentic Workflow & Reasoning Pipeline", h1_style))
    story.append(Paragraph(
        "The reasoning process is modeled as an asynchronous <b>StateGraph</b> with dynamic conditional routing:",
        body_style
    ))
    story.append(Paragraph("<b>1. Planner Agent:</b> Analyzes the user question and the last 6 turns of conversation history. Produces a strict structured plan: <code>{use_documents: bool, use_web: bool}</code>.", bullet_style))
    story.append(Paragraph("<b>2. Retrieval Agent:</b> Invoked when <code>use_documents == True</code>. Embeds query into 384-dim vector and runs hybrid BM25 + vector search against Azure AI Search.", bullet_style))
    story.append(Paragraph("<b>3. MCP Tool Agent:</b> Invoked when <code>use_web == True</code>. Connects to Tavily Search via MCP. Detects topic categories ('news' vs 'general') and extracts high-relevance snippets.", bullet_style))
    story.append(Paragraph("<b>4. Response Synthesis Agent:</b> Compiles all retrieved sources into a numbered reference block (<code>[1]</code>, <code>[2]</code>, <code>[3]</code>) and instructs the model to answer strictly using the provided context.", bullet_style))
    story.append(Paragraph("<b>5. Citation & Guardrails Agent:</b> Validates all <code>[n]</code> brackets in the generated text, strips dangling hallucinations, and filters the metadata citations list to match only sources cited in the final answer.", bullet_style))

    story.append(Spacer(1, 10))

    # 4. Azure Container Apps Deep Dive
    story.append(Paragraph("4. How Azure Container Apps (ACA) Works & Scales", h1_style))
    story.append(Paragraph(
        "<b>Azure Container Apps</b> is a serverless platform built on top of managed Kubernetes (AKS), Envoy Proxy, and KEDA (Kubernetes Event-driven Autoscaler).",
        body_style
    ))
    story.append(Paragraph("<b>Key Architectural Capabilities:</b>", h2_style))
    story.append(Paragraph("• <b>Scale to Zero (minReplicas: 0):</b> When no requests arrive, ACA shuts down all running replicas. The application consumes 0 CPU and 0 RAM, costing exactly <b>$0.00</b> while idle.", bullet_style))
    story.append(Paragraph("• <b>KEDA Dynamic Scaling:</b> When traffic spikes (e.g. 50 concurrent requests per container), KEDA automatically spins up new replicas in seconds up to <code>maxReplicas</code>.", bullet_style))
    story.append(Paragraph("• <b>Envoy Ingress & Free HTTPS:</b> Automatic SSL certificate provisioning, internal/external ingress routing, and health checks on <code>/health</code> and <code>/_stcore/health</code>.", bullet_style))
    story.append(Paragraph("• <b>Microservice Decoupling:</b> The Streamlit frontend and FastAPI backend scale independently according to their specific load profiles.", bullet_style))

    story.append(Spacer(1, 10))

    # 5. Production Troubleshooting & Key Resolutions
    story.append(Paragraph("5. Key Production Issues & Engineering Solutions", h1_style))

    issues_data = [
        [Paragraph("<b>Issue Diagnosed</b>", body_style), Paragraph("<b>Root Cause</b>", body_style), Paragraph("<b>Production Resolution Applied</b>", body_style)],
        [
            Paragraph("Cosmos DB 401 on Swagger UI", body_style),
            Paragraph("Default Swagger template sent dummy conversation_id = 'string'.", body_style),
            Paragraph("Updated ChatService to defensively treat non-UUID / empty IDs as fresh conversations.", body_style)
        ],
        [
            Paragraph("Tavily Returning Irrelevant Code", body_style),
            Paragraph("Default basic search depth without topic routing matched shallow git repos.", body_style),
            Paragraph("Added dynamic keyword routing in tavily.py: queries with 'news/market' use topic='news' & depth='advanced'.", body_style)
        ],
        [
            Paragraph("Docker Permission Denied on Cache", body_style),
            Paragraph("Non-root container user (appuser) lacked write access to /app/.cache.", body_style),
            Paragraph("Configured HF_HOME and FASTEMBED_CACHE_PATH to /tmp/cache with full chown permissions in Dockerfile.", body_style)
        ],
        [
            Paragraph("Foundry 404 Endpoint Duplication", body_style),
            Paragraph("OpenAI client SDK appends /chat/completions to base endpoint url.", body_style),
            Paragraph("Corrected AZURE_AI_FOUNDRY_CHAT_ENDPOINT to https://<resource>.services.ai.azure.com/models.", body_style)
        ],
    ]

    t_issues = Table(issues_data, colWidths=[1.8 * inch, 2.2 * inch, 3.0 * inch])
    t_issues.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('TEXTCOLOR', (0, 0), (-1, -1), c_body),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, c_box_border),
    ]))
    story.append(t_issues)

    story.append(Spacer(1, 12))

    # 6. Live Production URLs
    story.append(Paragraph("6. Live Production Deployment Details", h1_style))
    story.append(Paragraph("<b>Streamlit Frontend App:</b> <font color='#1d4ed8'><u>https://aca-rag-frontend.salmonriver-feff8fb1.eastus.azurecontainerapps.io/</u></font>", body_style))
    story.append(Paragraph("<b>FastAPI Swagger Docs:</b> <font color='#1d4ed8'><u>https://aca-rag-backend.salmonriver-feff8fb1.eastus.azurecontainerapps.io/docs</u></font>", body_style))
    story.append(Paragraph("<b>Backend Health API:</b> <font color='#1d4ed8'><u>https://aca-rag-backend.salmonriver-feff8fb1.eastus.azurecontainerapps.io/health</u></font>", body_style))
    story.append(Paragraph("<b>GitHub Repository:</b> <font color='#1d4ed8'><u>https://github.com/tadevilas/End-to-End-AgenticBot-Azure-</u></font>", body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully built: {filename}")


if __name__ == "__main__":
    build_pdf()
