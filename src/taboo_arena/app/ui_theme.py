"""Shared Streamlit theme for the dashboard."""

from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    """Inject the app theme CSS."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Spectral:wght@600;700&display=swap');
        :root {
          --bg-top: #f6f1df;
          --bg-bottom: #efe1b7;
          --panel: rgba(255, 249, 235, 0.92);
          --ink: #18281d;
          --muted: #5e6d59;
          --accent: #126b52;
          --accent-soft: #dbeee6;
          --warn: #914d14;
          --warn-soft: #fff0dd;
        }
        .stApp {
          background:
            radial-gradient(circle at top left, rgba(18,107,82,0.12), transparent 32%),
            radial-gradient(circle at top right, rgba(190,120,42,0.15), transparent 28%),
            linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
          color: var(--ink);
          font-family: "Space Grotesk", "Aptos", sans-serif;
        }
        .block-container {
          padding-top: 0.2rem;
          padding-bottom: 0.45rem;
          padding-left: 0.9rem;
          padding-right: 0.9rem;
          max-width: 100%;
        }
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        .stAppToolbar,
        #MainMenu,
        footer {
          display: none !important;
        }
        h1, h2, h3, .taboo-card-title {
          font-family: "Spectral", Georgia, serif;
          color: var(--ink);
        }
        .taboo-card, .taboo-panel {
          background: var(--panel);
          border: 1px solid rgba(24, 40, 29, 0.1);
          border-radius: 18px;
          padding: 1rem 1.1rem;
          box-shadow: 0 10px 28px rgba(39, 47, 36, 0.08);
        }
        .soft-panel {
          background: rgba(255, 249, 235, 0.78);
          border: 1px solid rgba(24, 40, 29, 0.08);
          border-radius: 20px;
          padding: 0.95rem 1rem;
          box-shadow: 0 8px 20px rgba(39, 47, 36, 0.05);
        }
        .taboo-card {
          background:
            linear-gradient(180deg, rgba(255,255,255,0.78), rgba(248,241,224,0.92));
        }
        .result-banner {
          padding: 0.85rem 1rem;
          border-radius: 16px;
          margin-bottom: 0.85rem;
          font-weight: 700;
        }
        .result-success {
          background: var(--accent-soft);
          color: var(--accent);
        }
        .result-fail {
          background: var(--warn-soft);
          color: var(--warn);
        }
        .metric-label {
          color: var(--muted);
          font-size: 0.88rem;
        }
        .metric-value {
          font-size: 1.15rem;
          font-weight: 700;
        }
        .hero-strip {
          background: rgba(255, 249, 235, 0.9);
          border: 1px solid rgba(24, 40, 29, 0.1);
          border-radius: 22px;
          padding: 0.95rem 1.1rem;
          min-height: 104px;
          display: flex;
          flex-direction: column;
          justify-content: center;
          box-shadow: 0 10px 28px rgba(39, 47, 36, 0.08);
        }
        .hero-title {
          font-family: "Spectral", Georgia, serif;
          font-size: 2.25rem;
          line-height: 1;
          font-weight: 700;
          margin: 0;
        }
        .hero-subtitle {
          color: var(--muted);
          font-size: 0.9rem;
          margin-top: 0.28rem;
        }
        .resource-card {
          background: rgba(255, 249, 235, 0.9);
          border: 1px solid rgba(24, 40, 29, 0.1);
          border-radius: 20px;
          padding: 0.75rem 0.9rem;
          height: 122px;
          box-shadow: 0 8px 18px rgba(39, 47, 36, 0.06);
        }
        .resource-body {
          display: flex;
          align-items: center;
          height: 100%;
          gap: 0.55rem;
        }
        .resource-spark {
          flex: 1 1 auto;
          min-width: 0;
        }
        .resource-spark svg {
          width: 100%;
          height: 52px;
          display: block;
        }
        .resource-values {
          display: grid;
          gap: 0.15rem;
          align-content: center;
          justify-items: end;
          flex: 0 0 78px;
        }
        .resource-values span {
          font-size: 0.86rem;
          color: var(--muted);
        }
        .resource-values strong {
          font-size: 1rem;
          color: var(--ink);
        }
        .role-head {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.7rem;
          margin-bottom: 0.55rem;
        }
        .role-meta {
          display: grid;
          gap: 0.2rem;
        }
        .role-name {
          font-size: 1.1rem;
          font-weight: 700;
          line-height: 1.05;
        }
        .role-state {
          display: inline-flex;
          width: fit-content;
          padding: 0.2rem 0.48rem;
          border-radius: 999px;
          font-size: 0.72rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          background: rgba(24, 40, 29, 0.08);
          color: var(--muted);
        }
        .role-avatar {
          width: 58px;
          height: 58px;
          border-radius: 18px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 1.9rem;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.55);
        }
        .avatar-cluer {
          background: linear-gradient(180deg, #f7d8d1, #efb3a4);
          color: #8b3d29;
        }
        .avatar-guesser {
          background: linear-gradient(180deg, #e4edff, #bfd2ff);
          color: #2a4e98;
        }
        .avatar-judge {
          background: linear-gradient(180deg, #eee4ff, #d9c8ff);
          color: #5d3d9d;
        }
        .avatar-sleep {
          opacity: 0.55;
          filter: saturate(0.7);
        }
        .avatar-thinking {
          animation: pulsefloat 1.15s ease-in-out infinite;
        }
        @keyframes pulsefloat {
          0% { transform: translateY(0px); }
          50% { transform: translateY(-2px) scale(1.03); }
          100% { transform: translateY(0px); }
        }
        .role-note {
          font-size: 0.84rem;
          color: var(--muted);
          margin-top: 0.1rem;
        }
        .game-card-shell {
          max-width: 320px;
          min-height: 420px;
          margin: 0 auto 0.7rem;
          padding: 0.8rem;
          background: linear-gradient(180deg, #fff5ea, #fde8bc);
          border: 4px solid #cf5d46;
          border-radius: 28px;
          box-shadow: 0 12px 30px rgba(39, 47, 36, 0.1);
          display: flex;
          flex-direction: column;
          gap: 0.8rem;
        }
        .game-card-top {
          border-radius: 22px;
          padding: 1rem 0.95rem 1.1rem;
          background: linear-gradient(180deg, #d86a54, #bd4632);
          text-align: center;
        }
        .game-card-title {
          font-size: 0.74rem;
          text-transform: uppercase;
          letter-spacing: 0.22em;
          color: rgba(255, 250, 240, 0.86);
          margin-bottom: 0.5rem;
        }
        .game-card-target {
          font-family: "Spectral", Georgia, serif;
          font-size: 2rem;
          line-height: 1.05;
          color: #fffdf7;
          margin-bottom: 0.2rem;
        }
        .game-card-kind {
          color: rgba(255, 250, 240, 0.78);
          font-size: 0.84rem;
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }
        .game-card-taboo-stack {
          display: flex;
          flex-direction: column;
          gap: 0.55rem;
          flex: 1;
        }
        .game-card-taboo {
          border-radius: 16px;
          padding: 0.7rem 0.65rem;
          background: rgba(255, 252, 246, 0.92);
          border: 1px solid rgba(145, 77, 20, 0.12);
          color: #7e4617;
          font-weight: 700;
          font-size: 1rem;
          text-align: center;
        }
        .game-card-footer {
          margin-top: auto;
          padding-top: 0.35rem;
          color: var(--muted);
          font-size: 0.82rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          text-align: center;
        }
        .st-key-random_card_icon button,
        .st-key-open_app_settings button,
        .st-key-cluer_settings_button button,
        .st-key-guesser_settings_button button,
        .st-key-judge_settings_button button {
          min-height: 2.35rem;
          height: 2.35rem;
          padding: 0 0.35rem;
          border-radius: 999px;
          font-size: 1.05rem;
          font-weight: 700;
        }
        .st-key-random_card_icon button {
          background: rgba(207, 93, 70, 0.12);
          color: #9a3f2c;
          border: 1px solid rgba(207, 93, 70, 0.22);
          min-height: 3.4rem;
          height: 3.4rem;
          width: 100%;
          border-radius: 20px;
        }
        .st-key-open_app_settings button,
        .st-key-cluer_settings_button button,
        .st-key-guesser_settings_button button,
        .st-key-judge_settings_button button {
          background: rgba(24, 40, 29, 0.06);
          color: var(--ink);
          border: 1px solid rgba(24, 40, 29, 0.12);
        }
        .section-heading {
          font-size: 0.82rem;
          text-transform: uppercase;
          letter-spacing: 0.16em;
          color: var(--muted);
          margin-bottom: 0.5rem;
        }
        .pulse-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.5rem;
          margin-bottom: 0.55rem;
        }
        .metrics-group {
          margin-bottom: 0.7rem;
        }
        .metrics-group-title {
          color: var(--muted);
          font-size: 0.72rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          margin-bottom: 0.32rem;
        }
        .pulse-card {
          border-radius: 16px;
          padding: 0.55rem 0.65rem;
          background: rgba(255, 249, 235, 0.86);
          border: 1px solid rgba(24, 40, 29, 0.08);
        }
        .pulse-card-label {
          color: var(--muted);
          font-size: 0.68rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          margin-bottom: 0.14rem;
        }
        .pulse-card-value {
          color: var(--ink);
          font-size: 1.02rem;
          font-weight: 700;
          line-height: 1.05;
        }
        .subtle-note {
          color: var(--muted);
          font-size: 0.84rem;
        }
        .fit-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.65rem;
          margin-bottom: 0.75rem;
        }
        .fit-card {
          border-radius: 16px;
          padding: 0.7rem 0.8rem;
          background: rgba(255, 249, 235, 0.88);
          border: 1px solid rgba(24, 40, 29, 0.1);
        }
        .fit-card-label {
          color: var(--muted);
          font-size: 0.76rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          margin-bottom: 0.28rem;
        }
        .fit-card-value {
          color: var(--ink);
          font-size: 1.25rem;
          font-weight: 700;
          line-height: 1.05;
        }
        .compact-hint {
          color: var(--muted);
          font-size: 0.83rem;
        }
        .transcript-wrap {
          display: flex;
          flex-direction: column;
          gap: 0.42rem;
          padding-top: 0.25rem;
        }
        .transcript-row {
          display: flex;
          width: 100%;
        }
        .transcript-row.left {
          justify-content: flex-start;
        }
        .transcript-row.center {
          justify-content: center;
        }
        .transcript-row.right {
          justify-content: flex-end;
        }
        .transcript-bubble {
          border-radius: 18px;
          padding: 0.8rem 0.95rem;
          border: 1px solid rgba(24, 40, 29, 0.08);
          box-shadow: 0 8px 18px rgba(39, 47, 36, 0.06);
          line-height: 1.45;
          white-space: pre-wrap;
        }
        .transcript-meta {
          width: auto !important;
          max-width: 100%;
          padding: 0.15rem 0.35rem;
          border: none;
          box-shadow: none;
          background: transparent;
          color: var(--muted);
          font-size: 0.88rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }
        .transcript-pending {
          background: rgba(215, 213, 206, 0.92);
          color: #38443a;
        }
        .transcript-rejected {
          background: #f4d6d1;
          color: #7d281c;
        }
        .transcript-accepted {
          background: #f5e39c;
          color: #5d4707;
        }
        .transcript-judge {
          background: #e7dcff;
          color: #523090;
        }
        .transcript-guess {
          background: #d9e7ff;
          color: #204b8c;
        }
        .transcript-success {
          background: #d8efcd;
          color: #25603d;
        }
        .transcript-label {
          display: block;
          margin-bottom: 0.28rem;
          font-size: 0.8rem;
          font-weight: 700;
          letter-spacing: 0.02em;
          opacity: 0.72;
        }
        .transcript-inline-bubble {
          width: 57.1429%;
          max-width: 57.1429%;
        }
        .prompt-modal-pre {
          margin: 0;
          padding: 0.9rem 1rem;
          border-radius: 16px;
          background: rgba(250, 248, 243, 0.95);
          border: 1px solid rgba(24, 40, 29, 0.12);
          color: var(--ink);
          font-family: "Cascadia Code", "Consolas", monospace;
          font-size: 0.82rem;
          line-height: 1.45;
          white-space: pre-wrap;
          word-break: break-word;
          overflow-wrap: anywhere;
        }
        @media (max-width: 1100px) {
          .pulse-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }
          .transcript-inline-bubble {
            width: min(100%, 92%);
            max-width: min(100%, 92%);
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
