"""Design tokens and CSS generation, sourced verbatim from DESIGN.md.

Every value here maps directly to a token in the DESIGN.md front matter:
the monochrome core, the pastel color-block palette, the figmaSans/figmaMono
type roles (substituted with Inter / JetBrains Mono per DESIGN.md's guidance),
the 8px spacing scale, the border-radius scale, and the pill/circle button
shapes. Nothing about the visual system is invented — the CSS below is the
faithful implementation of the documented design.
"""

from __future__ import annotations

# --- Colors (DESIGN.md > colors) ------------------------------------------
COLORS = {
    "primary": "#000000",
    "on-primary": "#ffffff",
    "ink": "#000000",
    "canvas": "#ffffff",
    "inverse-canvas": "#000000",
    "inverse-ink": "#ffffff",
    "on-inverse-soft": "#ffffff",
    "hairline": "#e6e6e6",
    "hairline-soft": "#f1f1f1",
    "surface-soft": "#f7f7f5",
    "block-lime": "#dceeb1",
    "block-lilac": "#c5b0f4",
    "block-cream": "#f4ecd6",
    "block-pink": "#efd4d4",
    "block-mint": "#c8e6cd",
    "block-coral": "#f3c9b6",
    "block-navy": "#1f1d3d",
    "accent-magenta": "#ff3d8b",
    "semantic-success": "#1ea64a",
}

# --- Spacing (DESIGN.md > spacing), 8px base ------------------------------
SPACING = {
    "hair": "1px",
    "xxs": "4px",
    "xs": "8px",
    "sm": "12px",
    "md": "16px",
    "lg": "24px",
    "xl": "32px",
    "xxl": "48px",
    "section": "96px",
}

# --- Border radius (DESIGN.md > rounded) ----------------------------------
ROUNDED = {
    "xs": "2px",
    "sm": "6px",
    "md": "8px",
    "lg": "24px",
    "xl": "32px",
    "pill": "50px",
    "full": "9999px",
}

# Font substitutes documented in DESIGN.md ("Note on Font Substitutes").
FONT_SANS = "'Inter', 'SF Pro Display', system-ui, helvetica, sans-serif"
FONT_MONO = "'JetBrains Mono', 'SF Mono', menlo, monospace"


def _css_variables() -> str:
    """Emit design tokens as CSS custom properties for reuse across rules."""
    lines: list[str] = []
    for key, value in COLORS.items():
        lines.append(f"  --color-{key}: {value};")
    for key, value in SPACING.items():
        lines.append(f"  --space-{key}: {value};")
    for key, value in ROUNDED.items():
        lines.append(f"  --radius-{key}: {value};")
    return "\n".join(lines)


def build_css() -> str:
    """Return the full stylesheet implementing the DESIGN.md system."""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300..700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
{_css_variables()}
  --max-content: 1280px;
}}

/* ---- Canvas: pure white surface, black ink (DESIGN.md core) ---- */
html, body, [data-testid="stAppViewContainer"], .stApp {{
  background-color: var(--color-canvas);
  color: var(--color-ink);
  font-family: {FONT_SANS};
  font-feature-settings: "kern";
}}

/* Constrain content width to ~1280px with responsive gutters. */
.block-container {{
  max-width: var(--max-content);
  padding-top: var(--space-lg);
  padding-bottom: var(--space-section);
  padding-left: var(--space-xxl);
  padding-right: var(--space-xxl);
}}
@media (max-width: 768px) {{
  .block-container {{ padding-left: var(--space-lg); padding-right: var(--space-lg); }}
}}

/* Hide default Streamlit chrome so the editorial frame reads clean. */
#MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}

/* ---- Typography roles (DESIGN.md > typography) ---- */
.display-xl {{ font-size: 86px; font-weight: 340; line-height: 1.00; letter-spacing: -1.72px; }}
.display-lg {{ font-size: 64px; font-weight: 340; line-height: 1.10; letter-spacing: -0.96px; }}
.headline   {{ font-size: 26px; font-weight: 540; line-height: 1.35; letter-spacing: -0.26px; }}
.subhead    {{ font-size: 26px; font-weight: 340; line-height: 1.35; letter-spacing: -0.26px; }}
.card-title {{ font-size: 24px; font-weight: 700; line-height: 1.45; letter-spacing: 0; }}
.body-lg    {{ font-size: 20px; font-weight: 330; line-height: 1.40; letter-spacing: -0.14px; }}
.body       {{ font-size: 18px; font-weight: 320; line-height: 1.45; letter-spacing: -0.26px; }}
.body-sm    {{ font-size: 16px; font-weight: 330; line-height: 1.45; letter-spacing: -0.14px; }}
.eyebrow {{
  font-family: {FONT_MONO}; font-size: 18px; font-weight: 400; line-height: 1.30;
  letter-spacing: 0.54px; text-transform: uppercase;
}}
.caption {{
  font-family: {FONT_MONO}; font-size: 12px; font-weight: 400; line-height: 1.00;
  letter-spacing: 0.60px; text-transform: uppercase;
}}
@media (max-width: 560px) {{
  .display-xl {{ font-size: 48px; letter-spacing: -0.96px; }}
  .display-lg {{ font-size: 40px; }}
}}

/* ---- Marquee strip: thin black ribbon under the nav ---- */
.marquee-strip {{
  background: var(--color-inverse-canvas); color: var(--color-inverse-ink);
  height: 36px; display: flex; align-items: center; justify-content: center;
  font-family: {FONT_MONO}; font-size: 12px; letter-spacing: 0.60px;
  text-transform: uppercase; border-radius: var(--radius-xs);
  margin-bottom: var(--space-xl); overflow: hidden;
}}

/* ---- Color-block sections (signature): rounded-lg, 48px padding ---- */
.color-block {{
  border-radius: var(--radius-lg); padding: var(--space-xxl);
  margin: var(--space-xl) 0;
}}
.block-lime  {{ background: var(--color-block-lime);  color: var(--color-ink); }}
.block-lilac {{ background: var(--color-block-lilac); color: var(--color-ink); }}
.block-cream {{ background: var(--color-block-cream); color: var(--color-ink); }}
.block-mint  {{ background: var(--color-block-mint);  color: var(--color-ink); }}
.block-pink  {{ background: var(--color-block-pink);  color: var(--color-ink); }}
.block-coral {{ background: var(--color-block-coral); color: var(--color-ink); }}
.block-navy  {{ background: var(--color-block-navy);  color: var(--color-inverse-ink); }}
@media (max-width: 768px) {{
  .color-block {{ border-radius: 0; margin-left: calc(-1 * var(--space-lg)); margin-right: calc(-1 * var(--space-lg)); }}
}}

/* ---- Cards: hairline border on white, rounded-lg (no shadow) ---- */
.rc-card {{
  background: var(--color-canvas); border: 1px solid var(--color-hairline);
  border-radius: var(--radius-lg); padding: var(--space-lg);
}}
.template-card {{
  background: var(--color-surface-soft); border-radius: var(--radius-md);
  padding: var(--space-md);
}}

/* ---- Buttons: every CTA is a pill (DESIGN.md Do's) ---- */
.stButton > button, .stDownloadButton > button {{
  background: var(--color-primary); color: var(--color-on-primary);
  border: none; border-radius: var(--radius-pill);
  font-family: {FONT_SANS}; font-size: 18px; font-weight: 480;
  letter-spacing: -0.10px; padding: 10px 24px; min-height: 44px;
  transition: transform 0.08s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  transform: scale(1.02); background: var(--color-primary); color: var(--color-on-primary);
}}
.stButton > button:active {{ transform: scale(0.98); }}
/* Secondary pill: white with black ink, used via 'secondary' type. */
.stButton > button[kind="secondary"] {{
  background: var(--color-canvas); color: var(--color-ink);
  border: 1px solid var(--color-hairline);
}}

/* ---- Inputs: hairline border, rounded-md, ring focus (no fill change) ---- */
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {{
  border-radius: var(--radius-md) !important; border: 1px solid var(--color-hairline) !important;
  background: var(--color-canvas) !important; color: var(--color-ink) !important;
  font-family: {FONT_SANS} !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
  outline: 2px solid var(--color-primary) !important; outline-offset: 1px;
}}

/* ---- Chat message bubbles ---- */
[data-testid="stChatMessage"] {{
  background: var(--color-surface-soft); border-radius: var(--radius-lg);
  border: 1px solid var(--color-hairline-soft);
}}

/* ---- Citation chip ---- */
.citation-chip {{
  display: inline-block; background: var(--color-surface-soft);
  border: 1px solid var(--color-hairline); border-radius: var(--radius-sm);
  padding: 2px 8px; font-family: {FONT_MONO}; font-size: 12px;
  letter-spacing: 0.4px; margin: 2px 4px 2px 0;
}}

/* ---- Success glyph (comparison checkmark color role) ---- */
.success {{ color: var(--color-semantic-success); }}

/* ---- Metric tiles for the dashboard ---- */
.metric-tile {{
  background: var(--color-surface-soft); border-radius: var(--radius-md);
  padding: var(--space-lg);
}}
.metric-value {{ font-size: 40px; font-weight: 540; line-height: 1.0; letter-spacing: -0.5px; }}

/* Sidebar nav on white canvas. */
[data-testid="stSidebar"] {{
  background: var(--color-canvas); border-right: 1px solid var(--color-hairline);
}}

/* Footer wordmark row. */
.rc-footer {{
  border-top: 1px solid var(--color-hairline-soft); margin-top: var(--space-section);
  padding: var(--space-xl) 0; color: var(--color-ink);
}}
</style>
"""
