"""Shared chart styling for AI_Recovery figures -- single source of truth
for the fixed color palette / line-style conventions the user specified
(see project memory feedback_ai_recovery_graph_style.md, and
AI_RECOVERY_EXECUTION_PLAN.md §12 Goal 4 Category A). Every figure-
generating script imports from here instead of redefining colors locally.
"""

# Fixed color ROLES -- not a generic cycled categorical palette. Each name
# is what the color means, not just a hue; keep the mapping stable across
# figures so a reader learns the palette once.
COLOR_REFERENCE      = '#00FFFF'   # cyan    -- reference/supposed path (or a 2nd predicted variant)
COLOR_PREDICTED      = '#FF00FF'   # magenta -- reconstructed / predicted path
COLOR_GROUND_TRUTH   = '#000000'   # black   -- ground truth / executed trajectory
COLOR_EKF            = '#708090'   # slate gray -- classical filter baseline
COLOR_PURE_INERTIAL  = '#CD7F63'   # muted coral -- naive/no-correction baseline

# Extra accents -- proven matplotlib tab10 shades, not neon. Use sparingly
# (one or two per chart), only when the 5 roles above aren't enough series.
COLOR_ACCENT_ORANGE  = '#FF7F0E'
COLOR_ACCENT_RED     = '#D62728'
COLOR_ACCENT_GREEN   = '#2CA02C'
COLOR_ACCENT_PURPLE  = '#9467BD'
COLOR_ACCENT_BROWN   = '#8C564B'
COLOR_ACCENT_BLUE    = '#1F77B4'

GRID_KW = dict(color='gray', alpha=0.3, linewidth=0.5)

# Fixed categorical order for the ablation matrix (Track A, Phase A5) --
# one color per model family, assigned in a fixed order and never cycled
# (dataviz skill convention), reusing the trajectory-role colors above where
# a model IS one of those roles (EKF, pure-inertial) rather than picking a
# second, redundant color for the same entity.
ABLATION_MODEL_COLORS = {
    'LSTM':            COLOR_ACCENT_BLUE,
    'SSL-LSTM':         COLOR_ACCENT_PURPLE,
    'Transformer':      COLOR_ACCENT_GREEN,
    'SSL-Transformer':  COLOR_ACCENT_BROWN,
    'RandomForest':     COLOR_ACCENT_ORANGE,
    'GBT':              COLOR_ACCENT_RED,
    'EKF':              COLOR_EKF,
    'Pure Inertial':    COLOR_PURE_INERTIAL,
}


def style_axes(ax, title=None, xlabel=None, ylabel=None, zlabel=None):
    """Applies the shared title/label/grid/spine conventions to a 2D or 3D
    matplotlib Axes."""
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if zlabel and hasattr(ax, 'set_zlabel'):
        ax.set_zlabel(zlabel, fontsize=10)
    ax.grid(True, **GRID_KW)
    if hasattr(ax, 'spines'):
        for side in ('top', 'right'):
            if side in ax.spines:
                ax.spines[side].set_visible(False)
