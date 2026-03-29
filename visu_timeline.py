import os
import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models.formatters import DatetimeTickFormatter

from visu_front import MACRO_COLORS

OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def visualisation_timeline(df_macro):
    output_file(f"{OUTPUT_DIR}/timeline.html", title="Évolution temporelle")

    # Filter out noise cluster
    df = df_macro[df_macro["macro_sujet"] != "Bruit"].copy()
    df["date"] = pd.to_datetime(df["date"])

    df["semaine"] = df["date"].dt.to_period("W").dt.start_time
    grouped = (
        df.groupby(["semaine", "macro_sujet"])
        .size()
        .reset_index(name="nb_articles")
    )

    cluster_noms = sorted(grouped["macro_sujet"].unique())

    TOP_N = 8
    totaux = grouped.groupby("macro_sujet")["nb_articles"].sum().sort_values(ascending=False)
    cluster_noms = [n for n in totaux.index[:TOP_N]]

    couleur_macro = {
        nom: MACRO_COLORS[i % len(MACRO_COLORS)]
        for i, nom in enumerate(cluster_noms)
    }

    y_max = grouped["nb_articles"].max()

    p = figure(
        x_axis_type="datetime",
        y_range=(0, y_max * 1.1),
        tools="pan,wheel_zoom,reset",
        title="Évolution temporelle des clusters principaux",
        sizing_mode="stretch_both",
        toolbar_location="above",
    )
    p.background_fill_color  = "#0f0f1a"
    p.border_fill_color      = "#0f0f1a"
    p.outline_line_color     = None
    p.title.text_color       = "#e0e0e0"
    p.title.text_font_size   = "14px"
    p.xaxis.major_label_text_color = "#aaa"
    p.yaxis.major_label_text_color = "#aaa"
    p.xaxis.axis_line_color  = "#444"
    p.yaxis.axis_line_color  = "#444"
    p.xaxis.major_tick_line_color = "#444"
    p.yaxis.major_tick_line_color = "#444"
    p.grid.grid_line_color   = "#222"
    p.xaxis.formatter = DatetimeTickFormatter(
        months="%b %Y",
        days="%d %b",
    )

    for nom in cluster_noms:
        data = grouped[grouped["macro_sujet"] == nom].sort_values("semaine")
        if data.empty:
            continue
        coul = couleur_macro[nom]
        src = ColumnDataSource(dict(
            x=data["semaine"].tolist(),
            y=data["nb_articles"].tolist(),
            nom=[nom] * len(data),
        ))
        renderer = p.line(
            x="x", y="y", source=src,
            line_color=coul, line_width=2.5,
            legend_label=nom,
        )
        p.add_tools(HoverTool(
            renderers=[renderer],
            tooltips=f"""
                <div style="font-family:monospace; padding:5px 9px;
                            background:#1a1a2e; border:1px solid #444; border-radius:5px;
                            white-space:nowrap">
                    <span style="color:{coul}; font-weight:bold">{nom}</span><br>
                    <span style="color:#aaa; font-size:11px">@y articles — @x{{%b %Y}}</span>
                </div>
            """,
            formatters={"@x": "datetime"},
            mode="mouse",
        ))

    p.legend.label_text_color       = "#ddd"
    p.legend.background_fill_color  = "#0f0f1a"
    p.legend.background_fill_alpha  = 0.8
    p.legend.border_line_color      = None
    p.legend.inactive_fill_color    = "#0f0f1a"
    p.legend.inactive_fill_alpha    = 0.8
    p.legend.click_policy           = "hide"

    save(p)
    print(f"---- timeline.html sauvegardé dans {OUTPUT_DIR}/")
