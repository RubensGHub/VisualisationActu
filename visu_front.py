import os
import json
import math
import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Div
from bokeh.layouts import row
from bokeh.palettes import Turbo256


OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualisation_chart(df_bertopic, resume_bertopic):
    output_file(f"{OUTPUT_DIR}/visualisation.html", title="Clustering des articles")
    

    print("df_bertopic : \n\n")
    print(df_bertopic)
    print("------------------------------------")
    print("resume_bertopic : \n\n")
    print(resume_bertopic)

    # rajout des titres des articles dans résu
    id_to_titre = df_bertopic.set_index("id_article")["titre"].to_dict()
    resume_bertopic["liste_titres"] = resume_bertopic["liste_ids_articles"].apply(
        lambda ids: [id_to_titre[i] for i in ids if i in id_to_titre]
    )


    # Taille des bulles :  sqrt(nb articles)
    df_vis  = resume_bertopic[resume_bertopic["id_sujet"] != -1].reset_index(drop=True)
    n = len(df_vis)
    max_articles = df_vis["nombre_articles"].max()
    R_MAX = 80   # rayon max en coordonnées data
    SCALE = 300  # rayon du cercle de disposition
    color_palette = [Turbo256[int(i * 255 / max(n-1, 1))] for i in range(n)]

    # Bulles :
    xs, ys, rs, noms, nbs, couleurs, titres_json = [], [], [], [], [], [], []
    for i, (_, s) in enumerate(df_vis.iterrows()):
        angle = 2 * math.pi * i / n
        xs.append(SCALE * math.cos(angle))
        ys.append(SCALE * math.sin(angle))
        rs.append(R_MAX * math.sqrt(s["nombre_articles"] / max_articles))

        # Mise au propre des noms des sujets :
        parts = s["nom_sujet"].split("_")
        noms.append(" / ".join(parts[1:4])[:30] if len(parts) > 1 else s["nom_sujet"][:30])
        nbs.append(int(s["nombre_articles"]))
        couleurs.append(color_palette[i])
 
        titres = s["liste_titres"] if isinstance(s["liste_titres"], list) else []
        titres_json.append(json.dumps(titres, ensure_ascii=False))
 
    source = ColumnDataSource(dict(
        x=xs, y=ys, r=rs, nom=noms,
        nb_articles=nbs, couleur=couleurs, titres_json=titres_json,
    ))
 
    source = ColumnDataSource(dict(
        x=xs, y=ys, r=rs, nom=noms,
        nb_articles=nbs, couleur=couleurs, titres_json=titres_json,
    ))

    # Formes des builles : 
    lim = SCALE + R_MAX + 30
    p = figure(
        tools="pan,wheel_zoom,reset,tap",
        x_range=(-lim, lim),
        y_range=(-lim, lim),
        match_aspect=True,
        toolbar_location="above",
        title="Clusters d'articles sur l'actualité de la transition énergétique",
        sizing_mode="stretch_both",
    )
    p.axis.visible = p.grid.visible = False
    p.background_fill_color = "#0f0f1a"
    p.border_fill_color     = "#0f0f1a"
    p.outline_line_color    = None
    p.title.text_color      = "#e0e0e0"
 
    renderer = p.circle(
        x="x", y="y", radius="r",
        fill_color="couleur", fill_alpha=0.75,
        line_color="white", line_alpha=0.3, line_width=0.8,
        hover_fill_alpha=1.0,
        nonselection_fill_alpha=0.3,
        selection_line_color="white", selection_line_width=2,
        source=source,
    )
 
    p.text(
        x="x", y="y", text="nom",
        text_color="white", text_font_size="20px",
        text_align="center", text_baseline="middle",
        source=source,
    )
 
    p.add_tools(HoverTool(renderers=[renderer], tooltips=[
        ("Cluster", "@nom"), ("Articles", "@nb_articles"),
    ]))




    # - - - - HTML : -  - - - 
    div = Div(
        text="<p style='color:#777;font-style:italic;padding:16px;font-family:monospace'>"
             "Cliquez sur une bulle</p>",
        width=320,
        sizing_mode="fixed",
        styles={
            "background":  "#0f0f1a",
            "border-left": "2px solid #333",
            "overflow-y":  "auto",
            "color":       "#e0e0e0",
            "height":      "100vh",
        },
    )
 
    source.selected.js_on_change("indices", CustomJS(args=dict(src=source, div=div), code="""
        const i = src.selected.indices[0];
        if (i == null) return;
        const titres = JSON.parse(src.data.titres_json[i]);
        let html = `<div style="padding:16px;font-family:monospace">
            <b style="color:${src.data.couleur[i]};font-size:15px">${src.data.nom[i]}</b>
            <p style="color:#888;font-size:12px">${src.data.nb_art[i]} articles</p>
            <ol style="padding-left:18px">`;
        for (const t of titres)
            html += `<li style="color:#ccc;font-size:11px;margin-bottom:4px">${t}</li>`;
        html += `</ol></div>`;
        div.text = html;
    """))
 
    layout = row(p, div, sizing_mode="stretch_both") 
    save(layout)
    print(f"---- visualisation.html sauvegardé dans {OUTPUT_DIR}/")
