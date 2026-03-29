import os
import json
import math
import random
import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Div
from bokeh.layouts import row
from bokeh.palettes import Turbo256


OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MACRO_COLORS = [
    "#FF5E00", 
    "#EEC7A8",  
    "#FFB703",  
    "#FFFD6A",  
    "#6DDA2D",  
    "#0DBD47",  
    "#2A9D8F",  
    "#A8DADC", 
    "#3A86FF", 
    "#457B9D",  
    "#AA00FF",
    "#CC44FF",  
    "#94008D",
    "#F72585",  
    "#FF1744",
    "#FF5252",
]

# Permet de mieux répartir les bulles sur le layout afin d'éviter trop de chevauchements
def _force_layout_groupes(groupes, iterations=200):
    # Initialisation : sous-bulles placées aléatoirement dans leur macro
    random.seed(42)
    all_positions = []
    for g in groupes:
        positions = []
        for r in g["rayons"]:
            angle = random.uniform(0, 2 * math.pi)
            dist  = random.uniform(0, max(0, g["r_macro"] - r) * 0.6)
            positions.append([
                g["cx"] + dist * math.cos(angle),
                g["cy"] + dist * math.sin(angle),
            ])
        all_positions.append(positions)
 
    # Simulation de forces
    for _ in range(iterations):
        # Aplatir pour traiter toutes les bulles ensemble
        flat_pos    = [p for grp in all_positions for p in grp]
        flat_rayons = [r for g in groupes for r in g["rayons"]]
        flat_group  = [gi for gi, g in enumerate(groupes) for _ in g["rayons"]]
 
        n = len(flat_pos)
        dx = [0.0] * n
        dy = [0.0] * n
 
        # Répulsion entre sous-bulles
        for a in range(n):
            for b in range(a + 1, n):
                ddx  = flat_pos[a][0] - flat_pos[b][0]
                ddy  = flat_pos[a][1] - flat_pos[b][1]
                dist = math.sqrt(ddx**2 + ddy**2) or 0.001
                gap  = flat_rayons[a] + flat_rayons[b] + 20 # 20px de marge en + entre les petites bulles
                if dist < gap:
                    f = (gap - dist) / dist * 0.5
                    dx[a] += ddx * f;  dy[a] += ddy * f
                    dx[b] -= ddx * f;  dy[b] -= ddy * f
 
        # Rappel vers le centre du macro-groupe
        for i, (pos, gi) in enumerate(zip(flat_pos, flat_group)):
            g   = groupes[gi]
            ddx = pos[0] - g["cx"]
            ddy = pos[1] - g["cy"]
            dist = math.sqrt(ddx**2 + ddy**2) or 0.001
            r_sous = flat_rayons[i]
            limit  = g["r_macro"] - r_sous - 4
            # Si la bulle dépasse le bord de la macro, on la ramène
            if dist > limit and limit > 0:
                dx[i] -= ddx * 0.3
                dy[i] -= ddy * 0.3
            else:
                # Rappel vers le centre
                dx[i] -= ddx * 0.02
                dy[i] -= ddy * 0.02
 
        # Appliquer les déplacements
        idx = 0
        for grp in all_positions:
            for pos in grp:
                pos[0] += dx[idx]
                pos[1] += dy[idx]
                idx += 1
 
    result = []
    flat_pos = [p for grp in all_positions for p in grp]
    idx = 0
    for g in groupes:
        grp_result = []
        for _ in g["rayons"]:
            grp_result.append((flat_pos[idx][0], flat_pos[idx][1]))
            idx += 1
        result.append(grp_result)
 
    return result
 
# pour éviter les chevauchement des bulles :
def _place_macros(macro_noms, macro_rayons, iterations=200):
    random.seed(0)
    n  = len(macro_noms)
    xs = [random.uniform(-500, 500) for _ in range(n)]
    ys = [random.uniform(-500, 500) for _ in range(n)]
 
    for _ in range(iterations):
        dx = [0.0] * n
        dy = [0.0] * n
        for a in range(n):
            for b in range(a + 1, n):
                ddx  = xs[a] - xs[b]
                ddy  = ys[a] - ys[b]
                dist = math.sqrt(ddx**2 + ddy**2) or 0.001
                gap  = macro_rayons[a] + macro_rayons[b] + 60 # Gap de min 20px entre les grosses bulles
                if dist < gap:
                    f = (gap - dist) / dist * 0.5
                    dx[a] += ddx * f;  dy[a] += ddy * f
                    dx[b] -= ddx * f;  dy[b] -= ddy * f
        for a in range(n):
            dx[a] -= xs[a] * 0.008
            dy[a] -= ys[a] * 0.008
            xs[a] += dx[a]
            ys[a] += dy[a] 
    return xs, ys

def visualisation_chart(df, resume):
    output_file(f"{OUTPUT_DIR}/visualisation.html", title="Clustering des articles")
    
    # rajout des titres des articles de df dans résu :
    id_to_titre = df.set_index("id_article")["titre"].to_dict()
    resume["liste_titres"] = resume["liste_ids_articles"].apply(
        lambda ids: [id_to_titre[i] for i in ids if i in id_to_titre]
    )

    df_vis  = resume[resume["id_sujet"] != -1].reset_index(drop=True) # on retire les articles classé dans Bruit (-1)
    
    # groupes généraux + leur couleur: 
    macro_noms_uniques = list(df_vis["macro_sujet"].unique())
    n_macro = len(macro_noms_uniques)
    couleur_macro = {
        nom: MACRO_COLORS[i % len(MACRO_COLORS)]
        for i, nom in enumerate(macro_noms_uniques)
    }
    


    # Calcul des taille des bulles :  
    R_SOUS_MAX  = 70    # rayon max d'une spetite bulle
    R_MACRO_PAD = 25    # marge intérieure de la grande bulle 
    max_art_global = df_vis["nombre_articles"].max()
 
    # Rayon petites bulles = sqr(nb_articles)
    df_vis = df_vis.copy()
    df_vis["rayon_sous"] = df_vis["nombre_articles"].apply(
        lambda n: R_SOUS_MAX * math.sqrt(n / max_art_global)
    )
 
    # Rayon macro = estimé à partir de la somme des aires de ses sous-bulles
    macro_rayons_dict = {}
    for nom in macro_noms_uniques:
        scs = df_vis[df_vis["macro_sujet"] == nom]
        aire_totale = sum(math.pi * r**2 for r in scs["rayon_sous"])
        # Rayon tel que l'aire macro ≈ 2× la somme des aires des sous-bulles
        macro_rayons_dict[nom] = math.sqrt(2 * aire_totale / math.pi) + R_MACRO_PAD
 
    macro_rayons_list = [macro_rayons_dict[n] for n in macro_noms_uniques]


    # Position :
    # grosses bulles :
    macro_cx_list, macro_cy_list = _place_macros(macro_noms_uniques, macro_rayons_list)
    macro_centres = {
        nom: (macro_cx_list[i], macro_cy_list[i])
        for i, nom in enumerate(macro_noms_uniques)
    }
    # petites bulles :
    groupes_input = []
    for nom in macro_noms_uniques:
        scs = df_vis[df_vis["macro_sujet"] == nom]
        cx, cy = macro_centres[nom]
        groupes_input.append({
            "cx":      cx,
            "cy":      cy,
            "r_macro": macro_rayons_dict[nom],
            "rayons":  scs["rayon_sous"].tolist(),
        })
 
    positions_par_groupe = _force_layout_groupes(groupes_input)

    # grosses bulles : 
    macro_src = ColumnDataSource(dict(
        x    =[macro_centres[n][0] for n in macro_noms_uniques],
        y    =[macro_centres[n][1] for n in macro_noms_uniques],
        r    =macro_rayons_list,
        nom  =macro_noms_uniques,
        nb   =[int(df_vis[df_vis["macro_sujet"] == n]["nombre_articles"].sum())
               for n in macro_noms_uniques],
        couleur=[couleur_macro[n] for n in macro_noms_uniques],
    ))
    # petites bulles :
    xs_s, ys_s, rs_s, noms_s, nbs_s, couleurs_s, titres_json_s = \
        [], [], [], [], [], [], []
 
    for gi, nom_macro in enumerate(macro_noms_uniques):
        scs      = df_vis[df_vis["macro_sujet"] == nom_macro].reset_index(drop=True)
        positions = positions_par_groupe[gi]
        coul      = couleur_macro[nom_macro]
 
        for j, (_, s) in enumerate(scs.iterrows()):
            xs_s.append(positions[j][0])
            ys_s.append(positions[j][1])
            rs_s.append(s["rayon_sous"])
 
            parts = s["nom_sujet"].split("_")
            label = " / ".join(parts[1:4]) if len(parts) > 1 else s["nom_sujet"]
            noms_s.append(label)
            nbs_s.append(int(s["nombre_articles"]))
            couleurs_s.append(coul)
 
            ids    = s["liste_ids_articles"]
            titres = [id_to_titre[i] for i in ids if i in id_to_titre]
            titres_json_s.append(json.dumps(titres, ensure_ascii=False))
 
    sous_src = ColumnDataSource(dict(
        x=xs_s, y=ys_s, r=rs_s, nom=noms_s,
        nb_articles=nbs_s, couleur=couleurs_s, titres_json=titres_json_s,
    ))



    ### - - - -  Figures : - - - - 
    toutes_x = macro_cx_list + xs_s
    toutes_y = macro_cy_list + ys_s
    toutes_r = macro_rayons_list + rs_s
    lim = max(
        max(abs(x) + r for x, r in zip(toutes_x, toutes_r)),
        max(abs(y) + r for y, r in zip(toutes_y, toutes_r)),
    ) + 30
 
    p = figure(
        tools="pan,wheel_zoom,reset,tap",
        x_range=(-lim, lim),
        y_range=(-lim, lim),
        match_aspect=True,
        toolbar_location="above",
        title="Clusters d'articles — Transition énergétique",
        sizing_mode="stretch_both",
    )
    p.axis.visible          = False
    p.grid.visible          = False
    p.background_fill_color = "#0f0f1a"
    p.border_fill_color     = "#0f0f1a"
    p.outline_line_color    = None
    p.title.text_color      = "#e0e0e0"
    p.title.text_font_size  = "14px"
 
    # Grandes bulles (fond semi-transparent, en arrière-plan)
    p.circle(
        x="x", y="y", radius="r",
        fill_color="couleur", fill_alpha=0.12,
        line_color="couleur", line_alpha=0.60, line_width=2,
        source=macro_src,
        level="underlay",
    )
    # Label des grosses bulles (au centre)
    p.text(
        x="x", y="y", text="nom",
        text_color="white",
        text_font_size="18px",
        text_font_style="bold",
        text_align="center",
        text_baseline="middle",
        source=macro_src,
        level="overlay",
    )
 
    # Petites bulles (interactives)
    renderer = p.circle(
        x="x", y="y", radius="r",
        fill_color="couleur", fill_alpha=0.80,
        line_color="white",   line_alpha=0.25, line_width=0.8,
        hover_fill_alpha=1.0,
        nonselection_fill_alpha=0.35,
        selection_line_color="white", selection_line_width=2,
        source=sous_src,
    )
    p.add_tools(HoverTool(renderers=[renderer], tooltips="""
        <div style="font-family:monospace; padding:6px 10px;
                    background:#1a1a2e; border:1px solid #444; border-radius:6px;
                    white-space:nowrap">
            <span style="color:@couleur; font-size:15px; font-weight:bold">@nom</span><br>
            <span style="color:#888; font-size:11px">@nb_articles articles</span>
        </div>
    """))




    # - - - - HTML / JS de la page : -  - - - 
    div = Div(
        text="<p style='color:#777;font-style:italic;padding:16px;font-family:monospace'>"
             "Cliquez sur une bulle</p>",
        width=320,
        sizing_mode="stretch_height",
        styles={
            "background":  "#0f0f1a",
            "border-left": "2px solid #333",
            "overflow-y":  "auto",
            "color":       "#e0e0e0",
            "height":      "100vh",
        },
    )
 
    sous_src.selected.js_on_change("indices", CustomJS(args=dict(src=sous_src, div=div), code="""
        const i = src.selected.indices[0];
        if (i == null) return;
        const titres = JSON.parse(src.data.titres_json[i]);
        let html = `<div style="padding:16px;font-family:monospace">
            <b style="color:${src.data.couleur[i]};font-size:15px">${src.data.nom[i]}</b>
            <p style="color:#888;font-size:12px">${src.data.nb_articles[i]} articles</p>
            <ol style="padding-left:18px">`;
        for (const t of titres)
            html += `<li style="color:#ccc;font-size:11px;margin-bottom:4px">${t}</li>`;
        html += `</ol></div>`;
        div.text = html;
    """))
 
    layout = row(p, div, sizing_mode="stretch_both")
    save(layout)
    print(f"---- visualisation.html sauvegardé dans {OUTPUT_DIR}/")
